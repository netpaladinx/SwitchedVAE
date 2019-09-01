import math

import torch
import torch.nn as nn
import torch.nn.functional as F


import utils as U
import nn as N


class EncSwitchedFC(nn.Module):
    def __init__(self, n_dims, n_dims_sm, n_branches, fc_operator_type, fc_switch_type):
        super(EncSwitchedFC, self).__init__()
        self.n_branches = n_branches
        self.fc_operators = nn.ModuleList([N.get_fc_operator(fc_operator_type, n_dims, n_dims_sm)
                                           for _ in range(n_branches)])
        self.switch = N.get_fc_switch(fc_switch_type, n_dims, n_dims_sm, n_branches)

    def forward(self, x, backward_on_y=False):
        ''' x: B x n_dims
        '''
        ctrl = self.switch(x).reshape(-1, 3, self.n_branches)  # B x 3 x n_branches
        y_logits, z_mean, z_logvar = ctrl.unbind(1)  # all: B x n_branches
        y_idx, _, y = N.sample_from_gumbel_softmax(y_logits)  # y_idx: B x 1, y: B x n_branches
        z = N.sample_from_gaussian(z_mean, z_logvar)  # B x n_branches
        out = N.compute_fc_operators(self.fc_operators, self.n_branches, x, y_idx, y, z,
                                     self.training and backward_on_y)  # B x n_dims
        return x + out, y_logits, y_idx, y, z_mean, z_logvar, z


class DecSwitchedFC(nn.Module):
    def __init__(self, n_dims, n_dims_sm, n_branches, fc_operator_type):
        super(DecSwitchedFC, self).__init__()
        self.n_branches = n_branches
        self.fc_operators = nn.ModuleList([N.get_fc_operator(fc_operator_type, n_dims, n_dims_sm)
                                  for _ in range(n_branches)])

    def forward(self, x, y_idx, y, z, backward_on_y=False):
        ''' x: B x n_dims
            y_idx: B x 1
            y: B x n_branches
            z: B x n_branches
        '''
        out = N.compute_fc_operators(self.fc_operators, self.n_branches, x, y_idx, y, z,
                                     self.training and backward_on_y)
        return x + out


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, n_branches, n_switches, n_dims_sm, fc_operator_type, fc_switch_type, n_latent_z2):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, padding=1)  # 3 x 64 x 64 --4x4+2--> 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, padding=1)  # 32 x 32 x 32 --4x4+2--> 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 64, 4, 2, padding=1)  # 32 x 16 x 16 --4x4+2--> 64 x 8 x 8
        self.conv4 = nn.Conv2d(64, 64, 4, 2, padding=1)  # 64 x 8 x 8 --4x4+2--> 64 x 4 x 4
        self.fc = nn.Linear(1024, 256)
        self.fc_switched = nn.ModuleList([EncSwitchedFC(256, n_dims_sm, n_branches, fc_operator_type, fc_switch_type)
                                          for _ in range(n_switches)])
        self.fc_mean = nn.Linear(256, n_latent_z2)
        self.fc_logvar = nn.Linear(256, n_latent_z2)

    def forward(self, x, backward_on_y=False):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.reshape(-1, 1024)
        out = self.fc(out)

        ys_logits, ys_idx, ys, zs_mean, zs_logvar, zs = [], [], [], [], [], []
        ys_logits2 = []
        out2 = out
        for switch in self.fc_switched:
            out, y_logits, y_idx, y, z_mean, z_logvar, z = switch(out, backward_on_y=backward_on_y)
            U.append((ys_logits, ys_idx, ys, zs_mean, zs_logvar, zs),
                     (y_logits, y_idx, y, z_mean, z_logvar, z))

            if self.training:
                out2, y_logits2, _, _, _, _, _ = switch(out2, backward_on_y=backward_on_y)
                ys_logits2.append(y_logits2)
        out = F.relu(out)

        z2_mean = self.fc_mean(out)
        z2_logvar = self.fc_logvar(out)
        return z2_mean, z2_logvar, ys_logits, ys_logits2, ys_idx, ys, zs_mean, zs_logvar, zs


class DeconvDecoder(nn.Module):
    def __init__(self, out_channels, n_branches, n_switches, n_dims_sm, fc_operator_type, n_latent_z2):
        super(DeconvDecoder, self).__init__()
        self.fc_latent = nn.Linear(n_latent_z2, 256)
        self.fc_switched = nn.ModuleList([DecSwitchedFC(256, n_dims_sm, n_branches, fc_operator_type)
                                          for _ in range(n_switches)])
        self.fc = nn.Linear(256, 1024)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, 2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, 4, 2, padding=1)

    def forward(self, z2, ys_idx, ys, zs, backward_on_y=False):
        out = self.fc_latent(z2)

        i = len(ys_idx) - 1
        for switch in self.fc_switched:
            y_idx = ys_idx[i]
            y = ys[i] if ys is not None else None
            z = zs[i]
            out = switch(out, y_idx, y, z, backward_on_y=backward_on_y)
            i -= 1
        out = F.relu(out)

        out = F.relu(self.fc(out))
        out = out.reshape(-1, 64, 4, 4)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = self.deconv4(out)
        return out


class FCSwitchedVAE(nn.Module):
    def __init__(self, y_ce_beta, y_hsic_beta, y_mmd_beta, z_hsic_beta, z_kl_beta, z2_kl_beta_max, z2_kl_stop_step,
                 channels, n_branches, n_switches, n_dims_sm, fc_operator_type, fc_switch_type, n_latent_z2):
        super(FCSwitchedVAE, self).__init__()
        self.y_ce_beta = y_ce_beta
        self.y_hsic_beta = y_hsic_beta
        self.y_mmd_beta = y_mmd_beta
        self.z_hsic_beta = z_hsic_beta
        self.z_kl_beta = z_kl_beta
        self.z2_kl_beta_max = z2_kl_beta_max
        self.z2_kl_stop_step = z2_kl_stop_step
        self.n_branches = n_branches
        self.encoder = ConvEncoder(channels, n_branches, n_switches, n_dims_sm, fc_operator_type, fc_switch_type, n_latent_z2)
        self.decoder = DeconvDecoder(channels, n_branches, n_switches, n_dims_sm, fc_operator_type, n_latent_z2)

    def forward(self, x, backward_on_y=False):
        z2_mean, z2_logvar, ys_logits, ys_logits2, ys_idx, ys, zs_mean, zs_logvar, zs = \
            self.encoder(x, backward_on_y=backward_on_y)
        z2 = N.sample_from_gaussian(z2_mean, z2_logvar)
        recon_x = self.decoder(z2, ys_idx, ys, zs, backward_on_y=backward_on_y)
        return recon_x, z2, z2_mean, z2_logvar, ys_logits, ys_logits2, ys_idx, ys, zs_mean, zs_logvar, zs

    def loss(self, x, recon_x, z2_mean, z2_logvar, ys_logits, ys_logits2, ys, zs_mean, zs_logvar, zs, step):
        recon_loss = N.bernoulli_reconstruction_loss(recon_x, x)

        z2_kl_loss = N.gaussian_kl_divergence(z2_mean, z2_logvar)

        z_kl_loss = 0
        for z_mean, z_logvar in zip(zs_mean, zs_logvar):
            z_kl_loss += N.gaussian_kl_divergence(z_mean, z_logvar)

        z_hsic_loss = 0
        for i1, z1 in enumerate(zs):
            for i2, z2 in enumerate(zs):
                if i1 != i2:
                    z_hsic_loss += N.pairwise_gaussian_hsic(z1, z2)

        y_ce_loss = 0
        for y_logits, y_logits_2 in zip(ys_logits, ys_logits2):
            y_ce_loss += N.conditional_entropy_upper_bound(y_logits, y_logits_2)

        y_hsic_loss = 0
        for i1, y1 in enumerate(ys):
            for i2, y2 in enumerate(ys):
                if i1 != i2:
                    y_hsic_loss += N.pairwise_categorical_hsic(y1, y2)

        y_mmd_loss = 0
        for y in ys:
            y_mmd_loss += N.categorical_mmd_from_uniform(y, self.n_branches)

        if self.z2_kl_stop_step <= 0:
            z2_kl_beta = self.z2_kl_beta_max
        else:
            z2_kl_beta = (self.z2_kl_beta_max - 1) * min(1, step / self.z2_kl_stop_step) + 1
        loss = (recon_loss +
                z2_kl_beta * z2_kl_loss +
                self.z_kl_beta * z_kl_loss +
                self.z_hsic_beta * z_hsic_loss +
                self.y_ce_beta * y_ce_loss +
                self.y_hsic_beta * y_hsic_loss +
                self.y_mmd_beta * y_mmd_loss)

        return loss, recon_loss, z2_kl_loss, z_hsic_loss, z_kl_loss, y_ce_loss, y_hsic_loss, y_mmd_loss
