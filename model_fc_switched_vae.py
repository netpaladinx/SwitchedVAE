import math

import torch
import torch.nn as nn
import torch.nn.functional as F


import utils as U


def gaussian_kl_divergence(z_mean, z_logvar):
    # 1/B sum_i{ 1/2 * sum_d{ mu_d^2 + sigma_d^2 - log(sigma_d^2) - 1 } }
    kl_i = 0.5 * torch.sum(z_mean * z_mean + z_logvar.exp() - z_logvar - 1, 1)  # B
    return torch.mean(kl_i)


def sample_from_gaussian(z_mean, z_logvar):
    noise = torch.randn_like(z_mean)
    return noise * (z_logvar / 2).exp() + z_mean


def bernoulli_reconstruction_loss(recon_x, x):
    bs = x.size(0)
    x = x.reshape(bs, -1)
    recon_x = recon_x.reshape(bs, -1)
    loss_per_sample = F.binary_cross_entropy_with_logits(recon_x, x, reduction='none').sum(1)
    clamped_x = x.clamp(1e-6, 1-1e-6)
    loss_lower_bound = F.binary_cross_entropy(clamped_x, x, reduction='none').sum(1)
    loss = (loss_per_sample - loss_lower_bound).mean()
    return loss


def sample_from_gumbel_softmax(y_logits, tau=1):
    dim = 1
    gumbels = - torch.log(torch.empty_like(y_logits).exponential_() + 1e-20)  # ~ Gumbel(0,1)
    gumbels = (y_logits + gumbels) / tau  # ~ Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    _, y_index = y_soft.max(dim, keepdim=True)
    y_hard = torch.zeros_like(y_logits).scatter_(dim, y_index, 1.0)
    y_hard = y_hard - y_soft.detach() + y_soft
    return y_index, y_soft, y_hard


def categorical_mmd_from_uniform(y_hard, n):
    B = y_hard.size(0)
    gram = y_hard.mm(y_hard.t())  # B x B
    mmd = gram.sum() / B ** 2 - 1 / n
    return mmd


def pairwise_categorical_hsic(y_hard_1, y_hard_2):
    B = y_hard_1.size(0)
    gram1 = y_hard_1.mm(y_hard_1.t())  # B x B
    gram2 = y_hard_2.mm(y_hard_2.t())  # B x B
    hsic = ((gram1 * gram2).sum() / B ** 2
            + gram1.flatten().unsqueeze(1).mm(gram2.flatten().unsqueeze(0)).sum() / B ** 4
            - 2 * gram1.mm(gram2).sum() / B ** 3)
    return hsic


def conditional_entropy_upper_bound(y_logits, y_logits_2):
    p = F.softmax(y_logits, 1)  # B x n
    log_p_2 = F.log_softmax(y_logits_2, 1)  # B x n
    return (- p * log_p_2).sum(1).mean()


class EncSwitchedFC(nn.Module):
    def __init__(self, n_dims, n_dims_sm, n_branches):
        super(EncSwitchedFC, self).__init__()
        self.n_branches = n_branches
        self.fcs = nn.ModuleList([nn.Sequential(nn.Linear(n_dims, n_dims_sm),
                                                nn.ReLU(),
                                                nn.Linear(n_dims_sm, n_dims))
                                  for _ in range(n_branches)])
        self.switch = nn.Linear(n_dims, 3 * n_branches)

    def forward(self, x, backward_on_y_hard=False):
        out = F.relu(x)  # B x n_dims

        ctrl = self.switch(out).reshape(-1, 3, self.n_branches)  # B x 3 x n_branches
        y_logits, z_mean, z_logvar = ctrl.unbind(1)  # B x n_branches
        y_index, _, y_hard = sample_from_gumbel_softmax(y_logits)  # y_index: B x 1, y_hard: B x n_branches
        z = sample_from_gaussian(z_mean, z_logvar)  # B x n_branches

        sp_sum = None
        for i in range(self.n_branches):
            sub_index = y_index.squeeze(1).eq(i).nonzero().squeeze(1)  # sub_B
            sub_z = z[:, i].index_select(0, sub_index).unsqueeze(1)  # sub_B x 1
            sub_out = out.index_select(0, sub_index)  # sub_B x n_dims
            sub_out = self.fcs[i](sub_out) * sub_z   # sub_B x n_dims
            if self.training and backward_on_y_hard:
                sub_y_hard = y_hard[:, i].index_select(0, sub_index).unsqueeze(1)  # sub_B x 1
                sub_out = sub_out * sub_y_hard  # sub_B x n_dims

            sp_out = torch.sparse_coo_tensor(sub_index.unsqueeze(0), sub_out, out.size())  # (sparse) B x n_dims
            sp_sum = sp_out if sp_sum is None else sp_sum + sp_out
        sp_sum = sp_sum.to_dense()  # B x n_dims

        z_mean = z_mean.gather(1, y_index)  # B x 1
        z_logvar = z_logvar.gather(1, y_index)  # B x 1
        z = z.gather(1, y_index)  # B x 1

        return x + sp_sum, y_logits, y_index, y_hard, z_mean, z_logvar, z


class DecSwitchedFC(nn.Module):
    def __init__(self, n_dims, n_dims_sm, n_branches):
        super(DecSwitchedFC, self).__init__()
        self.n_branches = n_branches
        self.fcs = nn.ModuleList([nn.Sequential(nn.Linear(n_dims, n_dims_sm),
                                                nn.ReLU(),
                                                nn.Linear(n_dims_sm, n_dims))
                                  for _ in range(n_branches)])

    def forward(self, x, y_index, y_hard, z, backward_on_y_hard=False):
        out = F.relu(x)  # B x n_dims

        sp_sum = None
        for i in range(self.n_branches):
            sub_index = y_index.squeeze(1).eq(i).nonzero().squeeze(1)  # sub_B
            sub_z = z.index_select(0, sub_index)  # sub_B x 1
            sub_out = out.index_select(0, sub_index)  # sub_B x n_dims
            sub_out = self.fcs[i](sub_out) * sub_z
            if self.training and (y_hard is not None) and backward_on_y_hard:
                sub_y_hard = y_hard[:, i].index_select(0, sub_index).unsqueeze(1)  # sub_B x 1
                sub_out = sub_out * sub_y_hard  # sub_B x n_dims

            sp_out = torch.sparse_coo_tensor(sub_index.unsqueeze(0), sub_out, out.size())  # (sparse) B x n_dims
            sp_sum = sp_out if sp_sum is None else sp_sum + sp_out
        sp_sum = sp_sum.to_dense()  # B x n_dims

        return x + sp_sum


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, n_dims_sm, n_branches, n_switches):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, padding=1)  # 3 x 64 x 64 --4x4+2--> 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, padding=1)  # 32 x 32 x 32 --4x4+2--> 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 64, 4, 2, padding=1)  # 32 x 16 x 16 --4x4+2--> 64 x 8 x 8
        self.conv4 = nn.Conv2d(64, 64, 4, 2, padding=1)  # 64 x 8 x 8 --4x4+2--> 64 x 4 x 4
        self.switches = nn.ModuleList([EncSwitchedFC(1024, n_dims_sm, n_branches) for _ in range(n_switches)])
        self.fc_mean = nn.Linear(1024, 10)
        self.fc_logvar = nn.Linear(1024, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.conv4(out)
        out = out.reshape(-1, 1024)

        ys_logits, ys_index, ys_hard, zs_mean, zs_logvar, zs = [], [], [], [], [], []
        ys_logits_2 = []
        out2 = out
        for switch in self.switches:
            out, y_logits, y_index, y_hard, z_mean, z_logvar, z = switch(out)
            ys_logits.append(y_logits)
            ys_index.append(y_index)
            ys_hard.append(y_hard)
            zs_mean.append(z_mean)
            zs_logvar.append(z_logvar)
            zs.append(z)

            if self.training:
                out2, y_logits_2, _, _, _, _, _ = switch(out2)
                ys_logits_2.append(y_logits_2)
        out = F.relu(out)

        z2_mean = self.fc_mean(out)
        z2_logvar = self.fc_logvar(out)
        return z2_mean, z2_logvar, ys_logits, ys_logits_2, ys_index, ys_hard, zs_mean, zs_logvar, zs


class DeconvDecoder(nn.Module):
    def __init__(self, out_channels, n_dims_sm, n_branches, n_switches):
        super(DeconvDecoder, self).__init__()
        self.fc_latent = nn.Linear(10, 1024)
        self.switches = nn.ModuleList([DecSwitchedFC(1024, n_dims_sm, n_branches) for _ in range(n_switches)])
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, 2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, 4, 2, padding=1)

    def forward(self, z2, ys_index, ys_hard, zs):
        out = self.fc_latent(z2)

        for i, switch in enumerate(self.switches):
            y_index = ys_index[i]
            y_hard = ys_hard[i] if ys_hard is not None else None
            z = zs[i]
            out = switch(out, y_index, y_hard, z)
        out = F.relu(out)

        out = out.reshape(-1, 64, 4, 4)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = self.deconv4(out)
        return out


class FCSwitchedVAE(nn.Module):
    def __init__(self, y_ce_beta, y_phsic_beta, y_mmd_beta, z_beta, z2_beta, channels, n_dims_sm, n_branches, n_switches):
        super(FCSwitchedVAE, self).__init__()
        self.y_ce_beta = y_ce_beta
        self.y_phsic_beta = y_phsic_beta
        self.y_mmd_beta = y_mmd_beta
        self.z_beta = z_beta
        self.z2_beta = z2_beta
        self.n_branches = n_branches
        self.encoder = ConvEncoder(channels, n_dims_sm, n_branches, n_switches)
        self.decoder = DeconvDecoder(channels, n_dims_sm, n_branches, n_switches)

    def forward(self, x):
        z2_mean, z2_logvar, ys_logits, ys_logits_2, ys_index, ys_hard, zs_mean, zs_logvar, zs = self.encoder(x)
        z2 = sample_from_gaussian(z2_mean, z2_logvar)
        recon_x = self.decoder(z2, ys_index, ys_hard, zs)
        return recon_x, z2, z2_mean, z2_logvar, ys_logits, ys_logits_2, ys_index, ys_hard, zs_mean, zs_logvar, zs

    def loss(self, x, recon_x, z2_mean, z2_logvar, ys_logits, ys_logits_2, ys_hard, zs_mean, zs_logvar):
        recon_loss = bernoulli_reconstruction_loss(recon_x, x)

        z2_kl_loss = gaussian_kl_divergence(z2_mean, z2_logvar)

        z_kl_loss = 0
        for z_mean, z_logvar in zip(zs_mean, zs_logvar):
            z_kl_loss += gaussian_kl_divergence(z_mean, z_logvar)

        y_ce_loss = 0
        for y_logits, y_logits_2 in zip(ys_logits, ys_logits_2):
            y_ce_loss += conditional_entropy_upper_bound(y_logits, y_logits_2)

        y_phsic_loss = 0
        for i1, y_hard_1 in enumerate(ys_hard):
            for i2, y_hard_2 in enumerate(ys_hard):
                if i1 != i2:
                    y_phsic_loss += pairwise_categorical_hsic(y_hard_1, y_hard_2)

        y_mmd_loss = 0
        for y_hard in ys_hard:
            y_mmd_loss += categorical_mmd_from_uniform(y_hard, self.n_branches)

        loss = (recon_loss +
                self.z2_beta * z2_kl_loss +
                self.z_beta * z_kl_loss +
                self.y_ce_beta * y_ce_loss +
                self.y_phsic_beta * y_phsic_loss +
                self.y_mmd_beta * y_mmd_loss)

        return loss, recon_loss, z2_kl_loss, z_kl_loss, y_ce_loss, y_phsic_loss, y_mmd_loss
