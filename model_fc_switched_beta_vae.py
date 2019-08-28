import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def gumbel_softmax_kl_divergence(y_logits):
    # 1/B sum_i{ - entroy(prob_i) + log(n) }
    log_p = F.log_softmax(y_logits, 1)  # B x n
    kl_i = (log_p.exp() * log_p).sum(1) + math.log(y_logits.size(1))  # B x n
    return torch.mean(kl_i)


class EncFCSwitch(nn.Module):
    def __init__(self, n_branches, n_sub_dims, n_dims):
        super(EncFCSwitch, self).__init__()
        self.fcs_in = nn.ModuleList([nn.Linear(n_dims, n_sub_dims) for _ in range(n_branches)])
        self.fcs_out = nn.ModuleList([nn.Linear(n_sub_dims, n_dims) for _ in range(n_branches)])
        self.fcs_switch = nn.ModuleList([nn.Linear(n_sub_dims, 1) for _ in range(n_branches)])

    def forward(self, x):
        hiddens = [fc_in(F.relu(x)) for fc_in in self.fcs_in]  # [B x n_sub_dims] * n_branches
        y_logits = torch.cat([fc_switch(hidden) for fc_switch, hidden in zip(self.fcs_switch, hiddens)], 1)  # B x n_branches
        y_index, _, y_hard = sample_from_gumbel_softmax(y_logits)  # y_index: B x 1, y_hard: B x n_branches
        outs = [fc_out(hidden) for fc_out, hidden in zip(self.fcs_out, hiddens)]  # [B x n_dims] * n_branches
        outs = torch.stack(outs, 1)  # B x n_branches x n_dims
        out = outs.gather(1, y_index.unsqueeze(2).repeat(1, 1, outs.size(2))).squeeze(1)  # B x n_dims
        if self.training:
            out = out * y_hard.gather(1, y_index)  # B x n_dims
        return x + out, y_logits, y_index, y_hard


class DecFCSwitch(nn.Module):
    def __init__(self, n_branches, n_sub_dims, n_dims):
        super(DecFCSwitch, self).__init__()
        self.fcs_in = nn.ModuleList([nn.Linear(n_dims, n_sub_dims) for _ in range(n_branches)])
        self.fcs_out = nn.ModuleList([nn.Linear(n_sub_dims, n_dims) for _ in range(n_branches)])
        self.fcs_switch = nn.ModuleList([nn.Linear(n_sub_dims, 1) for _ in range(n_branches)])

    def forward(self, x, y_index, y_hard=None):
        hiddens = [fc_in(F.relu(x)) for fc_in in self.fcs_in]  # [B x n_sub_dims] * n_branches
        outs = [fc_out(hidden) for fc_out, hidden in zip(self.fcs_out, hiddens)]  # [B x n_dims] * n_branches
        outs = torch.stack(outs, 1)  # B x n_branches x n_dims
        out = outs.gather(1, y_index.unsqueeze(2).repeat(1, 1, outs.size(2))).squeeze(1)  # B x n_dims
        if self.training and y_hard is not None:
            out = out * y_hard.gather(1, y_index)  # B x n_dims
        return x + out


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, n_latents):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, padding=1)  # 3 x 64 x 64 --4x4+2--> 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, padding=1)  # 32 x 32 x 32 --4x4+2--> 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 64, 4, 2, padding=1)  # 32 x 16 x 16 --4x4+2--> 64 x 8 x 8
        self.conv4 = nn.Conv2d(64, 64, 4, 2, padding=1)  # 64 x 8 x 8 --4x4+2--> 64 x 4 x 4
        self.fc = nn.Linear(1024, 192)
        self.switches = nn.ModuleList([EncFCSwitch(10, 10, 192) for _ in range(n_latents)])
        self.fc_mean = nn.Linear(192, n_latents)
        self.fc_logvar = nn.Linear(192, n_latents)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.reshape(-1, 1024)
        out = self.fc(out)

        ys_logits, ys_index, ys_hard = [], [], []
        for switch in self.switches:
            out, y_logits, y_index, y_hard = switch(out)
            ys_logits.append(y_logits)
            ys_index.append(y_index)
            ys_hard.append(y_hard)
        out = F.relu(out)

        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        return mean, logvar, ys_logits, ys_index, ys_hard


class DeconvDecoder(nn.Module):
    def __init__(self, out_channels, n_latents):
        super(DeconvDecoder, self).__init__()
        self.fc_latent = nn.Linear(n_latents, 192)
        self.switches = nn.ModuleList([DecFCSwitch(10, 10, 192) for _ in range(n_latents)])
        self.fc = nn.Linear(192, 1024)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, 2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, 4, 2, padding=1)

    def forward(self, z, ys_index, ys_hard=None):
        out = self.fc_latent(z)

        for i, switch in enumerate(self.switches):
            y_index = ys_index[i]
            y_hard = ys_hard[i] if ys_hard is not None else None
            out = switch(out, y_index, y_hard)
        out = F.relu(out)

        out = F.relu(self.fc(out))
        out = out.reshape(-1, 64, 4, 4)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = self.deconv4(out)
        return out


class FCSwitchedBetaVAE(nn.Module):
    def __init__(self, z_beta, y_beta, img_channels, n_latents):
        super(FCSwitchedBetaVAE, self).__init__()
        self.z_beta = z_beta
        self.y_beta = y_beta
        self.encoder = ConvEncoder(img_channels, n_latents)
        self.decoder = DeconvDecoder(img_channels, n_latents)

    def forward(self, x):
        z_mean, z_logvar, ys_logits, ys_index, ys_hard = self.encoder(x)
        z = sample_from_gaussian(z_mean, z_logvar)
        recon_x = self.decoder(z, ys_index, ys_hard)
        return recon_x, z, z_mean, z_logvar, ys_index, ys_hard, ys_logits

    def loss(self, x, recon_x, z_mean, z_logvar, ys_logits):
        recon_loss = bernoulli_reconstruction_loss(recon_x, x)
        z_kl_loss = gaussian_kl_divergence(z_mean, z_logvar)
        y_kl_loss = 0
        for y_logits in ys_logits:
            y_kl_loss += gumbel_softmax_kl_divergence(y_logits)
        loss = recon_loss + self.z_beta * z_kl_loss + self.y_beta * y_kl_loss
        neg_elbo = recon_loss + z_kl_loss + y_kl_loss
        return loss, neg_elbo, recon_loss, z_kl_loss, y_kl_loss
