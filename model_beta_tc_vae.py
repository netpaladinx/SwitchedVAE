import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_total_correlation(z, z_mean, z_logvar):
    # samples: (x_i, z_i), i = 1, ..., B
    # E_z[log q(z)] ~= 1/B sum_i{ log(sum_j q(z_i|x_j)) - log(N*B) }
    #   q(z_i|x_j) = prod_d q(z_i^d|x_j)
    # E_z[log prod_d q(z^d)] ~= 1/B sum_i{ sum_d log(sum_j q(z_i^d|x_j)) - log(N*B) }

    # compute log(q(z_i^d|x_j))
    diff = z.unsqueeze(1) - z_mean.unsqueeze(0)  # B (for i) x B (for j) x n_latents
    inv_sigma = torch.exp(-z_logvar.unsqueeze(0))  # 1 (for i) x B (for j) x n_latents
    normalization = math.log(2 * math.pi)  # 1
    log_q_zi_d_xj = -0.5 * (diff * diff * inv_sigma + z_logvar + normalization)  # B (for i) x B (for j) x n_latents

    # compute log q(z_i^d) = log(sum_j q(z_i^d|x_j) - log(N*B) (the constant term ignored)
    log_q_zi_d = log_q_zi_d_xj.logsumexp(1)  # B (for i) x n_latent

    # compute i.i.d. log q(z_i) = log prod_d q(z_i^d) = sum log q(z_i^d)
    log_q_zi_iid = log_q_zi_d.sum(1)  # B (for i)

    # compute log q(z_i|x_j) = log(prod_d q(z_i^d|x_j))
    log_q_zi_xj = log_q_zi_d_xj.sum(2)  # B (for i) x B (for j)

    # compute log q(z_i) = log(sum_j q(z_i|x_j)) - log(N*B) (the constant term ignored)
    log_q_zi = log_q_zi_xj.logsumexp(1)  # B (for i)

    # compute 1/B sum_i{log q(z_i) - log q(z_i)_iid}
    return torch.mean(log_q_zi - log_q_zi_iid)  # 1


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


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, n_latents):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, padding=2)  # 3 x 64 x 64 --4x4+2--> 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, padding=2)  # 32 x 32 x 32 --4x4+2--> 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 64, 4, 2, padding=2)  # 32 x 16 x 16 --4x4+2--> 64 x 8 x 8
        self.conv4 = nn.Conv2d(64, 64, 4, 2, padding=2)  # 64 x 8 x 8 --4x4+2--> 64 x 4 x 4
        self.fc = nn.Linear(1024, 256)
        self.fc_mean = nn.Linear(256, n_latents)
        self.fc_logvar = nn.Linear(256, n_latents)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.reshape(-1, 1024)
        out = F.relu(self.fc(out))
        mean = self.fc_mean(out)
        logvar = self.fc_logvar(out)
        return mean, logvar


class DeconvDecoder(nn.Module):
    def __init__(self, out_channels, n_latents):
        super(DeconvDecoder, self).__init__()
        self.fc_latent = nn.Linear(n_latents, 256)
        self.fc = nn.Linear(256, 1024)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, 2, padding=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, 2, padding=2)

    def forward(self, z):
        out = F.relu(self.fc_latent(z))
        out = F.relu(self.fc(out))
        out = out.reshape(-1, 64, 4, 4)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = self.deconv4(out)
        return out


class BetaTCVAE(nn.Module):
    def __init__(self, beta, img_channels, n_latents):
        super(BetaTCVAE, self).__init__()
        self.beta = beta
        self.encoder = ConvEncoder(img_channels, n_latents)
        self.decoder = DeconvDecoder(img_channels, n_latents)

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = sample_from_gaussian(z_mean, z_logvar)
        recon_x = self.decoder(z)
        return recon_x, z, z_mean, z_logvar

    def loss(self, recon_x, x, z, z_mean, z_logvar):
        recon_loss = bernoulli_reconstruction_loss(recon_x, x)
        kl_loss = gaussian_kl_divergence(z_mean, z_logvar)
        tc_loss = gaussian_total_correlation(z, z_mean, z_logvar)
        loss = recon_loss + kl_loss + (self.beta - 1) * tc_loss
        elbo = recon_loss + kl_loss
        return loss, elbo


class RepresentationExtractor(nn.Module):
    def __init__(self):
        super(RepresentationExtractor, self).__init__()

