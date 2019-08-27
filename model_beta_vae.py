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


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, n_latents):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, padding=1)  # 3 x 64 x 64 --4x4+2--> 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, padding=1)  # 32 x 32 x 32 --4x4+2--> 32 x 16 x 16
        self.conv3 = nn.Conv2d(32, 64, 4, 2, padding=1)  # 32 x 16 x 16 --4x4+2--> 64 x 8 x 8
        self.conv4 = nn.Conv2d(64, 64, 4, 2, padding=1)  # 64 x 8 x 8 --4x4+2--> 64 x 4 x 4
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
        self.deconv1 = nn.ConvTranspose2d(64, 64, 4, 2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, 4, 2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, 4, 2, padding=1)

    def forward(self, z):
        out = F.relu(self.fc_latent(z))
        out = F.relu(self.fc(out))
        out = out.reshape(-1, 64, 4, 4)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = self.deconv4(out)
        return out


class BetaVAE(nn.Module):
    def __init__(self, beta, img_channels, n_latents):
        super(BetaVAE, self).__init__()
        self.beta = beta
        self.encoder = ConvEncoder(img_channels, n_latents)
        self.decoder = DeconvDecoder(img_channels, n_latents)

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = sample_from_gaussian(z_mean, z_logvar)
        recon_x = self.decoder(z)
        return recon_x, z, z_mean, z_logvar

    def loss(self, x, recon_x, z_mean, z_logvar):
        recon_loss = bernoulli_reconstruction_loss(recon_x, x)
        kl_loss = gaussian_kl_divergence(z_mean, z_logvar)
        loss = recon_loss + self.beta * kl_loss
        neg_elbo = recon_loss + kl_loss
        return loss, neg_elbo, recon_loss, kl_loss
