import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.optim as optim

from datasets import get_data_loader
from model_beta_vae import BetaVAE
from helper import trim_axs


BATCH_SIZE = 64
N_STEPS = 300000
N_LATENTS = 10
LEARNING_RATE = 0.0001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999

DATASET_NAME = 'dsprites_full'
IMG_CHANNELS = 1

PRINT_FREQ = 100
SAVE_FREQ = 10000
SAVE_DIR = './checkpoints/beta_vae_v2'
OUTPUT_DIR = './output/beta_vae_v2'


def train(train_loader, model, optimizer, device, save_dir):
    ckpt_paths = []
    model.train()

    for i, batch in enumerate(train_loader):
        _, inputs = batch
        x = inputs.to(device).float()
        recon_x, z, z_mean, z_logvar = model(x)
        loss, neg_elbo, recon_loss, kl_loss = model.loss(x, recon_x, z_mean, z_logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step = i + 1
        if step % PRINT_FREQ == 0:
            print('[Step %d] loss: %.4f, neg_elbo: %.4f, recon_loss: %.4f, kl_loss: %.4f' %
                  (step, loss, neg_elbo, recon_loss, kl_loss))
        if step % SAVE_FREQ == 0:
            path = os.path.join(save_dir, 'step-%d.ckpt' % step)
            torch.save({'step': step, 'loss': loss, 'neg_elbo': neg_elbo,
                        'recon_loss': recon_loss, 'kl_loss': kl_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, path)
            ckpt_paths.append(path)
    return ckpt_paths


def eval_visual(eval_loader, model, device, path, visual_dir):
    checkpoint = torch.load(path)
    step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(path)
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            _, inputs = batch
            x = inputs.to(device).float()
            recon_x, z, z_mean, z_logvar = model(x)
            recon_x_from_mean = model.decoder(z_mean)

            traversal = z_mean.new_tensor([-3, -2, -1, 0, 1, 2, 3])
            n_latents = z_mean.size(1)
            n_traversal = traversal.size(0)
            z_mean_traversal = z_mean.repeat(n_latents, n_traversal, 1)
            for d in range(n_latents):
                z_mean_traversal[d, :, d] = traversal

            z_mean_traversal = z_mean_traversal.reshape(-1, n_latents)
            recon_x_from_traversal = model.decoder(z_mean_traversal)

            images = torch.cat([recon_x_from_traversal, recon_x, recon_x_from_mean], 0).sigmoid()
            images = torch.cat([images, x], 0)

            fig, axs = plt.subplots(n_latents + 1, n_traversal)
            axs = trim_axs(axs, images.size(0))
            for ax, image in zip(axs, images):
                image = np.squeeze(np.moveaxis(image.cpu().numpy(), 0, 2))
                ax.imshow(image, cmap=cm.get_cmap('gray') if len(image.shape) == 2 else None)
                ax.set_axis_off()
            visual_path = os.path.join(visual_dir, 'step-%d-image-%d.png' % (step, i))
            fig.savefig(visual_path)
            plt.close(fig)
            print(visual_path)


def run(beta=50, seed=1234):
    save_dir = os.path.join(SAVE_DIR, DATASET_NAME)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, ds = get_data_loader(DATASET_NAME, BATCH_SIZE, N_STEPS)
    model = BetaVAE(beta, IMG_CHANNELS, N_LATENTS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA1, ADAM_BETA2))
    ckpt_paths = train(train_loader, model, optimizer, device, save_dir)

    visual_dir = os.path.join(OUTPUT_DIR, DATASET_NAME, 'visual')
    if os.path.exists(visual_dir):
        shutil.rmtree(visual_dir)
    os.makedirs(visual_dir)

    for path in ckpt_paths:
        eval_loader, _ = get_data_loader(DATASET_NAME, 1, 100)
        eval_visual(eval_loader, model, device, path, visual_dir)

def run_eval_visual(beta=10, seed=1234):
    save_dir = os.path.join(SAVE_DIR, DATASET_NAME)

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BetaVAE(beta, IMG_CHANNELS, N_LATENTS).to(device)

    visual_dir = os.path.join(OUTPUT_DIR, DATASET_NAME, 'visual')
    if os.path.exists(visual_dir):
        shutil.rmtree(visual_dir)
    os.makedirs(visual_dir)

    for path in os.listdir(save_dir):
        path = os.path.join(save_dir, path)
        if path[-5:] == '.ckpt':
            eval_loader, _ = get_data_loader(DATASET_NAME, 1, 100)
            eval_visual(eval_loader, model, device, path, visual_dir)

if __name__ == '__main__':
    #beta_choices = [1, 2, 4, 6, 8, 16]
    run()

    #run_eval_visual()