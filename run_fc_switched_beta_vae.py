import os
import shutil

import numpy as np

import torch
import torch.optim as optim

from datasets import get_data_loader
from model_fc_switched_beta_vae import FCSwitchedBetaVAE
from utils import plot_images


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
SAVE_DIR = './checkpoints/fc_switched_beta_vae'
OUTPUT_DIR = './output/fc_switched_beta_vae'


def train(train_loader, model, optimizer, device, save_dir):
    ckpt_paths = []
    model.train()

    for i, batch in enumerate(train_loader):
        _, inputs = batch
        x = inputs.to(device).float()
        recon_x, z, z_mean, z_logvar, ys_index, ys_hard, ys_logits = model(x)
        loss, neg_elbo, recon_loss, z_kl_loss, y_kl_loss = model.loss(x, recon_x, z_mean, z_logvar, ys_logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step = i + 1
        if step % PRINT_FREQ == 0:
            print('[Step %d] loss: %.4f, neg_elbo: %.4f, recon_loss: %.4f, z_kl_loss: %.4f, y_kl_loss: %.4f' %
                  (step, loss, neg_elbo, recon_loss, z_kl_loss, y_kl_loss))
        if step % SAVE_FREQ == 0:
            path = os.path.join(save_dir, 'step-%d.ckpt' % step)
            torch.save({'step': step, 'loss': loss, 'neg_elbo': neg_elbo,
                        'recon_loss': recon_loss, 'z_kl_loss': z_kl_loss, 'y_kl_loss': y_kl_loss,
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
            recon_x, z, z_mean, z_logvar, ys_index, ys_hard, ys_logits = model(x)
            recon_x_from_mean = model.decoder(z_mean, ys_index)

            # traversal over z
            traversal = z_mean.new_tensor([-3, -2, -1, 0, 1, 2, 3])
            n_latents = z_mean.size(1)
            n_traversal = traversal.size(0)
            z_mean_traversal = z_mean.repeat(n_latents, n_traversal, 1)
            for d in range(n_latents):
                z_mean_traversal[d, :, d] = traversal
            z_mean_traversal = z_mean_traversal.reshape(-1, n_latents)
            recon_x_from_traversal = model.decoder(z_mean_traversal,
                                                   [y_index.repeat(z_mean_traversal.size(0), 1) for y_index in ys_index])

            images = torch.cat([recon_x_from_traversal, recon_x, recon_x_from_mean], 0).sigmoid()
            images = torch.cat([images, x], 0)
            visual_path = os.path.join(visual_dir, 'step-%d-image-%d-z.png' % (step, i))
            plot_images(images, n_latents + 1, n_traversal, visual_path)

            # traversal over y
            yss_index = []
            n_latents = len(ys_index)
            n_branches = ys_hard[0].size(1)
            for d in range(n_latents):
                for j in range(n_branches):
                    ys_index_copy = [y_index.clone() for y_index in ys_index]
                    ys_index_copy[d] = j
                    yss_index.append(ys_index_copy)
            ys_index_traversal = [torch.cat([ys_index[d] for ys_index in yss_index], 0)
                                  for d in range(n_branches)]
            recon_x_from_traversal = model.decoder(z_mean.repeat(ys_index_traversal[0].size(0), 1),
                                     ys_index_traversal)

            images = torch.cat([recon_x_from_traversal, recon_x, recon_x_from_mean], 0).sigmoid()
            images = torch.cat([images, x], 0)
            visual_path = os.path.join(visual_dir, 'step-%d-image-%d-y.png' % (step, i))
            plot_images(images, n_latents + 1, n_branches, visual_path)


def run(z_beta=10, y_beta=10, seed=1234):
    save_dir = os.path.join(SAVE_DIR, DATASET_NAME)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, ds = get_data_loader(DATASET_NAME, BATCH_SIZE, N_STEPS)
    model = FCSwitchedBetaVAE(z_beta, y_beta, IMG_CHANNELS, N_LATENTS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(ADAM_BETA1, ADAM_BETA2))
    ckpt_paths = train(train_loader, model, optimizer, device, save_dir)

    visual_dir = os.path.join(OUTPUT_DIR, DATASET_NAME, 'visual')
    if os.path.exists(visual_dir):
        shutil.rmtree(visual_dir)
    os.makedirs(visual_dir)

    for path in ckpt_paths:
        eval_loader, _ = get_data_loader(DATASET_NAME, 1, 100)
        eval_visual(eval_loader, model, device, path, visual_dir)

def run_eval_visual(z_beta=10, y_beta=10, seed=1234):
    save_dir = os.path.join(SAVE_DIR, DATASET_NAME)

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FCSwitchedBetaVAE(z_beta, y_beta, IMG_CHANNELS, N_LATENTS).to(device)

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
    run(z_beta=100, y_beta=5)

    #run_eval_visual()