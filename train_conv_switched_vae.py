import os
import argparse

import numpy as np
import torch
import torch.optim as optim

from data import get_data_loader
from model_conv_switched_vae import ConvSwitchedVAE
from utils import mkdir


SAVE_BASE = './checkpoints'
OUTPUT_BASE = './output'
VERSION = 'v2'

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--dataset', type=str, default='dsprites_full')
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--save_dir', type=str, default='conv_switched_vae')
parser.add_argument('--output_dir', type=str, default='conv_switched_vae')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_steps', type=int, default=300000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)

parser.add_argument('--y_ce_beta', type=int, default=1)
parser.add_argument('--y_phsic_beta', type=int, default=1)
parser.add_argument('--y_mmd_beta', type=int, default=1)
parser.add_argument('--z_beta', type=int, default=1)
parser.add_argument('--z2_beta', type=int, default=4)

parser.add_argument('--n_branches', type=int, default=3)
parser.add_argument('--backward_on_y_hard', action='store_true', default=False)
parser.add_argument('--use_batchnorm', action='store_true', default=False)

parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=10000)

args = parser.parse_args()
args.exp_id = 'exp-%d' % args.exp_id
args.cuda_id = 'cuda:%d' % args.cuda_id
args.save_dir = os.path.join(SAVE_BASE, args.save_dir, args.dataset, args.exp_id)
args.output_dir = os.path.join(OUTPUT_BASE, args.output_dir, args.dataset, args.exp_id)


def train():
    mkdir(args.save_dir)
    device = torch.device(args.cuda_id if torch.cuda.is_available() else 'cpu')

    train_loader, ds = get_data_loader(args.dataset, args.batch_size, args.n_steps)
    model = ConvSwitchedVAE(args.y_ce_beta, args.y_phsic_beta, args.y_mmd_beta, args.z_beta, args.z2_beta,
                            args.channels, args.n_branches, args.use_batchnorm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2))

    ckpt_paths = []
    model.train()

    for i, batch in enumerate(train_loader):
        _, inputs = batch
        x = inputs.to(device).float()
        recon_x, z2, z2_mean, z2_logvar, ys_logits, ys_logits_2, ys_index, ys_hard, zs_mean, zs_logvar, zs = \
            model(x, backward_on_y_hard=args.backward_on_y_hard)
        loss, recon_loss, z2_kl_loss, z_kl_loss, y_ce_loss, y_phsic_loss, y_mmd_loss = \
            model.loss(x, recon_x, z2_mean, z2_logvar, ys_logits, ys_logits_2, ys_hard, zs_mean, zs_logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step = i + 1
        if step % args.print_freq == 0:
            print('[Step %d] loss: %.4f, recon_loss: %.4f, z2_kl_loss: %.4f, z_kl_loss: %.4f, y_ce_loss: %.4f, '
                  'y_phsic_loss: %.4f, y_mmd_loss: %.4f' %
                  (step, loss, recon_loss, z2_kl_loss, z_kl_loss, y_ce_loss, y_phsic_loss, y_mmd_loss))
        if step % args.save_freq == 0:
            path = os.path.join(args.save_dir, 'step-%d.ckpt' % step)
            torch.save({'step': step, 'loss': loss, 'recon_loss': recon_loss,
                        'z2_kl_loss': z2_kl_loss,  'z_kl_loss': z_kl_loss,
                        'y_ce_loss': y_ce_loss, 'y_phsic_loss': y_phsic_loss, 'y_mmd_loss': y_mmd_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, path)
            ckpt_paths.append(path)
    return ckpt_paths


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train()
