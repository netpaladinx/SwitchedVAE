import os
import argparse

import numpy as np
import torch
import torch.optim as optim

from data import get_data_loader
from model_fc_switched_vae import FCSwitchedVAE
from utils import mkdir


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='fc_switched_vae')
parser.add_argument('--version', type=str, default='v2')
parser.add_argument('--exp_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--dataset', type=str, default='dsprites_full')
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--save_dir', type=str, default='./checkpoints')
parser.add_argument('--output_dir', type=str, default='./output')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_steps', type=int, default=300000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)

parser.add_argument('--y_ce_beta', type=int, default=1)
parser.add_argument('--y_hsic_beta', type=int, default=1)
parser.add_argument('--y_mmd_beta', type=int, default=2)
parser.add_argument('--z_kl_beta', type=int, default=1)
parser.add_argument('--z_hsic_beta', type=int, default=1)
parser.add_argument('--z2_kl_beta_max', type=int, default=16)
parser.add_argument('--z2_kl_stop_step', type=int, default=300000)

parser.add_argument('--n_branches', type=int, default=3)
parser.add_argument('--n_switches', type=int, default=5)
parser.add_argument('--n_dims_sm', type=int, default=10)
parser.add_argument('--n_latent_z2', type=int, default=10)
parser.add_argument('--fc_operator_type', type=str, default='I')
parser.add_argument('--fc_switch_type', type=str, default='I')
parser.add_argument('--backward_on_y', action='store_true', default=False)

parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=10000)

args = parser.parse_args()
args.exp_id = 'exp-%d' % args.exp_id
args.cuda_id = 'cuda:%d' % args.cuda_id
args.save_dir = os.path.join(args.save_dir, args.model, args.version, args.dataset, args.exp_id)
args.output_dir = os.path.join(args.output_dir, args.model, args.version, args.dataset, args.exp_id)


def train():
    mkdir(args.save_dir)
    device = torch.device(args.cuda_id if torch.cuda.is_available() else 'cpu')

    train_loader, ds = get_data_loader(args.dataset, args.batch_size, args.n_steps)
    model = FCSwitchedVAE(args.y_ce_beta, args.y_hsic_beta, args.y_mmd_beta, args.z_hsic_beta, args.z_kl_beta,
                          args.z2_kl_beta_max, args.z2_kl_stop_step, args.channels, args.n_branches, args.n_switches,
                          args.n_dims_sm, args.fc_operator_type, args.fc_switch_type, args.n_latent_z2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2))

    ckpt_paths = []
    model.train()

    for i, batch in enumerate(train_loader):
        step = i + 1
        _, inputs = batch
        x = inputs.to(device).float()
        recon_x, z2, z2_mean, z2_logvar, ys_logits, ys_logits2, ys_idx, ys, zs_mean, zs_logvar, zs = \
            model(x, backward_on_y=args.backward_on_y)
        loss, recon_loss, z2_kl_loss, z_hsic_loss, z_kl_loss, y_ce_loss, y_hsic_loss, y_mmd_loss = \
            model.loss(x, recon_x, z2_mean, z2_logvar, ys_logits, ys_logits2, ys, zs_mean, zs_logvar, zs, step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.print_freq == 0:
            print('[Step %d] loss: %.4f, recon_loss: %.4f, z2_kl_loss: %.4f, z_hsic_loss: %.4f, z_kl_loss: %.4f, '
                  'y_ce_loss: %.4f, y_hsic_loss: %.4f, y_mmd_loss: %.4f' %
                  (step, loss, recon_loss, z2_kl_loss, z_hsic_loss, z_kl_loss, y_ce_loss, y_hsic_loss, y_mmd_loss))
        if step % args.save_freq == 0:
            path = os.path.join(args.save_dir, 'step-%d.ckpt' % step)
            torch.save({'step': step, 'loss': loss, 'recon_loss': recon_loss,
                        'z2_kl_loss': z2_kl_loss,  'z_hsic_loss': z_hsic_loss, 'z_kl_loss': z_kl_loss,
                        'y_ce_loss': y_ce_loss, 'y_hsic_loss': y_hsic_loss, 'y_mmd_loss': y_mmd_loss,
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
