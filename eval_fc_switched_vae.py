import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from visdom import Visdom

from data import get_data_loader
from model_fc_switched_vae import FCSwitchedVAE
from utils import mkdir, grid2gif


parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--dataset', type=str, default='dsprites_full')
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--model', type=str, default='fc_switched_vae')
parser.add_argument('--save_dir', type=str, default='./checkpoints')
parser.add_argument('--output_dir', type=str, default='./output')


parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_steps', type=int, default=300000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--adam_beta1', type=float, default=0.9)
parser.add_argument('--adam_beta2', type=float, default=0.999)

parser.add_argument('--y_ce_beta', type=int, default=1)
parser.add_argument('--y_phsic_beta', type=int, default=1)
parser.add_argument('--y_mmd_beta', type=int, default=2)
parser.add_argument('--z_beta', type=int, default=1)
parser.add_argument('--z2_beta', type=int, default=4)

parser.add_argument('--n_branches', type=int, default=3)
parser.add_argument('--n_switches', type=int, default=6)
parser.add_argument('--backward_on_y_hard', action='store_true', default=False)

parser.add_argument('--print_freq', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=100)

parser.add_argument('--port', type=int, default=8097)
parser.add_argument('--hostname', type=str, default='http://localhost')
parser.add_argument('--base_url', type=str, default='/')
parser.add_argument('--traversal_limit', type=float, default=2.0)
parser.add_argument('--traversal_inter', type=float, default=0.5)
parser.add_argument('--ckpts', type=str, default='step-9400.ckpt')

args = parser.parse_args()
args.exp_id = 'exp-%d' % args.exp_id
args.save_dir = os.path.join(args.save_dir, args.model, args.dataset, args.exp_id)
args.output_dir = os.path.join(args.output_dir, args.model, args.dataset, args.exp_id)
args.ckpts = args.ckpts.split(',') if args.ckpts else []

# def eval_print_code(eval_loader, model, device, path):
#     checkpoint = torch.load(path)
#     step = checkpoint['step']
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#
#     print(path)
#     with torch.no_grad():
#         counts = defaultdict(lambda: defaultdict(int))
#         for i, batch in enumerate(eval_loader):
#             _, inputs = batch
#             x = inputs.to(device).float()
#             z2_mean, z2_logvar, ys_logits, ys_logits_2, ys_index, ys_hard, zs_mean, zs_logvar, zs = model.encoder(x)
#             print('[Step %d - %d] %s, %s' % (step, i,
#                                              ', '.join(['(%d, %.4f)' % (y_index.item(), z.item())
#                                                     for y_index, z in zip(ys_index, zs)]),
#                                              ', '.join([str(y_logits.cpu().numpy()) for y_logits in ys_logits])))
#             for i1, y_index_1 in enumerate(ys_index):
#                 counts[i1][y_index_1.item()] += 1
#                 for i2, y_index_2 in enumerate(ys_index):
#                     if i1 != i2:
#                         counts[(i1, i2)][(y_index_1.item(), y_index_2.item())] += 1
#         for k1 in counts:
#             for k2 in counts[k1]:
#                 print(k1, k2, counts[k1][k2])


def copy_code(z2, ys_index, ys_hard, zs):
    z2_copy = z2.clone()
    ys_index_copy = [y_index.clone() for y_index in ys_index]
    ys_hard_copy = [y_hard.clone() for y_hard in ys_hard]
    zs_copy = [z.clone() for z in zs]
    return z2_copy, ys_index_copy, ys_hard_copy, zs_copy


def traversal(z2, ys_index, ys_hard, zs, n_branches, n_switches, limit=3, inter=1):
    ''' z2: 1 x 10
        ys_index: [ 1 x 1 ] x 6 (n_switches)
        ys_hard: [ 1 x 3 (n_branches) ] x 6
        zs: [ 1 x 1 ] x 6
    '''
    interpolation = torch.arange(-limit, limit + 1e-4, inter)
    codes = []
    for i in range(n_switches):
        for j in range(n_branches):
            for val in interpolation:
                z2_copy, ys_index_copy, ys_hard_copy, zs_copy = copy_code(z2, ys_index, ys_hard, zs)
                ys_index_copy[i][0, 0] = j
                ys_hard_copy[i] = torch.zeros_like(ys_hard_copy[i])
                ys_hard_copy[i][0, j] = 1
                zs_copy[i][0, 0] = val
                codes.append((z2_copy, ys_index_copy, ys_hard_copy, zs_copy))
    for i in range(z2.size(1)):
        for val in interpolation:
            z2_copy, ys_index_copy, ys_hard_copy, zs_copy = copy_code(z2, ys_index, ys_hard, zs)
            z2_copy[0, i] = val
            codes.append((z2_copy, ys_index_copy, ys_hard_copy, zs_copy))
    z2_copy, ys_index_copy, ys_hard_copy, zs_copy = zip(*codes)
    z2_copy = torch.cat(z2_copy, 0)
    ys_index_copy = [torch.cat([ys_index_cp[i] for ys_index_cp in ys_index_copy], 0) for i in range(n_switches)]
    ys_hard_copy = [torch.cat([ys_hard_cp[i] for ys_hard_cp in ys_hard_copy], 0) for i in range(n_switches)]
    zs_copy = [torch.cat([zs_cp[i] for zs_cp in zs_copy], 0) for i in range(n_switches)]
    return z2_copy, ys_index_copy, ys_hard_copy, zs_copy, interpolation


def eval_visual(vis, n_samples=10):
    visual_dir = os.path.join(args.output_dir, 'visual')
    mkdir(visual_dir)
    device = torch.device('cuda:%d' % args.cuda_id if torch.cuda.is_available() else 'cpu')

    model = FCSwitchedVAE(args.y_ce_beta, args.y_phsic_beta, args.y_mmd_beta, args.z_beta, args.z2_beta,
                          args.channels, args.n_branches, args.n_switches).to(device)

    for ckpt in os.listdir(args.save_dir):
        if ckpt[-5:] != '.ckpt' or (args.ckpts and ckpt not in args.ckpts):
            continue
        ckpt_path = os.path.join(args.save_dir, ckpt)
        print('Load from %s' % ckpt_path)
        checkpoint = torch.load(ckpt_path)
        step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        eval_loader, _ = get_data_loader(args.dataset, 1, n_samples)

        with torch.no_grad():
            gifs = []
            true_images, recon_images = [], []
            for i, batch in enumerate(eval_loader):
                _, inputs = batch
                x = inputs.to(device).float()
                recon_x, z2, z2_mean, z2_logvar, ys_logits, ys_logits_2, ys_index, ys_hard, zs_mean, zs_logvar, zs = model(x)
                true_images.append(x.cpu())
                recon_images.append(torch.sigmoid(recon_x).cpu())

                for i, (y_index, z_mean, z_logvar) in enumerate(zip(ys_index, zs_mean, zs_logvar)):
                    print('[%d]: y_index: %d, z_mean: %.4f, z_std: %.4f' %
                          (i, y_index.item(), z_mean.item(), (z_logvar/2).exp()))

                z2_tr, ys_index_tr, ys_hard_tr, zs_tr, interpolation = \
                    traversal(z2, ys_index, ys_hard, zs, args.n_branches, args.n_switches,
                              limit=args.traversal_limit, inter=args.traversal_inter)
                images_tr = torch.sigmoid(model.decoder(z2_tr, ys_index_tr, ys_hard_tr, zs_tr)).cpu()  # n_images x C x H x W

                vis.images(images_tr, env=os.path.join(args.model, args.dataset, args.exp_id), nrow=interpolation.size(0),
                           opts=dict(title='Latent Space Traversal (iter:%d)' % step))

                gifs.append(images_tr)

            # _, C, H, W = gifs[0].size()
            # n_z = args.n_switches * args.n_branches + z2_tr.size(1)
            # gifs = torch.cat(gifs, 0).view(-1, n_z, interpolation.size(0), C, H, W).transpose(1, 2)
            # for i in range(gifs.size(0)):
            #     for j in range(interpolation.size(0)):
            #         save_image(tensor=gifs[i][j], filename=os.path.join(visual_dir, 'traversal_%d_%d.jpg' % (i, j)),
            #                    nrow=n_z, pad_value=1)
            #     grid2gif(os.path.join(visual_dir, 'traversal_%d_*.jpg' % i),
            #              os.path.join(visual_dir, 'traversal_%d.gif' % i), delay=10)

            _, C, H, W = true_images[0].size()
            true_images = torch.cat(true_images, 0)
            recon_images = torch.cat(recon_images, 0)
            images = torch.stack([true_images, recon_images], 1).view(-1, C, H, W)
            images = make_grid(images, nrow=2)
            vis.images(images, env=os.path.join(args.model, args.dataset, args.exp_id),
                       opts=dict(title='Reconstruction (iter:%d)' % step))



def eval():
    vis = Visdom(port=args.port, server=args.hostname, base_url=args.base_url,
                 username='', password='', use_incoming_socket=True)
    eval_visual(vis)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    eval()
