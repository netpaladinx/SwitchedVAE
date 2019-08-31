import os
import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import torch.nn as nn


def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def plot_images(images, n_rows, n_cols, path):
    fig, axs = plt.subplots(n_rows, n_cols)
    axs = trim_axs(axs, images.size(0))
    for ax, image in zip(axs, images):
        image = np.squeeze(np.moveaxis(image.cpu().numpy(), 0, 2))
        ax.imshow(image, cmap=cm.get_cmap('gray') if len(image.shape) == 2 else None)
        ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)
    print(path)


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.
    code from:
        https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python/34555939#34555939
    """
    str1 = 'convert -delay '+str(delay)+' -loop 0 ' + image_str  + ' ' + output_gif
    subprocess.call(str1, shell=True)


def get_padding(kernel_size, stride=None, h_or_w=None):
    unused = (h_or_w - 1) % stride if h_or_w and stride else 0
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg - unused
    return (pad_beg, pad_end, pad_beg, pad_end)


def slice2d(x, padding):
    pad_t, pad_b, pad_l, pad_r = padding
    return x.narrow(2, pad_t, x.size(2) - pad_t - pad_b).narrow(3, pad_l, x.size(3) - pad_l - pad_r)


class Lambda(nn.Module):
    def __init__(self, fn, *args):
        super(Lambda, self).__init__()
        self.args = args
        self.fn = fn

    def forward(self, x):
        return self.fn(x, *self.args)


def conv2d(in_channels, out_channels, kernel_size, stride, bias=True, in_h_or_w=None):
    if kernel_size > 1:
        return nn.Sequential(
            nn.ZeroPad2d(get_padding(kernel_size, stride, in_h_or_w)),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias))
    else:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)


def deconv2d(in_channels, out_channels, kernel_size, stride, bias=True, out_h_or_w=None):
    if kernel_size > 1:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias),
            Lambda(slice2d, get_padding(kernel_size, stride, out_h_or_w)))
    else:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)
