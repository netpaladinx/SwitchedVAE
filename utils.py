import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import torch.nn as nn


def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def new_dir(path):
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


class Lambda(nn.Module):
    def __init__(self, fn, *args):
        super(Lambda, self).__init__()
        self.args = args
        self.fn = fn

    def forward(self, x):
        return self.fn(x, *self.args)
