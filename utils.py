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

def append(lists, elems):
    for ls, e in zip(lists, elems):
        ls.append(e)
