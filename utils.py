import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from helper import trim_axs


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
