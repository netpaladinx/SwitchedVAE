import os
import shutil


def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def new_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
