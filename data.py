import os
os.environ['DISENTANGLEMENT_LIB_DATA'] = './datasets'
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
from torch.utils import data as torch_data

from disentanglement_lib.data.ground_truth import named_data


ground_truth_data_names = ('dsprites_full', 'dsprites_noshape', 'color_dsprites', 'noisy_dsprites', 'scream_dsprites',
                           'smallnorb',
                           'cars3d',
                           'mpi3d_toy',  # 'mpi3d_realistic', 'mpi3d_real',
                           # 'shapes3d'
                           )


class GroundTruthDataset(torch_data.dataset.Dataset):
    def __init__(self, name, n_samples=0, seed=0):
        assert name in ground_truth_data_names
        self.name = name
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dataset = named_data.get_named_ground_truth_data(self.name)
        self.n_samples = len(self.dataset.images) if n_samples == 0 else n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        assert item < self.n_samples
        factors, observation = self.dataset.sample(1, self.random_state)
        factors = factors[0]  # (numpy array) n_factors
        observation = torch.from_numpy(np.moveaxis(observation[0], 2, 0))  # (torch tensor) C x H x W
        return factors, observation

    @property
    def size(self):
        return len(self.dataset.images)


def get_data_loader(ds_name, batch_size, n_steps, seed=0, n_workers=0, **kwargs):
    ds = GroundTruthDataset(ds_name, batch_size * n_steps, seed=seed)
    loader = torch_data.dataloader.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, **kwargs)
    return loader, ds


def output_samples(ds_name, n_rows=10, n_cols=10, n_figures=10):
    dir_path = os.path.join('output/samples', ds_name)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

    ds = GroundTruthDataset(ds_name)
    for i in range(n_figures):
        fig = plt.figure()
        n_samples = n_rows * n_cols
        for j in range(n_samples):
            factors, observation = ds[j]
            image = np.squeeze(np.moveaxis(observation.numpy(), 0, 2))
            ax = plt.subplot(n_rows, n_cols, j + 1)
            plt.tight_layout()
            ax.set_title(','.join(factors.astype(str)))
            ax.axis('off')
            plt.imshow(image, cmap=cm.get_cmap('gray') if len(image.shape) == 2 else None)
        plt.savefig(os.path.join(dir_path, str(i)), fomat='png')
        plt.close(fig)


if __name__ == '__main__':
    '''
    0 - elevation (4 different values)
    1 - azimuth (24 different values)
    2 - object type (183 different values)
    '''
    # output_samples('cars3d', n_rows=5, n_cols=5, n_figures=40)

    '''
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    '''
    #output_samples('dsprites_full', n_rows=5, n_cols=5, n_figures=40)

    '''
    0 - scale (6 different values)
    1 - orientation (40 different values)
    2 - position x (32 different values)
    3 - position y (32 different values)
    '''
    # output_samples('dsprites_noshape', n_rows=5, n_cols=5, n_figures=40)

    '''
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    '''
    output_samples('color_dsprites', n_rows=5, n_cols=5, n_figures=40)

    '''
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    '''
    # output_samples('noisy_dsprites', n_rows=5, n_cols=5, n_figures=40)

    '''
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    '''
    # output_samples('scream_dsprites', n_rows=5, n_cols=5, n_figures=40)

    '''
    0 - category (5 different values)
    1 - elevation (9 different values)
    2 - azimuth (18 different values)
    3 - lighting condition (6 different values)
    '''
    # output_samples('smallnorb', n_rows=5, n_cols=5, n_figures=40)

    '''
    0 - Object color (4 different values)
    1 - Object shape (4 different values)
    2 - Object size (2 different values)
    3 - Camera height (3 different values)
    4 - Background colors (3 different values)
    5 - First DOF (40 different values)
    6 - Second DOF (40 different values)
    '''
    # output_samples('mpi3d_toy', n_rows=4, n_cols=4, n_figures=50)