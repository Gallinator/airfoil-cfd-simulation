import math

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class AirfoilDataset(Dataset):
    def __init__(self, data_path: str):
        self.file = h5py.File(data_path, 'r')
        self.alphas = self.file['alpha']
        self.landmarks = self.file['landmarks']
        self.masks = self.file['masks']
        self.u = self.file['u']
        self.v = self.file['v']
        self.r = self.file['rho']
        self.e = self.file['energy']
        self.grid_coords_x, self.grid_coords_y = self.file['grid'][()]
        self.grid_shape = (128, 128)

    def get_free_flow_grid(self, alpha):
        grid_x = np.full(self.grid_shape, 1 * math.cos(math.radians(alpha)))
        grid_x = torch.tensor(grid_x, dtype=torch.float32)
        grid_y = np.full(self.grid_shape, 1 * math.sin(math.radians(alpha)))
        grid_y = torch.tensor(grid_y, dtype=torch.float32)
        return grid_x, grid_y

    def __getitem__(self, item):
        landmark = torch.tensor(self.landmarks[item], dtype=torch.float32)
        mask = torch.tensor(self.masks[item], dtype=torch.float32)
        u = torch.tensor(self.u[item], dtype=torch.float32)
        v = torch.tensor(self.v[item], dtype=torch.float32)
        r = torch.tensor(self.r[item], dtype=torch.float32)
        e = torch.tensor(self.e[item], dtype=torch.float32)
        grid_x, grid_y = self.get_free_flow_grid(self.alphas[item])

        return grid_x, grid_y, landmark, u, v, r, e, mask

    def __len__(self):
        return len(self.landmarks)
