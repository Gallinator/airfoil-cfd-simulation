import math

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class AirfoilDataset(Dataset):
    """
    PyTorch dataset containing airfoil samples. Contains the following features:
        - alphas: angle of attack
        - landmarks: coordinates of point defining the airfoil shape
        - masks: binary mask set to 1 where the airfoil is, 0 elsewhere
        - u: flow velocity in the x direction
        - v: flow velocity in the y direction
        - r: flow density
        - e: flow total energy
        - cd: coefficients of drag
        - cl: coefficients of lift
        - cm: coefficients of momentum
    """

    def __init__(self, data_path: str):
        self.file = h5py.File(data_path, 'r')
        self.alphas = self.file['alpha']
        self.landmarks = self.file['landmarks']
        self.masks = self.file['masks']
        self.u = self.file['u']
        self.v = self.file['v']
        self.r = self.file['rho']
        self.e = self.file['energy']
        self.cd = self.file['C_d']
        self.cl = self.file['C_l']
        self.cm = self.file['C_m']
        self.grid_coords_x, self.grid_coords_y = self.file['grid'][()]
        self.grid_shape = self.grid_coords_y.shape

    def _get_free_flow_grid(self, alpha) -> tuple:
        """
        :param alpha: angle of attack
        :return: the x and y free flow velocity grids
        """
        grid_x = np.full(self.grid_shape, 1 * math.cos(math.radians(alpha)))
        grid_x = torch.tensor(grid_x, dtype=torch.float32)
        grid_y = np.full(self.grid_shape, 1 * math.sin(math.radians(alpha)))
        grid_y = torch.tensor(grid_y, dtype=torch.float32)
        return grid_x, grid_y

    def __getitem__(self, item):
        """
        Returns a tuple containing three tensors, each containing the following data:
            - flow data: u, v, airfoil mask
            - coefficients labels: lift, drag and momentum coefficients
            - flow labels: u, v, density, energy
        :param item: position index
        :return: flow input data, coefficient labels and flow labels
        """
        flow_u, flow_v = self._get_free_flow_grid(self.alphas[item])
        flow_data = np.stack((flow_u, flow_v, self.masks[item]))
        coef_labels = [self.cl[item], self.cd[item], self.cm[item]]
        flow_labels = np.stack((self.u[item], self.v[item], self.r[item], self.e[item]))
        return (torch.tensor(flow_data, dtype=torch.float32),
                torch.tensor(coef_labels, dtype=torch.float32),
                torch.tensor(flow_labels, dtype=torch.float32))

    def __len__(self):
        return len(self.landmarks)
