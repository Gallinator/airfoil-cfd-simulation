import h5py
import torch
from torch.utils.data import Dataset


class AirfoilDataset(Dataset):
    def __init__(self, data_path: str):
        self.file = h5py.File(data_path, 'r')
        self.landmarks = self.file['landmarks']
        self.momentum_x = self.file['rho_u']
        self.momentum_y = self.file['rho_v']
        self.grid_x, self.grid_y = self.file['grid'][()]

    def __getitem__(self, item):
        landmark = self.landmarks[item]
        momentum_x = self.momentum_x[item]
        momentum_y = self.momentum_y[item]

        return (torch.tensor(landmark, dtype=torch.float32),
                torch.tensor(momentum_x, dtype=torch.float32),
                torch.tensor(momentum_y, dtype=torch.float32))

    def __len__(self):
        return len(self.landmarks)
