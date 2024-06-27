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
        landmark = torch.tensor(self.landmarks[item], dtype=torch.float32)
        momentum_x = torch.tensor(self.momentum_x[item], dtype=torch.float32)
        momentum_y = torch.tensor(self.momentum_y[item], dtype=torch.float32)

        return landmark, momentum_x, momentum_y

    def __len__(self):
        return len(self.landmarks)
