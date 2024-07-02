import h5py
import torch
from torch.utils.data import Dataset


class AirfoilDataset(Dataset):
    def __init__(self, data_path: str):
        self.file = h5py.File(data_path, 'r')
        self.alphas = self.file['alpha']
        self.landmarks = self.file['landmarks']
        self.u = self.file['u']
        self.v = self.file['v']
        self.p = self.file['p']
        self.grid_x, self.grid_y = self.file['grid'][()]

    def __getitem__(self, item):
        alpha = torch.tensor([self.alphas[item]], dtype=torch.float32)
        landmark = torch.tensor(self.landmarks[item], dtype=torch.float32)
        u = torch.tensor(self.u[item], dtype=torch.float32)
        v = torch.tensor(self.v[item], dtype=torch.float32)
        p = torch.tensor(self.p[item], dtype=torch.float32)

        return alpha, landmark, u, v, p

    def __len__(self):
        return len(self.landmarks)
