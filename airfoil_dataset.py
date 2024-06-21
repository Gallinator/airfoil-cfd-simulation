import h5py
from torch.utils.data import Dataset


class AirfoilDataset(Dataset):
    def __init__(self, data_path: str, split: str, size: float):
        self.file = h5py.File(data_path, 'r')
        self.landmarks = self.file['landmarks']
        self.momentum_x = self.file['rho_u']
        self.momentum_y = self.file['rho_v']

        self.length = int(len(self.landmarks) * size)

        self.start = 0 if split.lower() == 'train' else len(self.landmarks) - self.length

    def __getitem__(self, item):
        if item > self.length - 1:
            raise IndexError(f'Index {item} is out of range for size {self.length}')

        landmark = self.landmarks[self.start + item]
        grid_x, grid_y = self.file['grid'][()][0], self.file['grid'][()][1]
        momentum_x = self.momentum_x[self.start + item]
        momentum_y = self.momentum_y[self.start + item]

        return (landmark,
                grid_x,
                grid_y,
                momentum_x,
                momentum_y)

    def __len__(self):
        return self.length
