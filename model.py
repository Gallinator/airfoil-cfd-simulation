import torch.optim
from torch import nn


class Model(nn.Module):
    def __init__(self, grid_size: int, in_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.seq = nn.Sequential(
            nn.Linear(in_size + grid_size * 2, 2048),
            nn.Tanh(),
            nn.Linear(2048, 4096),
            nn.Tanh(),
            nn.Linear(4096, 8192),
            nn.Tanh())
        self.momentum_x = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())
        self.momentum_y = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())

    def forward(self, landmarks, grid_x, grid_y, ) -> tuple:
        in_data = torch.cat((landmarks, grid_x, grid_y), 1)
        x = self.seq(in_data)
        return self.momentum_x(x), self.momentum_y(x)
