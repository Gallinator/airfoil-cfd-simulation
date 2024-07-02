import torch.optim
from torch import nn


class Model(nn.Module):
    def __init__(self, grid_size: int, in_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.seq = nn.Sequential(
            # Landmark + grids + alpha
            nn.Linear(in_size + grid_size * 2 + 1, 2048),
            nn.Tanh(),
            nn.Linear(2048, 4096),
            nn.Tanh(),
            nn.Linear(4096, 8192),
            nn.Tanh())
        self.v = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())
        self.u = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())
        self.p = nn.Sequential(
            nn.Linear(8192, grid_size),
            nn.Tanh())

    def forward(self, alpha, grid_x, grid_y, landmarks) -> tuple:
        in_data = torch.cat((alpha, landmarks, grid_x, grid_y), 1)
        x = self.seq(in_data)
        return self.v(x), self.u(x), self.p(x)
