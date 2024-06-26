import numpy as np
import torch.optim
from torch import nn


class Model(nn.Module):
    def __init__(self, grid_x: torch.Tensor, grid_y: torch.Tensor, in_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_x = grid_x
        self.grid_y = grid_y
        out_size = grid_x.size()[0]

        self.seq = nn.Sequential(
            nn.Linear(in_size + out_size * 2, 2048),
            nn.Tanh(),
            nn.Linear(2048, 4096),
            nn.Tanh(),
            nn.Linear(4096, 8192),
            nn.Tanh())
        self.momentum_x = nn.Sequential(
            nn.Linear(8192, out_size),
            nn.Tanh())
        self.momentum_y = nn.Sequential(
            nn.Linear(8192, out_size),
            nn.Tanh())

    def forward(self, landmarks: torch.Tensor) -> tuple:
        bs = landmarks.size()[0]
        x = self.seq(torch.cat((landmarks, self.grid_x.repeat(bs, 1), self.grid_y.repeat(bs, 1)), 1))
        return self.momentum_x(x), self.momentum_y(x)
