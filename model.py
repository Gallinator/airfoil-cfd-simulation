import torch.optim
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True, dropout=0.005,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.use_batch_norm = batch_norm
        self.f = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        return self.f(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, batch_norm=True, dropout=0.005,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upsample = nn.Upsample(scale_factor=2)
        self.padding = nn.ReplicationPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.use_batch_norm = batch_norm
        self.f = nn.LeakyReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.padding(x)
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        return self.f(x)


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
