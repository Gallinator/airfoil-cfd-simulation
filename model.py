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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ds1 = EncoderBlock(3, 64, 4, 2, 1)
        self.ds2 = EncoderBlock(64, 128, 4, 2, 1)
        self.ds3 = EncoderBlock(128, 128, 4, 2, 1)
        self.ds4 = EncoderBlock(128, 256, 4, 2, 1)
        self.ds5 = EncoderBlock(256, 512, 2, 2)
        self.ds6 = EncoderBlock(512, 512, 2, 2)
        self.ds7 = EncoderBlock(512, 512, 2, 2)

        self.us1 = DecoderBlock(512, 512, 1, 1)
        self.us2 = DecoderBlock(1024, 512, 1, 1)
        self.us3 = DecoderBlock(1024, 256, 3, 1, 1)
        self.us4 = DecoderBlock(512, 128, 3, 1, 1)
        self.us5 = DecoderBlock(256, 128, 3, 1, 1)
        self.us6 = DecoderBlock(256, 64, 3, 1, 1)
        self.us7 = DecoderBlock(128, 4, 3, 1, 1)

        self.coefs_encoder = nn.Sequential(
            EncoderBlock(3, 32, 4, 2, 1),
            EncoderBlock(32, 64, 4, 2, 1),
            EncoderBlock(64, 64, 4, 2, 1),
            EncoderBlock(64, 128, 4, 2, 1),
            EncoderBlock(128, 256, 2, 2),
            EncoderBlock(256, 256, 2, 2),
            EncoderBlock(256, 256, 2, 2),
        )

        self.coefs_linear = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.01),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.01),
            nn.Linear(32, 3),
            nn.LeakyReLU())

    def flow_parameters(self):
        return (list(self.ds1.parameters()) +
                list(self.ds2.parameters()) +
                list(self.ds3.parameters()) +
                list(self.ds4.parameters()) +
                list(self.ds5.parameters()) +
                list(self.ds6.parameters()) +
                list(self.ds7.parameters()) +
                list(self.us1.parameters()) +
                list(self.us2.parameters()) +
                list(self.us3.parameters()) +
                list(self.us4.parameters()) +
                list(self.us5.parameters()) +
                list(self.us6.parameters()) +
                list(self.us7.parameters()))

    def coef_parameters(self):
        return list(self.coefs_encoder.parameters()) + list(self.coefs_linear.parameters())

    def forward(self, data):
        flow_x = self.flow_forward(data)
        coefs_data = torch.concatenate((data[:, -1, :, :].unsqueeze(1), flow_x[:, 0:2, :, :]), 1)
        coefs_x = self.coefs_forward(coefs_data)
        return flow_x, coefs_x

    def flow_forward(self, data):
        x1 = self.ds1(data)
        x2 = self.ds2(x1)
        x3 = self.ds3(x2)
        x4 = self.ds4(x3)
        x5 = self.ds5(x4)
        x6 = self.ds6(x5)
        x7 = self.ds7(x6)

        x = self.us1(x7)
        x = torch.cat((x, x6), 1)
        x = self.us2(x)
        x = torch.cat((x, x5), 1)
        x = self.us3(x)
        x = torch.cat((x, x4), 1)
        x = self.us4(x)
        x = torch.cat((x, x3), 1)
        x = self.us5(x)
        x = torch.cat((x, x2), 1)
        x = self.us6(x)
        x = torch.cat((x, x1), 1)
        x = self.us7(x)
        return x

    def coefs_forward(self, data):
        x = self.coefs_encoder(data)
        x = self.coefs_linear(torch.flatten(x, 1))
        return x

        coefs = self.coefs_linear(torch.flatten(x7, 1))
        return x, coefs
