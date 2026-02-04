import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diff = [x2.size(d) - x1.size(d) for d in range(2, 5)]
        x1 = F.pad(x1, [
            diff[2] // 2, diff[2] - diff[2] // 2,
            diff[1] // 2, diff[1] - diff[1] // 2,
            diff[0] // 2, diff[0] - diff[0] // 2,
        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16, depth=4):
        super().__init__()
        self.depth = depth
        channels = [base_channels * (2 ** i) for i in range(depth)]
        self.inc = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(Down(channels[i], channels[i + 1]))
        self.ups = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.ups.append(Up(channels[i], channels[i - 1]))
        self.outc = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        xs = [x1]
        for down in self.downs:
            xs.append(down(xs[-1]))
        x = xs[-1]
        for i, up in enumerate(self.ups):
            x = up(x, xs[-(i + 2)])
        return self.outc(x)
