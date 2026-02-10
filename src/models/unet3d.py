import torch.nn as nn
from monai.networks.nets import UNet


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16, depth=4,
                 num_res_units=2, norm="batch"):
        super().__init__()
        channels = [base_channels * (2 ** i) for i in range(depth)]
        strides = [2] * (depth - 1)
        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
        )

    def forward(self, x):
        return self.net(x)
