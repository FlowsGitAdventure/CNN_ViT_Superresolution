import torch
import torch.nn as nn
import torch.nn.functional as f
from DoubleConvolution import DoubleConv


# Info: When integrating all in one, think of the skip channels for Bottle and Swin.
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm, skip_channels=None):
        super().__init__()
        # 1. Standard Transposed Conv for 3D fMRI
        self.up = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        if skip_channels is not None:
            combined_channels = (in_channels // 2) + skip_channels
        else:
            combined_channels = in_channels // 2

        self.conv = DoubleConv(combined_channels, out_channels, use_norm=use_norm)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)

            x = f.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2
            ])
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)

