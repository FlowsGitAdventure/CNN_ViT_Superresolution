import torch
import torch.nn as nn
import torch.nn.functional as F
from DoubleConvolution import DoubleConv


class SkipGate(nn.Module):
    """
    Learnable gate for skip connections.
    Allows the network to control how much LR information is injected.
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, skip):
        return skip * self.gate(skip)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm, skip_channels=None):
        super().__init__()

        # Upsampling: doubles spatial resolution, halves channels
        self.up = nn.ConvTranspose3d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.use_skip = skip_channels is not None

        if self.use_skip:
            self.skip_gate = SkipGate(skip_channels)
            combined_channels = (in_channels // 2) + skip_channels
        else:
            combined_channels = in_channels // 2

        self.conv = DoubleConv(
            combined_channels,
            out_channels,
            use_norm=use_norm
        )

    def forward(self, x, skip=None):
        x = self.up(x)  # Upsampled tensor

        if self.use_skip and skip is not None:
            # 1. Calculate the difference in spatial dimensions
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)

            # 2. Pad x to match the size of skip
            # F.pad format: (left, right, top, bottom, front, back)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2,
                          diff_d // 2, diff_d - diff_d // 2])

            # 3. Apply learned gate to skip features
            skip = self.skip_gate(skip)

            # 4. Concatenate along channel dimension (dim 1)
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)
