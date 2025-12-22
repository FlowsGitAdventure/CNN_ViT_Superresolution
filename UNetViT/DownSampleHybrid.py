import torch.nn as nn
from HybridEncoderBlock import HybridEncoderBlock


class DownSampleHybrid(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = HybridEncoderBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p
