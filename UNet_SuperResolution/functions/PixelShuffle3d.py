import torch
import torch.nn as nn


class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, d, h, w = x.size()
        r = self.upscale_factor
        new_channels = channels // (r ** 3)

        # Reshape to separate the upscaling components
        x = x.view(batch_size, new_channels, r, r, r, d, h, w)
        # Permute to move upscaling factors to spatial dimensions
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        # Flatten into final 3D shape
        return x.view(batch_size, new_channels, d * r, h * r, w * r)