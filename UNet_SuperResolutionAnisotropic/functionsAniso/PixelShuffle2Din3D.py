import torch
import torch.nn as nn


class PixelShuffle2DIn3D(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super().__init__()
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        b, c, d, h, w = x.shape

        # 2. Reshape to treat Depth as part of the Batch (Anisotropic move)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)

        # 3. Apply 2D PixelShuffle
        x = self.ps(x)  # Shape: (B*D, C_out, H*r, W*r)

        # 4. Restore 3D shape
        _, c_out, h_new, w_new = x.shape
        x = x.view(b, d, c_out, h_new, w_new).permute(0, 2, 1, 3, 4)

        return x
