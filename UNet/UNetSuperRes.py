import torch
import torch.nn as nn
import torch.nn.functional as F
from DownSample import DownSample
from DoubleConvolution import DoubleConv
from UpSample import UpSample


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        """
        Args:
            base_filters: Number of filters in the first layer.
                          Reduced to 32 (from 64) to save memory on CPU.
        """
        super().__init__()

        f = base_filters

        # Encoder (Downsampling)
        self.down_conv1 = DownSample(in_channels, f)  # e.g., 32
        self.down_conv2 = DownSample(f, f * 2)  # e.g., 64
        self.down_conv3 = DownSample(f * 2, f * 4)  # e.g., 128
        self.down_conv4 = DownSample(f * 4, f * 8)  # e.g., 256

        # Bottleneck
        self.bottle_neck = DoubleConv(f * 8, f * 16)  # e.g., 512

        # Decoder (Upsampling)
        self.up_conv1 = UpSample(f * 16, f * 8)
        self.up_conv2 = UpSample(f * 8, f * 4)
        self.up_conv3 = UpSample(f * 4, f * 2)
        self.up_conv4 = UpSample(f * 2, f)

        # Super-Resolution Upsampler (4x)
        # We use the same base 'f' for the upsampling features to keep it light
        self.final_upsample_block = nn.Sequential(
            # First 2x upscale
            nn.ConvTranspose3d(f, f, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Second 2x upscale
            nn.ConvTranspose3d(f, f, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Final convolution to output
            nn.Conv3d(f, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Residual Connection (Interpolated Input)
        x_residual_upsampled = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)

        # Encoder
        down_1, p1 = self.down_conv1(x)
        down_2, p2 = self.down_conv2(p1)
        down_3, p3 = self.down_conv3(p2)
        down_4, p4 = self.down_conv4(p3)

        # Bottleneck
        b = self.bottle_neck(p4)

        # Decoder
        up_1 = self.up_conv1(b, down_4)
        up_2 = self.up_conv2(up_1, down_3)
        up_3 = self.up_conv3(up_2, down_2)
        up_4 = self.up_conv4(up_3, down_1)

        # Super-Res Upscale
        features_hr = self.final_upsample_block(up_4)

        # Shape Check (Padding safety)
        if features_hr.shape != x_residual_upsampled.shape:
            x_residual_upsampled = F.interpolate(x, size=features_hr.shape[2:], mode='trilinear', align_corners=False)

        out = features_hr + x_residual_upsampled
        return out
