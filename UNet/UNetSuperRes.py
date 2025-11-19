import torch
import torch.nn as nn
import torch.nn.functional as F
from DownSample import DownSample
from DoubleConvolution import DoubleConv
from UpSample import UpSample


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_conv1 = DownSample(in_channels, 64)
        self.down_conv2 = DownSample(64, 128)
        self.down_conv3 = DownSample(128, 256)
        self.down_conv4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_conv1 = UpSample(1024, 512)
        self.up_conv2 = UpSample(512, 256)
        self.up_conv3 = UpSample(256, 128)
        self.up_conv4 = UpSample(128, 64)

        self.final_upsample_block = nn.Sequential(
            # First 2x upscale (Double spatial dims: D, H, W)
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            # Second 2x upscale (Double spatial dims again)
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            # Final convolution to get to target out_channels
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_residual_upsampled = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)

        # UNet Decoder
        down_1, p1 = self.down_conv1(x)
        down_2, p2 = self.down_conv2(p1)
        down_3, p3 = self.down_conv3(p2)
        down_4, p4 = self.down_conv4(p3)

        # Bottleneck
        b = self.bottle_neck(p4)

        # UNet Decoder
        up_1 = self.up_conv1(b, down_4)
        up_2 = self.up_conv2(up_1, down_3)
        up_3 = self.up_conv3(up_2, down_2)
        up_4 = self.up_conv4(up_3, down_1)

        features_hr = self.final_upsample_block(up_4)
        out = features_hr + x_residual_upsampled

        return out
