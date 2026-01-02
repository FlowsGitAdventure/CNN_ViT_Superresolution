import torch.nn as nn
import torch.nn.functional as F
from DownSample import DownSample
from DoubleConvolution import DoubleConv
from UpSample import UpSample


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32, scale_factor=4):
        super().__init__()

        self.scale_factor = scale_factor
        if scale_factor > 2:
            self.scale_factor = 4

        f = base_filters

        # Encoder
        self.down_conv1 = DownSample(in_channels, f, use_norm=True)
        self.down_conv2 = DownSample(f, f * 2, use_norm=True)
        self.down_conv3 = DownSample(f * 2, f * 4, use_norm=True)
        self.down_conv4 = DownSample(f * 4, f * 8, use_norm=True)

        # Bottleneck
        self.bottle_neck = DoubleConv(f * 8, f * 16, use_norm=False)  # e.g., 512

        # Decoder
        # Decoder - Now passing the correct skip_channels for each level
        self.up_conv1 = UpSample(f * 16, f * 8, use_norm=True, skip_channels=f * 8)
        self.up_conv2 = UpSample(f * 8, f * 4, use_norm=True, skip_channels=f * 4)
        self.up_conv3 = UpSample(f * 4, f * 2, use_norm=False, skip_channels=f * 2)
        self.up_conv4 = UpSample(f * 2, f, use_norm=False, skip_channels=f)

        # Super-Resolution Upsampler (4x)
        self.final_upsample_block = nn.Sequential(
            # First 2x upscale
            nn.ConvTranspose3d(f, f, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Residual Connection (Interpolated Input)
        # x_residual_upsampled = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)

        target_d, target_h, target_w = x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2
        if self.scale_factor == 4:
            target_d, target_h, target_w = x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2

        x_residual_upsampled = F.interpolate(x, size=(target_d, target_h, target_w),
                                             mode='trilinear', align_corners=False)

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
        if self.scale_factor == 4:
            features_hr = self.final_upsample_block(features_hr)
        features_hr = self.final_conv(features_hr)

        # Shape Check for Padding safety
        if features_hr.shape != x_residual_upsampled.shape:
            x_residual_upsampled = F.interpolate(features_hr, size=features_hr.shape[2:], mode='trilinear', align_corners=False)

        out = features_hr + x_residual_upsampled
        return out
