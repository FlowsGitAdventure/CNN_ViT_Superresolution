import torch
import torch.nn as nn
import torch.nn.functional as F
from DownSampleHybrid import DownSampleHybrid
from HybridEncoderBlock import HybridEncoderBlock
from UpSampleHybrid import UpSample


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super().__init__()

        f = base_filters

        # Encoder
        self.down_conv1 = DownSampleHybrid(in_channels, f)
        self.down_conv2 = DownSampleHybrid(f, f * 2)
        self.down_conv3 = DownSampleHybrid(f * 2, f * 4)
        self.down_conv4 = DownSampleHybrid(f * 4, f * 8)

        # Bottleneck
        self.bottle_neck = HybridEncoderBlock(f * 8, f * 16)  # e.g., 512

        # Decoder
        self.up_conv1 = UpSample(f * 16, f * 8)
        self.up_conv2 = UpSample(f * 8, f * 4)
        self.up_conv3 = UpSample(f * 4, f * 2)
        self.up_conv4 = UpSample(f * 2, f)

        # Super-Resolution Upsampler (4x)
        self.final_upsample_block = nn.Sequential(
            # First 2x upscale
            nn.ConvTranspose3d(f, f, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Second 2x upscale
            # nn.ConvTranspose3d(f, f, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(inplace=True),

            # Final convolution to output
            nn.Conv3d(f, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        b, t, d, h, w = x.shape

        # 1. Create Initial Residual
        target_d, target_h, target_w = d * 2, h * 2, w * 2
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

        # 4. Reshape features_hr back to 5D to match the Residual
        b, t = x.shape[0], x.shape[1]
        out_c = features_hr.shape[1]
        features_hr = features_hr.view(b, t, out_c, *features_hr.shape[2:])

        # 5. FIXED: Correct Interpolation of Residual
        if features_hr.shape[3:] != x_residual_upsampled.shape[2:]:
            x_residual_upsampled = F.interpolate(x_residual_upsampled, size=features_hr.shape[3:],
                                  mode='trilinear', align_corners=False)

        # 6. Final Addition
        if x_residual_upsampled.ndim == 5:
            x_residual_upsampled = x_residual_upsampled.unsqueeze(2)

        out = features_hr + x_residual_upsampled
        return out
