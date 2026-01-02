import torch
import torch.nn as nn
import torch.nn.functional as F
from DownSample import DownSample
from UpSample import UpSample


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super().__init__()

        f = base_filters

        # Encoder
        self.down_conv1 = DownSample(in_channels, f, use_norm=True)
        self.down_conv2 = DownSample(f, f * 2, use_norm=True)
        self.down_conv3 = DownSample(f * 2, f * 4, use_norm=True)

        # Bottleneck
        # self.bottle_neck = DoubleConv(f * 8, f * 16)  # e.g., 512
        self.vit_layer = nn.TransformerEncoderLayer(
            d_model=f*4,
            nhead=4,
            batch_first=True,
        )

        # Decoder
        self.up_conv2 = UpSample(f * 4, f * 4, f * 4, use_norm=True)
        self.up_conv3 = UpSample(f * 4, f * 2, f * 2, use_norm=False)
        self.up_conv4 = UpSample(f * 2, f, f, use_norm=False)

        # Super-Resolution Upsampler (4x)
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
        b, t, d, h, w = x.shape

        # 1. Prepare Residual Connection
        target_size = (d * 4, h * 4, w * 4)
        x_residual = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)

        # 2. Encoder (Shared Weights Approach)
        x_in = x.reshape(b * t, 1, d, h, w)

        down_1, p1 = self.down_conv1(x_in)
        down_2, p2 = self.down_conv2(p1)
        down_3, p3 = self.down_conv3(p2)

        # 3. ViT Bottleneck (Global Temporal Reasoning)
        _, c, bd, bh, bw = p3.shape
        v_in = p3.view(b, t, c, bd, bh, bw).permute(0, 1, 3, 4, 5, 2)  # (B, T, D, H, W, C)
        v_in = v_in.reshape(b, t * bd * bh * bw, c)

        v_out = self.vit_layer(v_in)

        # Restore to CNN shape
        v_out = v_out.view(b, t, bd, bh, bw, c).permute(0, 1, 5, 2, 3, 4)
        p3_vit = v_out.reshape(b * t, c, bd, bh, bw)

        # 4. Decoder
        up_2 = self.up_conv2(p3_vit, down_3)
        up_3 = self.up_conv3(up_2, down_2)
        up_4 = self.up_conv4(up_3, down_1)

        # 5. Super-Res Output
        features_hr = self.final_upsample_block(up_4)

        _, out_c, d_hr, h_hr, w_hr = features_hr.shape
        features_hr = features_hr.view(b, t, out_c, d_hr, h_hr, w_hr)

        if features_hr.shape[3:] != x_residual.shape[2:]:
            x_residual = F.interpolate(x_residual, size=features_hr.shape[3:], mode='trilinear')

        out = features_hr + x_residual.unsqueeze(2) if x_residual.ndim == 5 else features_hr + x_residual
        return out
