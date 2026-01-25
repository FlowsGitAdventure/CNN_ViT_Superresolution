import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv = nn.Sequential(
            nn.Conv3d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(filters, filters, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.conv(x) * self.res_scale

class EDSRBaseline(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32, n_resblocks=16, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        f = base_filters

        self.head = nn.Conv3d(in_channels, f, kernel_size=3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(f) for _ in range(n_resblocks)],
            nn.Conv3d(f, f, kernel_size=3, padding=1)
        )

        # 3. Upsampling Stage
        self.upscale_stage1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, scale_factor, scale_factor), mode="trilinear", align_corners=False),
            nn.Conv3d(f, f, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.upscale_stage2 = nn.Sequential(
            nn.Upsample(scale_factor=(scale_factor, scale_factor, 1), mode="trilinear", align_corners=False),
            nn.Conv3d(f, f, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ReLU(inplace=True)
        )

        # 4. Refinement Blocks
        self.refine1 = nn.Sequential(
            nn.Conv3d(f, f, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(f, f // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.refine2 = nn.Sequential(
            nn.Conv3d(f // 2, f, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(f, f // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv3d(f // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x, target_shape):
        # x: (B, T, D, H, W)
        x_in = x
        batch_size, t, d, h, w = x.shape
        x = x.view(batch_size * t, 1, d, h, w)

        # EDSR Logic
        x_head = self.head(x)
        res = self.body(x_head)
        x = x_head + res

        x_upscaled = self.upscale_stage1(x)
        if self.scale_factor == 4:
            x_upscaled = self.upscale_stage2(x_upscaled)

        features_hr = self.refine1(x_upscaled)
        features_hr = self.refine2(features_hr)
        out_residual = self.final_conv(features_hr)

        _, _, D_t, H_t, W_t = target_shape
        out_residual = out_residual.view(batch_size, t, D_t, H_t, W_t)

        base_hr = F.interpolate(x_in, size=(D_t, H_t, W_t), mode="trilinear", align_corners=False)

        x = 0.5 * out_residual + base_hr

        # Hard crop
        x = x[:, :, :D_t, :H_t, :W_t]

        if x.shape[2] == 1:
            x = x.squeeze(2)

        return x
