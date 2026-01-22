import torch.nn as nn
import torch.nn.functional as F
from DownSample import DownSample
from DoubleConvolution import DoubleConv
from UpSample import UpSample
from functionsAniso.PixelShuffle2Din3D import PixelShuffle2DIn3D


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32, scale_factor=2):
        super().__init__()

        self.scale_factor = scale_factor
        if scale_factor > 2:
            self.scale_factor = 4

        f = base_filters

        # Encoder
        self.down_conv1 = DownSample(in_channels, f, use_norm=True)
        self.down_conv2 = DownSample(f, f * 2, use_norm=True)
        self.down_conv3 = DownSample(f * 2, f * 4, use_norm=False)

        # Bottleneck
        self.bottle_neck = DoubleConv(f * 4, f * 8, use_norm=False)  # e.g., 512

        # Decoder
        # Decoder - Now passing the correct skip_channels for each level
        self.up_conv2 = UpSample(f * 8, f * 4, use_norm=False, skip_channels=f * 4)
        self.up_conv3 = UpSample(f * 4, f * 2, use_norm=False, skip_channels=f * 2)
        self.up_conv4 = UpSample(f * 2, f, use_norm=False, skip_channels=f)

        # # 1. Anisotropic PixelShuffle Upsampler (2x per block)
        # self.upscale_stage2 = nn.Sequential(
        #     nn.Conv3d(f, f * (scale_factor**2), kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        #     PixelShuffle2DIn3D(in_channels=f, upscale_factor=scale_factor),
        #     nn.ReLU(inplace=True)
        # )

        self.upscale_stage1 = nn.Sequential(
            nn.Upsample(scale_factor=(scale_factor, scale_factor, 1), mode="trilinear", align_corners=False),
            nn.Conv3d(f, f, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.ReLU(inplace=True)
        )

        # 2. Refinement Block: These layers sharpen the anatomy
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
        """
        x: LR input tensor (B, C, D, H, W)
        target_shape: HR target shape (B, C, D_hr, H_hr, W_hr)
        """

        # -------------------------
        # Encoder
        # -------------------------
        down_1, p1 = self.down_conv1(x)
        down_2, p2 = self.down_conv2(p1)
        down_3, p3 = self.down_conv3(p2)

        # -------------------------
        # Bottleneck
        # -------------------------
        b = self.bottle_neck(p3)

        # -------------------------
        # Decoder
        # -------------------------
        up_2 = self.up_conv2(b, down_3)
        up_3 = self.up_conv3(up_2, down_2)
        up_4 = self.up_conv4(up_3, down_1)

        # -------------------------
        # Learned upsampling
        # -------------------------
        x_upscaled = self.upscale_stage1(up_4)
        # if self.scale_factor == 4:
        #     x_upscaled = self.upscale_stage2(x_upscaled)

        # -------------------------
        # HR refinement
        # -------------------------
        features_hr = self.refine1(x_upscaled)
        features_hr = self.refine2(features_hr)
        out_residual = self.final_conv(features_hr)

        # -------------------------
        # Residual shortcut (allowed interpolation)
        # -------------------------
        base_hr = F.interpolate(
            x,
            size=out_residual.shape[2:],
            mode="trilinear",
            align_corners=False
        )

        x = 0.5 * out_residual + base_hr

        # -------------------------
        # HARD shape alignment (no interpolation)
        # -------------------------
        _, _, D_t, H_t, W_t = target_shape

        x = x[:, :, :D_t, :H_t, :W_t]

        # -------------------------
        # Final safety check
        # -------------------------
        assert x.shape[2:] == (D_t, H_t, W_t), \
            f"SR output shape mismatch: {x.shape} vs {target_shape}"

        return x
