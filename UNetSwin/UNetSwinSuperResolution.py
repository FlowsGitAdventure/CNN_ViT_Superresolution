import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer
from UpSampleSwin import UpSample


class MonaiSwinSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=24, window_size=(4, 4, 4)):
        super().__init__()
        f = base_filters

        # 1. MONAI Swin Backbone
        self.swin_backbone = SwinTransformer(
            in_chans=in_channels,
            embed_dim=f,
            window_size=window_size,
            patch_size=(2, 2, 2),
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            spatial_dims=3,
            norm_layer=nn.LayerNorm,
            downsample="merging"
        )

        # 2. FIXED Decoder Channel Logic
        # features[4] has f*16 channels (384 if f=24)
        # features[3] has f*8 channels (192)
        # features[2] has f*4 channels (96)
        # features[1] has f*2 channels (48)
        # features[0] has f channels (24)

        # Level 4 to Level 2 (Skipping level 3 as it's the same spatial size)
        self.up1 = UpSample(in_channels=f * 16, skip_channels=f * 4, out_channels=f * 8)

        # Level 2 to Level 1
        self.up2 = UpSample(in_channels=f * 8, skip_channels=f * 2, out_channels=f * 4)

        # Level 1 to Level 0 (Patch Embed)
        self.up3 = UpSample(in_channels=f * 4, skip_channels=f, out_channels=f * 2)

        # 3. Final Super-Res Upscale
        # Final output of up3 is f*2 (48). We'll process this to get our final output.
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose3d(f * 2, f, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(f, f, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(f, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x: (B, T, D, H, W)
        b, t, d, h, w = x.shape
        x_in = x.reshape(b * t, 1, d, h, w)

        # 4. Encoder
        features = self.swin_backbone(x_in)
        # features[4] = 1x1x1, [3] = 1x1x1, [2] = 2x2x2, [1] = 4x4x4, [0] = 8x8x8

        # 5. Decoder Logic
        # We skip features[3] because it's the same spatial size as features[4]
        u1 = self.up1(features[4], features[2])  # 1x1x1 up to 2x2x2
        u2 = self.up2(u1, features[1])  # 2x2x2 up to 4x4x4
        u3 = self.up3(u2, features[0])  # 4x4x4 up to 8x8x8

        # 6. Final Super-Res
        feat_hr = self.final_upsample(u3)  # 8x8x8 up to 32x32x32 (if D=16, 2x SR = 32)

        # 7. Reshape and Residual
        feat_hr = feat_hr.view(b, t, -1, *feat_hr.shape[2:])

        # Target shape for residual is the output of final_upsample
        x_res = F.interpolate(x, size=feat_hr.shape[3:], mode='trilinear').unsqueeze(2)

        return feat_hr + x_res