import torch
import torch.nn as nn


class HybridEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.cnn_path = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.vit_path = nn.TransformerDecoderLayer(
            d_model=in_channels,
            nhead=2 if in_channels % 2 == 0 else 1,
            dim_feedforward=in_channels*2,
            batch_first=True
        )

        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Sigmoid()
        )

        self.final_conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        b, t, d, h, w = x.shape

        x_cnn = x.view(b * t, 1, d, h, w)
        spatial_info = self.cnn_path(x_cnn)
        spatial_info = spatial_info.view(b, t, -1, d, h, w).mean(dim=1)

        tokens = x.view(b, t, -1).permute(0, 2, 1)
        global_info = self.vit_path(tokens)
        global_info = global_info.permute(0, 2, 1).view(b, t, d, h, w)

        combined = torch.cat([spatial_info, global_info], dim=1)
        gate = self.fusion(combined)

        combined_gated = self.final_conv(combined * gate)

        return combined_gated
