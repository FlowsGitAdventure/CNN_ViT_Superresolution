import torch
import torch.nn as nn
from DoubleConvolution import DoubleConv



class HybridEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # --- CNN PATH ---
        # We use in_channels=1 because we process each time-step
        # as a separate item in the batch to learn shared spatial features.
        self.cnn_path = nn.Sequential(
            DoubleConv(1, out_channels),
            DoubleConv(out_channels, out_channels),
        )

        # --- ViT PATH ---
        # CHANGE: Use TransformerEncoderLayer (does not require 'memory')
        self.vit_path = nn.TransformerEncoderLayer(
            d_model=in_channels,  # d_model is the sequence length (Time Steps)
            nhead=2 if in_channels % 2 == 0 else 1,
            dim_feedforward=in_channels * 4,
            batch_first=True
        )

        # --- FUSION ---
        # Fusion projects (CNN_out + ViT_out) back to out_channels
        self.fusion_gate = nn.Sequential(
            nn.Conv3d(out_channels + in_channels, out_channels + in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.final_conv = nn.Conv3d(out_channels + in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: (Batch, Time, D, H, W)
        b, t, d, h, w = x.shape

        # 1. CNN Path (Spatial Anatomy)
        # Reshape to (Batch*Time, 1, D, H, W)
        x_cnn = x.view(b * t, 1, d, h, w)
        spatial_info = self.cnn_path(x_cnn)

        # Aggregate across time (Mean) to get one 3D volume of features
        spatial_info = spatial_info.view(b, t, -1, d, h, w).mean(dim=1)

        # 2. ViT Path (Temporal/Global Dynamics)
        # Flatten voxels: (B, Time, D*H*W) -> (B, Voxels, Time)
        tokens = x.view(b, t, -1).permute(0, 2, 1)

        # SUCCESS: TransformerEncoderLayer only needs 'tokens'
        global_info = self.vit_path(tokens)

        # Reshape back to (Batch, Time, D, H, W)
        global_info = global_info.permute(0, 2, 1).view(b, t, d, h, w)

        # 3. Gated Fusion
        combined = torch.cat([spatial_info, global_info], dim=1)
        gate = self.fusion_gate(combined)

        return self.final_conv(combined * gate)