import torch.nn as nn


# ToDo: Add a mode option for the different models
# If it is CNN only us Instance norm, the rest Layer norm (if needed).
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, dropout_prob=0.0):
        super().__init__()

        # Layer 1
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        if dropout_prob > 0:
            layers.append(nn.Dropout3d(p=dropout_prob))

        # Layer 2
        layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)
