from torch import nn
import torch.nn.functional as F


class ExpansinonResBlock(nn.Module):
    def __init__(self, in_channels, inter_channels, downsampling_layer=None, stride=1):
        super().__init__()
        expansion = 4

        # conv 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)

        # conv 2
        self.conv2 = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)

        # conv 3
        self.conv3 = nn.Conv2d(in_channels=inter_channels, out_channels=inter_channels * expansion, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_channels * expansion)

        self.relu = nn.ReLU()

        self.downsampling_layer = downsampling_layer

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # downsample
        if self.downsampling_layer is not None:
            res = self.downsampling_layer(res)

        out += res
        out = F.relu(out)

        return out
