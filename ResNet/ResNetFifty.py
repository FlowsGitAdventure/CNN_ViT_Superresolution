from torch import nn
from ResNet.ResidualBlock import ExpansionResBlock


class ResNetFifty(nn.Module):
    def __init__(self, num_blocks, img_channels, num_classes, use_final=True):
        super().__init__()
        self.expansion = 4
        self.use_final = use_final
        self.in_channels = 64

        # Input Conv
        self.conv1 = nn.Conv2d(img_channels, self.in_channels, kernel_size=7, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        # Residual Layers
        self.layer1 = self._layer(ExpansionResBlock, num_blocks[0], stride=1, inter_channels=64)
        self.layer2 = self._layer(ExpansionResBlock, num_blocks[1], stride=2, inter_channels=128)
        self.layer3 = self._layer(ExpansionResBlock, num_blocks[2], stride=2, inter_channels=256)
        self.layer4 = self._layer(ExpansionResBlock, num_blocks[3], stride=2, inter_channels=512)

        # Final Layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        if use_final:
            self.fc = nn.Linear(512 * self.expansion, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)

        if self.use_final:
            out = self.fc(out)

        return out

    def _layer(self, block, num_blocks, stride, inter_channels):
        layers = []
        downsampling_layer = None
        if stride != 1 or self.in_channels != inter_channels * self.expansion:
            downsampling_layer = nn.Sequential(
                nn.Conv2d(self.in_channels, inter_channels * self.expansion, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(inter_channels * self.expansion)
            )

        layers.append(block(self.in_channels, inter_channels, downsampling_layer, stride))
        self.in_channels = inter_channels * self.expansion

        for _ in range(num_blocks -1):
            layers.append(block(self.in_channels, inter_channels))

        return nn.Sequential(*layers)


def res_net_fifty(img_channels, num_classes, device='cuda', use_final=True):
    if use_final and num_classes is None:
        raise Exception('when use_final is set to true, it is necessary to give an integer value to num_classes')

    model = ResNetFifty(
        num_blocks=[3, 4, 6, 3],
        img_channels=img_channels,
        num_classes=num_classes if use_final else None,
        use_final=use_final
    ).to(device=device)

    return model





