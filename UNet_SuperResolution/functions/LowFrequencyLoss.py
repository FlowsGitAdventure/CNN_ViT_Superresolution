import torch
import torch.nn as nn
import torch.nn.functional as F


def match_size(pred, target):
    if pred.shape != target.shape:
        pred = F.interpolate(
            pred,
            size=target.shape[-3:],
            mode="trilinear",
            align_corners=False
        )
    return pred


class LowFrequencyLoss(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.pool = nn.AvgPool3d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, pred, target):
        pred = match_size(pred, target)
        return F.l1_loss(self.pool(pred), self.pool(target))

