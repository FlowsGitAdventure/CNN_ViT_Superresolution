import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def match_size(self, pred, target):
        if pred.shape != target.shape:
            pred = F.interpolate(
                pred,
                size=target.shape[-3:],
                mode="trilinear",
                align_corners=False
            )
        return pred

    def forward(self, pred, target):
        pred = self.match_size(pred, target)
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps))

