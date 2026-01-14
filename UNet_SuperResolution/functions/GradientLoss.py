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


class GradientLoss3D(nn.Module):
    def forward(self, pred, target):
        pred = match_size(pred, target)

        dz_p = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        dy_p = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dx_p = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])

        dz_t = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        dy_t = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        dx_t = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])

        return (
            F.l1_loss(dx_p, dx_t) +
            F.l1_loss(dy_p, dy_t) +
            F.l1_loss(dz_p, dz_t)
        )

