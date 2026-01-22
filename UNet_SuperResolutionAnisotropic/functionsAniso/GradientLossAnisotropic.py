import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss3D(nn.Module):
    def __init__(self, weights=(1.0, 1.0, 1.0), eps=1e-6):
        super().__init__()
        self.wz, self.wy, self.wx = weights
        self.eps = eps

    def forward(self, pred, target, spacing):
        """
        spacing: (sz, sy, sx) in mm (tensor or tuple)
        """
        # if torch.is_tensor(spacing):
        #     sz, sy, sx = spacing[0]
        # else:
        #     sz, sy, sx = spacing

        if torch.is_tensor(spacing):
            sz, sy, sx = spacing.view(-1)  # flatten to 3 elements
        else:
            sz, sy, sx = spacing

        # finite differences
        dz_p = (pred[:, :, 1:] - pred[:, :, :-1]) / sz
        dy_p = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) / sy
        dx_p = (pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]) / sx

        dz_t = (target[:, :, 1:] - target[:, :, :-1]) / sz
        dy_t = (target[:, :, :, 1:] - target[:, :, :, :-1]) / sy
        dx_t = (target[:, :, :, :, 1:] - target[:, :, :, :, :-1]) / sx

        # nan-safe sqrt
        loss_z = torch.sqrt(torch.clamp((dz_p - dz_t) ** 2, min=self.eps)).mean()
        loss_y = torch.sqrt(torch.clamp((dy_p - dy_t) ** 2, min=self.eps)).mean()
        loss_x = torch.sqrt(torch.clamp((dx_p - dx_t) ** 2, min=self.eps)).mean()

        return self.wz * loss_z + self.wy * loss_y + self.wx * loss_x
