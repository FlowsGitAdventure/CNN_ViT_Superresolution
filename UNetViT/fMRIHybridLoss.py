import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim


class HybridLossSSIML1Temp(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.2, lambda_temp=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()

        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_temp = lambda_temp

    def forward(self, pred, target):
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:], mode='trilinear', align_corners=False)

        # 1. Spatial Loss (L1)
        loss_l1 = self.l1(pred, target)

        # 2. Structural Loss (SSIM)
        b, t, d, h, w = pred.shape
        pred_2d = pred.reshape(b * t * d, 1, h, w)
        target_2d = target.reshape(b * t * d, 1, h, w)

        # Note: If Z-normalized, data_range should be adjusted (e.g., 6.0)
        ssim_val = ssim(pred_2d, target_2d, data_range=6.0, size_average=True)
        loss_ssim = 1.0 - ssim_val

        # 3. Temporal Consistency Loss
        # Penalize differences in the temporal derivative
        diff_pred = pred[:, 1:] - pred[:, :-1]
        diff_target = target[:, 1:] - target[:, :-1]
        loss_temp = self.l1(diff_pred, diff_target)

        total_loss = (self.lambda_l1 * loss_l1) + \
                     (self.lambda_ssim * loss_ssim) + \
                     (self.lambda_temp * loss_temp)

        return total_loss, {"L1": loss_l1.item(), "SSIM": ssim_val.item(), "Temp": loss_temp.item()}