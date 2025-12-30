import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class CombinedSSIML1Loss(nn.Module):
    # ToDo: Got rid of reduction in L1 check difference
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.2, lambda_temp=0.1, data_range=6.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_temp = lambda_temp
        self.data_range = data_range

    def forward(self, pred, target):
        if pred.ndim == 6:
            pred = pred.squeeze(2)
        if target.ndim == 6:
            target = target.squeeze(2)

        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[-3:], mode='trilinear', align_corners=False)

        # L1 Loss calculation
        loss_l1 = self.l1(pred, target)

        # SSIM Loss calculation
        h, w = target.shape[-2:]
        total_slices = target.numel() // (h * w)
        pred_2d = pred.reshape(total_slices, 1, h, w)
        target_2d = target.reshape(total_slices, 1, h, w)

        # Temporal Consistency Loss - Only calculate if Time > 1
        if pred.shape[1] > 1:
            diff_pred = pred[:, 1:] - pred[:, :-1]
            diff_target = target[:, 1:] - target[:, :-1]
            loss_temp = self.l1(diff_pred, diff_target)
        else:
            loss_temp = torch.tensor(0.0).to(pred.device)

        ssim_val = ssim(pred_2d, target_2d, data_range=self.data_range, size_average=True)
        loss_ssim = 1.0 - ssim_val

        total_loss = (self.lambda_l1 * loss_l1) + \
                     (self.lambda_ssim * loss_ssim) + \
                     (self.lambda_temp * loss_temp)

        return total_loss, {
            "L1": loss_l1.item(),
            "SSIM": ssim_val.item(),
            "Temp": loss_temp.item()
        }