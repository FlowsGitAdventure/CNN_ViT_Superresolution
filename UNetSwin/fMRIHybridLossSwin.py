import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class HybridLossSSIML1Temp(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.2, lambda_temp=0.1, data_range=6.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_temp = lambda_temp
        self.data_range = data_range

    def forward(self, pred, target):
        # 1. Shape Alignment
        # If Swin output is (B, T, 1, D, H, W) and target is (B, T, D, H, W)
        if pred.ndim == 6 and target.ndim == 5:
            pred = pred.squeeze(2)

        if target.ndim == 6 and pred.ndim == 5:
            target = target.squeeze(2)

        if pred.shape[-3:] != target.shape[-3:]:
            pred = F.interpolate(pred, size=target.shape[-3:], mode='trilinear', align_corners=False)

        # 2. Spatial Loss (L1) - Voxel-wise intensity
        loss_l1 = self.l1(pred, target)

        # 3. Structural Loss (SSIM) - Anatomy & Window boundaries
        # Reshape everything into a giant stack of 2D slices: (Total_Slices, 1, H, W)
        h, w = target.shape[-2:]
        total_slices = target.numel() // (h * w)

        pred_2d = pred.reshape(total_slices, 1, h, w)
        target_2d = target.reshape(total_slices, 1, h, w)

        ssim_val = ssim(pred_2d, target_2d, data_range=self.data_range, size_average=True)
        loss_ssim = 1.0 - ssim_val

        # 4. Temporal Consistency Loss - BOLD Signal preservation
        # Ensure we are comparing the temporal change between volumes
        if pred.shape[1] > 1:  # Only if T > 1
            diff_pred = pred[:, 1:] - pred[:, :-1]
            diff_target = target[:, 1:] - target[:, :-1]
            loss_temp = self.l1(diff_pred, diff_target)
        else:
            loss_temp = torch.tensor(0.0).to(pred.device)

        # 5. Total Combined Loss
        total_loss = (self.lambda_l1 * loss_l1) + \
                     (self.lambda_ssim * loss_ssim) + \
                     (self.lambda_temp * loss_temp)

        return total_loss, {
            "L1": loss_l1.item(),
            "SSIM": ssim_val.item(),
            "Temp": loss_temp.item()
        }