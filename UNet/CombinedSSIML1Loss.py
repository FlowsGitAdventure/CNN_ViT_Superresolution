import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class CombinedSSIML1Loss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.05, reduction='mean'):
        super().__init__()
        self.l1_loss_formal = nn.L1Loss(reduction=reduction)
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim

    def forward(self, pred, target):
        # This is the "Best Option": Dynamically match the model output to the HR file
        # Even if the math results in 224 and the file is 208, this fixes it
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:], mode='trilinear', align_corners=False)

        # L1 Loss calculation
        loss_l1 = self.l1_loss_formal(pred, target) * self.lambda_l1

        # SSIM Loss calculation
        n, c, d, h, w = pred.shape
        # Permute (N, C, D, H, W) -> (N, D, C, H, W) for slice-wise 2D SSIM
        pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)
        target_2d = target.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)

        ssim_val = ssim(pred_2d, target_2d, data_range=1.0, size_average=True)
        loss_ssim = (1.0 - ssim_val) * self.lambda_ssim

        total_loss = loss_l1 + loss_ssim
        return total_loss, {'L1': loss_l1.item(), 'SSIM': ssim_val.item()}