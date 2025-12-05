import torch
import torch.nn as nn
from pytorch_msssim import ssim


class CombinedSSIML1Loss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.05, reduction='mean'):
        super().__init__()
        self.l1_loss_formal = nn.L1Loss(reduction=reduction)
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim

        # NOTE: SSIM loss is calculated as (1 - SSIM)

    def forward(self, pred, target):

        # Calculate L1 Loss
        loss_l1 = self.l1_loss_formal(pred, target) * self.lambda_l1

        # Calculate SSIM Loss
        n, c, d, h, w = pred.shape
        pred_2d = pred.permute(0, 1, 2, 3, 4).reshape(n * d, c, h, w)
        target_2d = target.permute(0, 1, 2, 3, 4).reshape(n * d, c, h, w)
        ssim_val = ssim(pred_2d, target_2d, data_range=1.0, size_average=True)
        loss_ssim = (1.0 - ssim_val) * self.lambda_ssim

        total_loss = loss_l1 + loss_ssim

        return total_loss, {
            'L1_Loss': loss_l1.item(),
            'SSIM_Loss': loss_ssim.item(),
            'SSIM_Score': ssim_val.item()
        }
