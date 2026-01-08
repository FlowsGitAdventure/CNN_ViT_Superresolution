import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from functions.SobelEdgeLoss3D import SobelEdgeLoss3D
from functions.StructuralSimilarityLoss3d import StructuralSimilarityLoss3D


class CombinedSSIML1Loss(nn.Module):
    # ToDo: Get rid of reduction in L1 check difference
    def __init__(self, device, epochs, lambda_l1=0.5, lambda_ssim=0.8, lambda_temp=0.05, lambda_edge=0.3, data_range=1.0):
        super().__init__()
        self.device = device
        self.epochs = epochs
        self.sobel_edge = SobelEdgeLoss3D(device)
        self.ssim = StructuralSimilarityLoss3D(data_range, device)
        self.l1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_temp = lambda_temp
        self.lambda_edge = lambda_edge
        self.data_range = data_range

    def charbonnier(self, x, y, eps=1e-3):
        return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))

    def forward(self, pred, target, current_epoch):
        if pred.ndim == 6:
            pred = pred.squeeze(2)
        if target.ndim == 6:
            target = target.squeeze(2)

        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[-3:], mode='trilinear', align_corners=False)

        progress = (current_epoch / self.epochs) * 100

        # Charbonnier Loss calculation
        loss_cbn = self.charbonnier(pred, target)

        # SSIM Loss calculation
        if progress < 75:
            h, w = target.shape[-2:]
            total_slices = target.numel() // (h * w)
            pred_2d = pred.reshape(total_slices, 1, h, w)
            target_2d = target.reshape(total_slices, 1, h, w)

            ssim_val = ssim(pred_2d, target_2d, data_range=self.data_range, size_average=True)
            loss_ssim = 1.0 - ssim_val
        else:
            loss_ssim = 1.0 - self.ssim.temporal_ssim_3d(pred, target)

        edge_warmup_percent = 10
        if progress < edge_warmup_percent:
            current_lambda_edge = (progress / edge_warmup_percent) * self.lambda_edge
        else:
            current_lambda_edge = self.lambda_edge

        edge_loss = self.sobel_edge.forward(pred, target)

        # Temporal Consistency Loss - Only calculate if Time > 1
        if pred.shape[1] > 1:
            diff_pred = pred[:, 1:] - pred[:, :-1]
            diff_target = target[:, 1:] - target[:, :-1]
            diff_pred = diff_pred / (diff_target.abs().mean() + 1e-6)
            diff_target = diff_target / (diff_target.abs().mean() + 1e-6)
            loss_temp = self.l1(diff_pred, diff_target)
        else:
            loss_temp = torch.tensor(0.0).to(pred.device)

        total_loss = (self.lambda_l1 * loss_cbn) + \
                     (self.lambda_ssim * loss_ssim) + \
                     (self.lambda_temp * loss_temp) + \
                     (current_lambda_edge * edge_loss)

        return total_loss, {
            "L1": loss_cbn.item(),
            "SSIM": loss_ssim.item(),
            "Temp": loss_temp.item(),
            "Edge": edge_loss.item(),
            "Active_Edge_Lambda": current_lambda_edge  # Helpful for logging
        }