import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


class FourierLoss3D(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: Weighting factor between amplitude and phase loss.
               Default is 0.5 (equal weight) as per[cite: 185].
        """
        super(FourierLoss3D, self).__init__()
        self.alpha = alpha

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
        # Apply 3D FFT (over the last 3 dimensions: D, H, W)
        # Note: We use rfftn for efficiency if data is real-valued
        pred_fft = torch.fft.rfftn(pred, dim=(-3, -2, -1))
        target_fft = torch.fft.rfftn(target, dim=(-3, -2, -1))

        # 1. Amplitude Loss (L1 norm of absolute differences)
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        loss_amp = torch.mean(torch.abs(pred_amp - target_amp))

        # 2. Phase Loss (L1 norm of angular differences)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        # We use the min(angle difference) to handle periodicity [cite: 185]
        phase_diff = pred_phase - target_phase
        loss_phase = torch.mean(torch.abs(torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))))

        return self.alpha * loss_amp + (1 - self.alpha) * loss_phase

# Usage Example:
# criterion = FourierLoss3D()
# vol_pred = torch.randn(1, 1, 64, 64, 64)  # (Batch, Channel, D, H, W)
# vol_target = torch.randn(1, 1, 64, 64, 64)
# loss = criterion(vol_pred, vol_target)