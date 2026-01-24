import torch
import torch.nn as nn
import torch.fft


class KSpaceMaskedFourierLoss3D(nn.Module):
    def __init__(
        self,
        scale_factor,
        reduction="mean"
    ):
        """
        scale_factor: same factor used during k-space truncation
                      (e.g. 2, 3, 4)
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.reduction = reduction

    def _measured_kspace_mask(self, shape, device):
        """
        Cartesian central k-space mask matching FFT-shifted cropping.
        shape: (D, H, W_rfft)
        """
        D, H, W = shape

        d = int(D / self.scale_factor["z"])
        h = int(H / self.scale_factor["y"])
        w = int(W / self.scale_factor["x"])

        mask = torch.zeros((D, H, W), device=device)

        d0 = (D - d) // 2
        h0 = (H - h) // 2
        w0 = 0  # rFFT: low frequencies start at index 0

        mask[d0:d0 + d, h0:h0 + h, w0:w0 + w] = 1.0
        return mask

    def forward(self, pred, target):
        """
        pred, target: (B, C, D, H, W)
        """

        # FFT (real-valued input)
        pred_k = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm="ortho")
        target_k = torch.fft.rfftn(target, dim=(-3, -2, -1), norm="ortho")

        pred_k = torch.fft.fftshift(pred_k, dim=(-3, -2))
        target_k = torch.fft.fftshift(target_k, dim=(-3, -2))

        # Build measured-frequency mask
        mask = self._measured_kspace_mask(
            pred_k.shape[-3:], pred.device
        )
        mask = mask[None, None, ...]

        # Data consistency loss ONLY where LR had measurements
        diff = (pred_k - target_k) * mask

        loss = torch.abs(diff)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("Invalid reduction")
