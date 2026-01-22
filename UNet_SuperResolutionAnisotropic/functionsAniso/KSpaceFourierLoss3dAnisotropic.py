import torch
import torch.nn as nn
import torch.fft


class KSpaceMaskedFourierLoss3D(nn.Module):
    def __init__(self, scale_factors, reduction="mean"):
        """
        scale_factors: dict, e.g. {"z": 2, "y": 2, "x": 1}
        """
        super().__init__()
        self.scale_factors = scale_factors
        self.reduction = reduction

    def _measured_kspace_mask(self, shape, device):
        D, H, W = shape

        d_size = D if self.scale_factors["z"] <= 1 else int(D / self.scale_factors["z"])
        h_size = H if self.scale_factors["y"] <= 1 else int(H / self.scale_factors["y"])
        w_size = W if self.scale_factors["x"] <= 1 else int(W / self.scale_factors["x"])

        mask = torch.zeros((D, H, W), device=device)

        d0 = (D - d_size) // 2
        h0 = (H - h_size) // 2
        w0 = 0  # rFFT

        mask[d0:d0 + d_size, h0:h0 + h_size, w0:w0 + w_size] = 1.0
        return mask

    def forward(self, pred, target):
        pred_k = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm="ortho")
        target_k = torch.fft.rfftn(target, dim=(-3, -2, -1), norm="ortho")

        pred_k = torch.fft.fftshift(pred_k, dim=(-3, -2))
        target_k = torch.fft.fftshift(target_k, dim=(-3, -2))

        mask = self._measured_kspace_mask(pred_k.shape[-3:], pred.device)
        mask = mask[None, None, ...]

        diff = torch.abs(pred_k - target_k) * mask

        if self.reduction == "mean":
            return diff.sum() / (mask.sum() + 1e-8)
        else:
            return diff.sum()
