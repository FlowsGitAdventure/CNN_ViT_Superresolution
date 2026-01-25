import torch
import torch.nn as nn
import torch.nn.functional as F
from functions.CharbonnierLoss import CharbonnierLoss
from functions.GradientLoss import GradientLoss3D
from functions.KSpaceFourierLoss3d import KSpaceMaskedFourierLoss3D


def linear_ramp(progress, start, end):
    """
    Linearly ramps from 0 to 1 between start and end (in percent).
    """
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / (end - start)


def center_crop_or_pad(x, target_shape):
    _, _, D, H, W = x.shape
    Dt, Ht, Wt = target_shape

    # Pad if needed
    pad_d = max(Dt - D, 0)
    pad_h = max(Ht - H, 0)
    pad_w = max(Wt - W, 0)

    if pad_d or pad_h or pad_w:
        x = F.pad(
            x,
            (
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2,
            )
        )

    # Crop if needed
    _, _, D, H, W = x.shape
    d0 = (D - Dt) // 2
    h0 = (H - Ht) // 2
    w0 = (W - Wt) // 2

    return x[:, :, d0:d0 + Dt, h0:h0 + Ht, w0:w0 + Wt]


class FourierCharbonier(nn.Module):

    def __init__(
        self,
        epochs,
        lambda_charb=1.0,
        lambda_grad=0.3,
        lambda_kspace=0.3,
    ):
        super().__init__()

        self.charb = CharbonnierLoss()
        self.grad = GradientLoss3D()
        self.kspace = KSpaceMaskedFourierLoss3D(scale_factor=4)

        # Maximum weights (used after ramp-up)
        self.lambda_charb = lambda_charb
        self.lambda_grad = lambda_grad
        self.lambda_kspace = lambda_kspace

        self.epochs = epochs
        self.epoch = 0

    def forward(self, pred, target):
        pred = center_crop_or_pad(pred, target.shape[-3:])

        # Progress in percent
        progress = (self.epoch / self.epochs) * 100.0

        # Loss ramps (percent-based)
        grad_ramp = linear_ramp(progress, 30.0, 60.0)
        kspace_ramp = linear_ramp(progress, 70.0, 90.0)

        # Dynamic weights
        w_grad = self.lambda_grad * grad_ramp
        w_kspace = self.lambda_kspace * kspace_ramp

        # Charbonnier dominates early, decreases later
        w_charb = self.lambda_charb - (w_grad + w_kspace)
        w_charb = max(w_charb, 0.2)  # safety floor

        # Individual losses
        loss_charb = self.charb(pred, target)

        loss_grad = (
            self.grad(pred, target) if grad_ramp > 0.0
            else pred.new_tensor(0.0)
        )

        loss_kspace = (
            self.kspace(pred, target) if kspace_ramp > 0.0
            else pred.new_tensor(0.0)
        )

        # Total loss
        total = (
            w_charb * loss_charb
            + w_grad * loss_grad
            + w_kspace * loss_kspace
        )

        return total, {
            "Total": total.item(),
            "Charb": loss_charb.item(),
            "Grad": loss_grad.item(),
            "KSpace": loss_kspace.item()
        }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate anisotropic SR
    B, C = 2, 1
    hr = torch.randn(B, C, 64, 128, 128, device=device)

    # Intentionally mismatched prediction
    pred = torch.randn(B, C, 60, 130, 126, device=device, requires_grad=True)

    criterion = FourierCharbonier(20).to(device)

    loss, components = criterion(pred, hr)
    criterion.epoch = 19

    print("Loss components:", components)
    loss.backward()

    print("Backward pass successful.")
