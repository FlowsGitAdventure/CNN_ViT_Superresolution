import torch.nn as nn
from functions.CharbonnierLoss import CharbonnierLoss
from functions.GradientLossAnisotropic import GradientLoss3D
from functions.KSpaceFourierLoss3dAnisotropic import KSpaceMaskedFourierLoss3D


def linear_ramp(progress, start, end):
    """
    Linearly ramps from 0 to 1 between start and end (in percent).
    """
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return (progress - start) / (end - start)


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

        scale_factor = {
            "x": 2,
            "y": 2,
            "z": 1
        }
        self.kspace = KSpaceMaskedFourierLoss3D(scale_factor)

        # Maximum weights (used after ramp-up)
        self.lambda_charb = lambda_charb
        self.lambda_grad = lambda_grad
        self.lambda_kspace = lambda_kspace

        self.epochs = epochs
        self.epoch = 0

    def forward(self, pred, target, spacing):
        # pred = center_crop_or_pad(pred, target.shape[-3:])

        # Progress in percent
        progress = (self.epoch / self.epochs) * 100.0

        # Loss ramps (percent-based)
        grad_ramp = linear_ramp(progress, 10.0, 20.0)   # 30 - 60
        kspace_ramp = linear_ramp(progress, 10.0, 20.0)   # 60 - 80

        # Dynamic weights
        w_grad = self.lambda_grad * grad_ramp
        w_kspace = self.lambda_kspace * kspace_ramp

        # Charbonnier dominates early, decreases later
        w_charb = self.lambda_charb - (w_grad + w_kspace)
        w_charb = max(w_charb, 0.2)  # safety floor

        # Individual losses
        loss_charb = self.charb(pred, target)

        loss_grad = (
            self.grad(pred, target, spacing) if grad_ramp > 0.0
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