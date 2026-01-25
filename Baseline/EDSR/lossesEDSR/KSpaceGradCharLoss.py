import torch
import torch.nn as nn
from functionsAniso.CharbonnierLossAnisotropic import CharbonnierLoss
from functionsAniso.GradientLossAnisotropic import GradientLoss3D
from functionsAniso.KSpaceFourierLoss3dAnisotropic import KSpaceMaskedFourierLoss3D


def linear_ramp(progress, start, end):
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
        lambda_grad=0.5,
        lambda_kspace=0.3
    ):
        super().__init__()

        self.charb = CharbonnierLoss()
        self.grad = GradientLoss3D()

        # Scale factors MUST match your anisotropic downsampling
        # (z=2, y=2, x=1 is correct for your setup)
        self.kspace = KSpaceMaskedFourierLoss3D(
            scale_factors={"z": 2, "y": 2, "x": 1}
        )

        self.lambda_charb = lambda_charb
        self.lambda_grad = lambda_grad
        self.lambda_kspace = lambda_kspace

        self.epochs = epochs
        self.epoch = 0

    def forward(self, pred, target, spacing):
        """
        Assumes:
        - pred and target are ALREADY cropped to valid FOV
        - No spatial padding reaches this function
        """

        progress = (self.epoch / self.epochs) * 100.0

        grad_ramp = linear_ramp(progress, 30.0, 60.0)
        kspace_ramp = linear_ramp(progress, 60.0, 70.0)

        # 1. Spatial fidelity
        loss_charb = self.charb(pred, target)

        # 2. Anatomical sharpness (physical gradients)
        loss_grad = (
            self.grad(pred, target, spacing)
            if grad_ramp > 0.0
            else pred.new_tensor(0.0)
        )

        # 3. Frequency-domain data consistency
        loss_kspace = (
            self.kspace(pred, target)
            if kspace_ramp > 0.0
            else pred.new_tensor(0.0)
        )

        total = (
            self.lambda_charb * loss_charb +
            self.lambda_grad * grad_ramp * loss_grad +
            self.lambda_kspace * kspace_ramp * loss_kspace
        )

        return total, {
            "Total": total.item(),
            "Charb": loss_charb.item(),
            "Grad": loss_grad.item(),
            "KSpace": loss_kspace.item()
        }
