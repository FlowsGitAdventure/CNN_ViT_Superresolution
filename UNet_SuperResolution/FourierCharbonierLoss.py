import torch.nn as nn
from functions.CharbonnierLoss import CharbonnierLoss
from functions.FourierLoss3d import FourierLoss3D
from functions.SSIM3D import SSIM3D


class FourierCharbonier(nn.Module):
    def __init__(
            self,
            lambda_charb=0.8,
            lambda_fourier=0.6,
            lambda_ssim=0.2
    ):
        super().__init__()

        self.charb = CharbonnierLoss()
        self.fourier = FourierLoss3D()
        self.ssim = SSIM3D()

        self.wc = lambda_charb
        self.wf = lambda_fourier
        self.ws = lambda_ssim

    def forward(self, pred, target):
        loss_charb = self.charb(pred, target)
        loss_forier = self.fourier(pred, target)
        loss_ssim = self.ssim(pred, target)

        total = (
            self.wc * loss_charb +
            self.wf * loss_forier +
            self.ws * loss_ssim
        )

        return total, {
            "Charb": loss_charb.item(),
            "Four": loss_forier.item(),
            "SSIM": loss_ssim.item()
        }
