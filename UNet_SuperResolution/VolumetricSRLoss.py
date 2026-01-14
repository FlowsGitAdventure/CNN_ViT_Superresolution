import torch.nn as nn
from functions.CharbonnierLoss import CharbonnierLoss
from functions.SSIM3D import SSIM3D
from functions.GradientLoss import GradientLoss3D
from functions.LowFrequencyLoss import LowFrequencyLoss


class VolumetricSRLoss(nn.Module):
    def __init__(
        self,
        lambda_charb=1.0,
        lambda_ssim=0.6,
        lambda_grad=0.15,
        lambda_lowfreq=0.1,
        data_range=1.0
    ):
        super().__init__()

        self.charb = CharbonnierLoss()
        self.ssim = SSIM3D(data_range=data_range)
        self.grad = GradientLoss3D()
        self.lowfreq = LowFrequencyLoss()

        self.wc = lambda_charb
        self.ws = lambda_ssim
        self.wg = lambda_grad
        self.wl = lambda_lowfreq

    def forward(self, pred, target):
        loss_charb = self.charb(pred, target)
        loss_ssim = 1.0 - self.ssim(pred, target)
        loss_grad = self.grad(pred, target)
        loss_lowf = self.lowfreq(pred, target)

        total = (
            self.wc * loss_charb +
            self.ws * loss_ssim +
            self.wg * loss_grad +
            self.wl * loss_lowf
        )

        return total, {
            "Charb": loss_charb.item(),
            "SSIM3D": 1.0 - loss_ssim.item(),
            "Grad": loss_grad.item(),
            "LowF": loss_lowf.item()
        }
