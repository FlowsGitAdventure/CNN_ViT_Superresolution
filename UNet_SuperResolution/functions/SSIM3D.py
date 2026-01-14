import torch
from torch import nn
import torch.nn.functional as F


def create_gaussian_window_3d(window_size, sigma, device, channels):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    g3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
    window = g3d.unsqueeze(0).unsqueeze(0)
    return window.repeat(channels, 1, 1, 1, 1)


def match_size(pred, target):
    if pred.shape != target.shape:
        pred = F.interpolate(
            pred,
            size=target.shape[-3:],
            mode="trilinear",
            align_corners=False
        )
    return pred


class SSIM3D(nn.Module):
    def __init__(self, window_size=7, sigma=1.5, data_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range

    def forward(self, x, y):
        x = match_size(x, y)

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        window = create_gaussian_window_3d(
            self.window_size, self.sigma, x.device, x.shape[1]
        )

        mu_x = F.conv3d(x, window, padding=self.window_size // 2, groups=x.shape[1])
        mu_y = F.conv3d(y, window, padding=self.window_size // 2, groups=y.shape[1])

        sigma_x = F.conv3d(x * x, window, padding=self.window_size // 2, groups=x.shape[1]) - mu_x ** 2
        sigma_y = F.conv3d(y * y, window, padding=self.window_size // 2, groups=y.shape[1]) - mu_y ** 2
        sigma_xy = F.conv3d(x * y, window, padding=self.window_size // 2, groups=x.shape[1]) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        return ssim_map.mean()

