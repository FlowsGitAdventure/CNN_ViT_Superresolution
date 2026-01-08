import torch
import torch.nn.functional as F
import math


class StructuralSimilarityLoss3D:
    def __init__(self, data_range, device):
        self.data_range = data_range
        self.device = device

    def gaussian_kernel_3d(self, kernel_size, sigma):
        coords = torch.arange(kernel_size, device=self.device).float() - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        g3d = g[:, None, None] * g[None, :, None] * g[None, None, :]
        g3d = g3d / g3d.sum()

        return g3d

    def ssim_3d(self, x, y, kernel_size=7, sigma=1.5, K1=0.01, K2=0.03):
        """
        3D SSIM for volumetric data.

        Parameters
        ----------
        kernel_size : int
            Size of Gaussian kernel (odd number)
        sigma : float
            Gaussian sigma
        """

        assert x.shape == y.shape
        assert x.ndim == 5

        device = x.device
        channel = x.size(1)

        kernel = self.gaussian_kernel_3d(kernel_size, sigma, device)
        kernel = kernel.expand(channel, 1, kernel_size, kernel_size, kernel_size)

        padding = kernel_size // 2

        mu_x = F.conv3d(x, kernel, padding=padding, groups=channel)
        mu_y = F.conv3d(y, kernel, padding=padding, groups=channel)

        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv3d(x * x, kernel, padding=padding, groups=channel) - mu_x2
        sigma_y2 = F.conv3d(y * y, kernel, padding=padding, groups=channel) - mu_y2
        sigma_xy = F.conv3d(x * y, kernel, padding=padding, groups=channel) - mu_xy

        C1 = (K1 * self.data_range) ** 2
        C2 = (K2 * self.data_range) ** 2

        ssim_map = (
            (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        ) / (
            (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        )

        return ssim_map.mean()

    def temporal_ssim_3d(self, pred, target,):
        """
        Apply 3D SSIM per time step and average.
        """
        B, T, D, H, W = pred.shape
        ssim_total = 0.0

        for t in range(T):
            ssim_total += self.ssim_3d(
                pred[:, t:t + 1],  # (B, 1, D, H, W)
                target[:, t:t + 1]
            )

        return ssim_total / T

