import torch
import torch.nn.functional as F


def calculate_psnr(super_image, hr_gt, data_range=None):
    # 1. Spatial Alignment (Targeting the last 3 dimensions: D, H, W)
    if super_image.shape[-3:] != hr_gt.shape[-3:]:
        super_image = F.interpolate(
            super_image,
            size=hr_gt.shape[-3:],
            mode='trilinear',
            align_corners=False
        )

    # 2. Dimensionality Matching
    if super_image.ndim == 6:
        super_image = super_image.squeeze(2)
    if hr_gt.ndim == 6:
        hr_gt = hr_gt.squeeze(2)

    # 3. Dynamic Data Range for Z-Scores
    if data_range is None:
        data_range = hr_gt.max() - hr_gt.min()

    # 4. MSE across all dimensions
    mse = torch.mean((super_image - hr_gt) ** 2)

    if mse == 0:
        return torch.tensor(float('inf'))

    # 5. PSNR Calculation
    psnr = 10. * torch.log10(data_range ** 2 / mse)

    return psnr
