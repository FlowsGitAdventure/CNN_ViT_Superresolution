import torch
import torch.nn.functional as F

def calculate_psnr(super_image, hr_gt, data_range=6.0):
    # 1. Spatial Alignment (Targeting Depth, Height, Width)
    if super_image.shape[-3:] != hr_gt.shape[-3:]:
        super_image = F.interpolate(
            super_image,
            size=hr_gt.shape[-3:],
            mode='trilinear',
            align_corners=False
        )

    # 2. Dimensionality Squeeze
    # Ensure both are reduced to the same number of dimensions (T, D, H, W)
    # This prevents broadcasting errors between (T, 1, D, H, W) and (T, D, H, W)
    super_image = super_image.squeeze()
    hr_gt = hr_gt.squeeze()

    # 3. MSE across all dimensions
    mse = torch.mean((super_image - hr_gt) ** 2)

    if mse == 0:
        return torch.tensor(float('inf'))

    # 4. PSNR Calculation
    psnr = 10. * torch.log10(torch.tensor(data_range)**2 / mse)

    return psnr