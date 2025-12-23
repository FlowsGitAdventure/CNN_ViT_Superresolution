import torch
import torch.nn.functional as F


def calculate_psnr(super_image, hr_gt, data_range=6.0):  # Adjusted for Z-score range
    # 1. Shape Alignment
    if super_image.shape != hr_gt.shape:
        super_image = F.interpolate(super_image, size=hr_gt.shape[2:], mode='trilinear', align_corners=False)

    # 2. MSE across all dimensions (Time, Depth, Height, Width)
    mse = torch.mean((super_image - hr_gt) ** 2)

    if mse == 0:
        return torch.tensor(float('inf'))

    # 3. PSNR Calculation
    # If data_range is 1.0 but your Z-scores go to 3.0, the PSNR will be wrong.
    psnr = 10. * torch.log10(data_range ** 2 / mse)

    return psnr