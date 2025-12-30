import torch
import torch.nn.functional as F


def calculate_psnr(super_image, hr_gt, data_range):
    if super_image.shape[-3:] != hr_gt.shape[-3:]:
        super_image = F.interpolate(
            super_image,
            size=hr_gt.shape[-3:],
            mode='trilinear',
            align_corners=False
        )

    # Dimensionality Matching
    if super_image.ndim == 6:
        super_image = super_image.squeeze(2)
    if hr_gt.ndim == 6:
        hr_gt = hr_gt.squeeze(2)

    mse = torch.mean((super_image - hr_gt) ** 2)
    if mse < 1e-10:  # Avoid division by near-zero
        return torch.tensor(100.0).to(super_image.device)

    psnr = 10. * torch.log10(data_range ** 2 / mse)

    return psnr
