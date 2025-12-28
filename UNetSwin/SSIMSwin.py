import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def calculate_ssim_score(super_image, hr_gt, data_range=6.0):
    # 1. Spatial Alignment (Targeting Depth, Height, Width)
    # Using [-3:] ensures we ignore whether the tensor is 5D or 6D
    if super_image.shape[-3:] != hr_gt.shape[-3:]:
        super_image = F.interpolate(
            super_image,
            size=hr_gt.shape[-3:],
            mode='trilinear',
            align_corners=False
        )

    # 2. Universal Squeeze & Flatten Logic
    # We want to turn everything into (Total_Slices, 1, H, W)
    # This works whether input is (B, T, D, H, W) OR (B, T, C, D, H, W)

    # First, squeeze out any singleton channel dimensions (like C=1 from Swin)
    if super_image.ndim == 6:
        super_image = super_image.squeeze(2)  # Removes C if it's 6D
    if hr_gt.ndim == 6:
        hr_gt = hr_gt.squeeze(2)

    # Now treat (B, T, D) as one giant batch of slices
    # We calculate the product of all leading dimensions except the last two (H, W)
    h, w = hr_gt.shape[-2:]
    total_slices = hr_gt.numel() // (h * w)

    # Reshape to (N, 1, H, W) which is what the ssim library expects
    super_image_2d = super_image.reshape(total_slices, 1, h, w)
    hr_gt_2d = hr_gt.reshape(total_slices, 1, h, w)

    # 3. SSIM Calculation
    # We ensure data_range is a float for the library's compatibility
    return ssim(super_image_2d, hr_gt_2d, data_range=float(data_range), size_average=True)