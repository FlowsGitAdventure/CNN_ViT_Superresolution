import torch.nn.functional as F
from pytorch_msssim import ssim


def calculate_ssim_score(super_image, hr_gt, data_range=6.0):  # Adjusted for Z-score range
    # 1. Shape Alignment
    if super_image.shape != hr_gt.shape:
        super_image = F.interpolate(super_image, size=hr_gt.shape[2:], mode='trilinear', align_corners=False)

    # 2. Handling 5D (Batch, Time, Depth, Height, Width)
    # We want to treat every slice of every timepoint as an image
    N, T, D, H, W = super_image.shape

    # Reshape to (N * T * D, 1, H, W)
    # This stacks all timepoints and all slices together for a global average
    super_image_2d = super_image.reshape(N * T * D, 1, H, W)
    hr_gt_2d = hr_gt.reshape(N * T * D, 1, H, W)

    # 3. SSIM Calculation
    # data_range 6.0 is a safe estimate for Z-normalized data (-3 to +3)
    ssim_score = ssim(super_image_2d, hr_gt_2d, data_range=data_range, size_average=True)

    return ssim_score