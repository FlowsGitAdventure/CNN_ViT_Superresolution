import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def calculate_ssim_score(super_image, hr_gt, mask, data_range):
    """
    Calculates SSIM only on valid (non-padded) slices.
    """
    # 2. Universal Squeeze (Removing singleton dimensions)
    if super_image.ndim == 6:
        super_image = super_image.squeeze(2)
    if hr_gt.ndim == 6:
        hr_gt = hr_gt.squeeze(2)
    if mask.ndim == 6:
        mask = mask.squeeze(2)

    # 3. Reshape to 2D batches: (Total_Slices, 1, H, W)
    # We treat Depth and Batch as the same dimension for 2D SSIM calculation
    h, w = hr_gt.shape[-2:]
    total_slices = hr_gt.numel() // (h * w)

    super_image_2d = super_image.reshape(total_slices, 1, h, w)
    hr_gt_2d = hr_gt.reshape(total_slices, 1, h, w)
    mask_2d = mask.reshape(total_slices, 1, h, w)

    # 4. Calculate Slice-wise SSIM
    # We set size_average=False to get a score for every individual slice
    ssim_per_slice = ssim(
        super_image_2d,
        hr_gt_2d,
        data_range=data_range,
        size_average=False
    )

    # 5. Masked Average
    # A slice is valid if any voxel in the mask for that slice is 1.0
    # slice_mask results in a 1D tensor of booleans [Total_Slices]
    slice_mask = mask_2d.sum(dim=(1, 2, 3)) > 0

    if slice_mask.sum() == 0:
        return torch.tensor(0.0).to(super_image.device)

    # Average only the SSIM scores of valid slices
    valid_ssim_scores = ssim_per_slice[slice_mask]
    return valid_ssim_scores.mean().cpu().item()