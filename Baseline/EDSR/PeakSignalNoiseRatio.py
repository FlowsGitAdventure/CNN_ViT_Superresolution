import torch
import torch.nn.functional as F

def calculate_psnr(super_image, hr_gt, mask, data_range):
    """
    super_image: Model output (B, C, D, H, W)
    hr_gt: Ground truth (B, C, D, H, W)
    mask: Binary mask (B, C, D, H, W)
    data_range: Peak value (usually 1.0)
    """
    # 2. Masked MSE Calculation
    # We square the difference, apply the mask to keep only real voxels,
    # then divide by the sum of the mask.
    diff_sq = (super_image - hr_gt) ** 2
    masked_mse = torch.sum(diff_sq * mask) / (torch.sum(mask) + 1e-10)

    # 3. PSNR Calculation
    if masked_mse < 1e-10:
        return torch.tensor(100.0).to(super_image.device)

    psnr = 10. * torch.log10((data_range ** 2) / masked_mse)

    return psnr.cpu().item()