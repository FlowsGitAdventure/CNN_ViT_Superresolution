import torch.nn.functional as F
from pytorch_msssim import ssim

def calculate_ssim_score(super_image, hr_gt, data_range=1.0):
    # Ensure super_image matches hr_gt spatial dimensions
    if super_image.shape != hr_gt.shape:
        super_image = F.interpolate(super_image, size=hr_gt.shape[2:], mode='trilinear', align_corners=False)
        
    if super_image.ndim != 5:
        # Standard 2D/4D SSIM
        return ssim(super_image, hr_gt, data_range=data_range, size_average=True)
        
    # Handling 5D (Batch, Channel, Depth, Height, Width)
    N, C, D, H, W = super_image.shape
    
    # Permute to (N, D, C, H, W) and reshape to stack Depth into the Batch dimension
    super_image_2d = super_image.permute(0, 2, 1, 3, 4).reshape(N * D, C, H, W)
    hr_gt_2d = hr_gt.permute(0, 2, 1, 3, 4).reshape(N * D, C, H, W)
    
    ssim_score = ssim(super_image_2d, hr_gt_2d, data_range=data_range, size_average=True)

    return ssim_score