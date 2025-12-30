import torch.nn.functional as F
from pytorch_msssim import ssim


def calculate_ssim_score(super_image, hr_gt, data_range):
    if super_image.shape[-3:] != hr_gt.shape[-3:]:
        super_image = F.interpolate(
            super_image,
            size=hr_gt.shape[-3:],
            mode='trilinear',
            align_corners=False
        )

    # 2. Universal Squeeze
    if super_image.ndim == 6:
        super_image = super_image.squeeze(2)
    if hr_gt.ndim == 6: hr_gt = \
        hr_gt.squeeze(2)

    # Check for 5D and squeeze C if it's there
    if super_image.ndim == 5 and super_image.shape[1] == 1:
        super_image = super_image.squeeze(1)
    if hr_gt.ndim == 5 and hr_gt.shape[1] == 1:
        hr_gt = hr_gt.squeeze(1)
        
    # Handling 5D
    h, w = hr_gt.shape[-2:]
    total_slices = hr_gt.numel() // (h * w)
    super_image_2d = super_image.reshape(total_slices, 1, h, w)
    hr_gt_2d = hr_gt.reshape(total_slices, 1, h, w)
    
    ssim_score = ssim(super_image_2d, hr_gt_2d, data_range=data_range, size_average=True)
    return ssim_score
