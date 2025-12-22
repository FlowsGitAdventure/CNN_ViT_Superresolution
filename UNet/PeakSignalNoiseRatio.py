import torch
import torch.nn.functional as F

def calculate_psnr(super_image, hr_gt, data_range=1.0):
    # Ensure super_image matches hr_gt spatial dimensions
    if super_image.shape != hr_gt.shape:
        super_image = F.interpolate(super_image, size=hr_gt.shape[2:], mode='trilinear', align_corners=False)
        
    mse = torch.mean((super_image - hr_gt) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    psnr = 10. * torch.log10(data_range ** 2 / mse)

    return psnr
