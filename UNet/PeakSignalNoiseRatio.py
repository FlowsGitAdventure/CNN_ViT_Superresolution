import torch


def calculate_psnr(super_image, hr_gt, data_range=1.0):
    mse = torch.mean((super_image - hr_gt) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    psnr = 10. * torch.log10(data_range ** 2 / mse)

    return psnr
