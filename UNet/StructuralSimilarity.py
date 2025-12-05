from pytorch_msssim import ssim


def calculate_ssim_score(super_image, hr_gt, data_range=1.0):
    if super_image.ndim != 5:
        return ssim(super_image, hr_gt, data_range=data_range, size_average=True, channel=super_image.shape[1])
    N, C, D, H, W = super_image.shape
    super_image_2d = super_image.permute(0, 2, 1, 3, 4).reshape(N * D, C, H, W)
    hr_gt_2d = hr_gt.permute(0, 2, 1, 3, 4).reshape(N * D, C, H, W)
    ssim_score = ssim(super_image_2d, hr_gt_2d, data_range=data_range, size_average=True, channel=C)

    return ssim_score
