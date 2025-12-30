import torch
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os


def verify_alignment(lr_path, hr_path):
    # 1. Load the NIfTI files
    lr_img = nib.load(lr_path)
    hr_img = nib.load(hr_path)

    lr_data = lr_img.get_fdata()
    hr_data = hr_img.get_fdata()

    # 2. Select a central slice (T=0, Z=Middle)
    t_idx = 0
    lr_z = lr_data.shape[2] // 2
    hr_z = hr_data.shape[2] // 2

    lr_slice = lr_data[:, :, lr_z, t_idx]
    hr_slice = hr_data[:, :, hr_z, t_idx]

    # 3. Print Intensity Stats
    print(f"--- Statistics ---")
    print(f"LR Range: {lr_data.min():.4f} to {lr_data.max():.4f} | Mean: {lr_data.mean():.4f}")
    print(f"HR Range: {hr_data.min():.4f} to {hr_data.max():.4f} | Mean: {hr_data.mean():.4f}")

    # 4. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot LR
    axes[0].imshow(np.rot90(lr_slice), cmap='gray')
    axes[0].set_title(f"Low-Res (Input) \n {lr_data.shape[:3]}")
    axes[0].axis('off')

    # Plot HR
    axes[1].imshow(np.rot90(hr_slice), cmap='gray')
    axes[1].set_title(f"High-Res (Target) \n {hr_data.shape[:3]}")
    axes[1].axis('off')

    # Plot Overlay/Difference (Requires resizing LR to HR size)
    # This checks if the brain 'sits' in the same spot
    from scipy.ndimage import zoom
    scale = hr_data.shape[0] / lr_data.shape[0]
    lr_resized = zoom(lr_slice, scale, order=1)

    # Normalize for visual comparison
    lr_norm = (lr_resized - lr_resized.min()) / (lr_resized.max() - lr_resized.min() + 1e-8)
    hr_norm = (hr_slice - hr_slice.min()) / (hr_slice.max() - hr_slice.min() + 1e-8)

    diff = np.abs(hr_norm - lr_norm)
    axes[2].imshow(np.rot90(diff), cmap='hot')
    axes[2].set_title("Difference Map (Alignment Check)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# Replace with actual file paths from your NEW directories
lr_sample = "../Downsampling/Low_Res_08_NEW/7892cf58-1f5f-45b2-b4e9-cb4e8253f354_sub-04_ses-r08_task-coverage_bold_LR.nii.gz"
hr_sample = "../Downsampling/High_Res_08_NEW/7892cf58-1f5f-45b2-b4e9-cb4e8253f354_sub-04_ses-r08_task-coverage_bold_HR.nii.gz"
verify_alignment(lr_sample, hr_sample)
