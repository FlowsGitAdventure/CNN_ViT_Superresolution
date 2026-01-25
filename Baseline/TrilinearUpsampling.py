import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


def show_data():
    img = nib.load(BASELINE_OUTPUT)
    data = img.get_fdata()
    plt.imshow(data[:, :, 15, 4], cmap='gray')
    plt.savefig("comparison_result.png")
    plt.show()


def in_plane_upscale(input_path, output_path, scale_xy=2):
    """
    Baseline: Trilinear upsampling for X and Y dimensions only.
    Voxel size in Z remains constant.
    """
    # 1. Load LR NIfTI
    img = nib.load(input_path)
    data = img.get_fdata()  # Usually (X, Y, Z, T)
    affine = img.affine
    header = img.header

    # 2. Reshape for PyTorch: (B, C, D, H, W)
    # We map NIfTI Z to Torch D, NIfTI X to Torch H, NIfTI Y to Torch W
    # If 4D, T becomes Channels. If 3D, we add a channel dim.
    if data.ndim == 4:
        tensor = torch.from_numpy(data).float().permute(3, 2, 0, 1).unsqueeze(0)
    else:
        tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3)

    # 3. Calculate Target Size
    orig_d, orig_h, orig_w = tensor.shape[2:]
    target_size = (orig_d, int(orig_h * scale_xy), int(orig_w * scale_xy))

    print(f"Original Spatial Shape: {orig_h}x{orig_w} with {orig_d} slices")
    print(f"Target Spatial Shape:   {target_size[1]}x{target_size[2]} with {target_size[0]} slices")

    # 4. Trilinear Interpolation
    # Even though D is not changing, trilinear will look at
    # neighbors in H and W to fill the gaps.
    with torch.no_grad():
        upscaled = F.interpolate(
            tensor,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

    # 5. Back to Numpy (X, Y, Z, T)
    # Undo the permutation: (1, T, Z, X_new, Y_new) -> (X_new, Y_new, Z, T)
    output_data = upscaled.squeeze(0).permute(2, 3, 1, 0).numpy()
    if data.ndim == 3:
        output_data = output_data.squeeze(-1)

    # 6. CRITICAL: Update the Affine Matrix
    # We must divide the X and Y pixel spacing by the scale factor.
    # In NIfTI affines, [0,0] is X-spacing and [1,1] is Y-spacing.
    new_affine = affine.copy()
    new_affine[0, 0] /= scale_xy
    new_affine[1, 1] /= scale_xy

    # Optional: Adjust the origin (translation) to keep image centered
    # new_affine[:2, 3] -= (new_affine[:2, :2] @ np.array([0.5, 0.5])) * (scale_xy - 1)

    # 7. Save
    new_img = nib.Nifti1Image(output_data, new_affine, header)
    nib.save(new_img, output_path)
    print(f"Success! Saved baseline to {output_path}")
    show_data()


if __name__ == "__main__":
    LR_INPUT = '../Downsampling/hr_Anisotropic/1b80b162-ae30-42b1-9ad0-9677ee768f81_sub-04_ses-r08_task-orientation_rec-dico_run-08_bold_HR.nii.gz'
    BASELINE_OUTPUT = 'trilinear_baseline.nii.gz'

    in_plane_upscale(LR_INPUT, BASELINE_OUTPUT)
