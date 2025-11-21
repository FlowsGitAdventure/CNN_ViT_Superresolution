import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_img


# CONFIGURATION
HR_INPUT_PATH = "C:\\Users\\floko\\Desktop\\Dev\\CNN_ViT_Superresolution\\studyforrest-data-multires7t\\sub-04\\ses-r08\\func\\sub-04_ses-r08_task-coverage_rec-dico_bold.nii.gz"

# Output directory for prepared data
OUT_DIR = "./DownscaledData"
os.makedirs(OUT_DIR, exist_ok=True)

# Target voxel size for LR
LR_VOXEL_SIZE = (3, 3, 3)
DOWN_SCALE_FACTOR = 2.0

# Noise level (SNR in dB)
NOISE_FACTOR = 0.005


# 2. ADD NOISE
def add_noise(input, snr_db):
    data = input.get_fdata()
    sigma = np.max(data) * snr_db
    noise = np.random.normal(0, sigma, data.shape)
    noisy_data = data + noise
    noisy_data = np.clip(noisy_data, 0, None)
    noisy_img = nib.Nifti1Image(noisy_data, input.affine, input.header)
    return noisy_img


def main():
    # Step 1: Load image
    img_4d = nib.load(HR_INPUT_PATH)

    # Step 2: Resample
    print("Resampling low resolution ...")

    target_affine = img_4d.affine.copy()
    target_affine[:3, :3] *= DOWN_SCALE_FACTOR
    print(f"Original Voxelgröße (ca.): {np.linalg.norm(img_4d.affine[:3, 0]):.2f} mm")
    print(f"Neue Voxelgröße (ca.):     {np.linalg.norm(target_affine[:3, 0]):.2f} mm")
    new_affine = np.diag([2, 2, 2, 1])
    lr_data = resample_img(img_4d, target_affine=target_affine, interpolation='continuous')

    # Step 3: Add noise
    print("Adding noise...")
    lr_data = add_noise(lr_data, NOISE_FACTOR)

    # Step 4: Save HR and LR
    lr_out_path = os.path.join(OUT_DIR, "LR_generated.nii.gz")

    nib.save(lr_data, lr_out_path)

    print("Saved LR:", lr_out_path)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
