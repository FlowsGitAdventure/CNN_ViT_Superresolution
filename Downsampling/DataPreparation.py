import os
import numpy as np
import nibabel as nib
from nilearn.image import resample_img, smooth_img

'''To mitigate overfitting and class imbalance, data augmentation
techniques are employed, including random rotations (±15◦ for 2D and
equivalent in 3D), horizontal and vertical flips, scaling (±10 %)'''


# CONFIGURATION
#HR_INPUT_PATH = "C:\\Users\\floko\\Desktop\\Dev\\CNN_ViT_Superresolution\\studyforrest-data-multires7t\\sub-04\\ses-r08\\func\\sub-04_ses-r08_task-coverage_rec-dico_bold.nii.gz"
HR_INPUT_PATH = "C:\\Users\\floko\\Desktop\\Dev\\CNN_ViT_Superresolution\\studyforrest-data-multires7t\\sub-04\\ses-r08\\func"

# Output directory for prepared data
LR_OUT_DIR = "./Low_Res_08"
HR_OUT_DIR = "./High_Res_08"
os.makedirs(LR_OUT_DIR, exist_ok=True)
os.makedirs(HR_OUT_DIR, exist_ok=True)

# Target voxel size for LR
DOWN_SCALE_FACTOR = 4.0

# Noise level (SNR in dB)
NOISE_FACTOR = 0.05


def get_file_names():
    dir_list = os.listdir(HR_INPUT_PATH)
    dir_list = [f for f in dir_list if '.nii' in f]
    return dir_list



# 2. ADD NOISE
def add_noise(input):
    data = input.get_fdata()
    # sigma = np.max(data) * NOISE_FACTOR
    sigma = np.percentile(data, 99) * NOISE_FACTOR
    noise = np.random.normal(0, sigma, data.shape)
    noisy_data = data + noise
    noisy_data = np.clip(noisy_data, 0, None)
    noisy_img = nib.Nifti1Image(noisy_data, input.affine, input.header)
    return noisy_img


def main():
    files = get_file_names()
    for file in files:
        # Load image
        filename = file.split('.')[0]
        filepath = os.path.join(HR_INPUT_PATH, file)
        print(f"Processing {filename}...")
        img_4d = nib.load(filepath)

        # NORMALIZE HR
        # Normalize HR first so your noise/blur is consistent across subjects.
        # img_4d = normalize_data(img_4d)

        # Smoothing the image
        # Before downsampling, we smooth to mimic the acquisition of larger voxels.
        # FWHM is usually set to the size of the NEW voxel or slightly larger.
        # If current voxel is 1mm and we downscale by 4, new voxel is 4mm.
        # We apply a smoothing of roughly 4mm (or slightly less, e.g., 3mm)
        # to simulate the partial volume effect.
        # http://jpeelle.net/mri/image_processing/smoothing.html
        current_voxel_size = np.linalg.norm(img_4d.affine[:3, 0])
        fwhm_mm = current_voxel_size * DOWN_SCALE_FACTOR

        print(f"  Smoothing with FWHM={fwhm_mm:.2f}mm before downsampling...")
        # smooth_img handles 4D data automatically
        img_smoothed = smooth_img(img_4d, fwhm=fwhm_mm)

        # Step 2: Resample
        print("Resampling low resolution ...")

        target_affine = img_4d.affine.copy()
        target_affine[:3, :3] *= DOWN_SCALE_FACTOR
        print(f"Original Voxelgröße (ca.): {np.linalg.norm(img_4d.affine[:3, 0]):.2f} mm")
        print(f"Neue Voxelgröße (ca.):     {np.linalg.norm(target_affine[:3, 0]):.2f} mm")
        # lr_data = resample_img(img_4d, target_affine=target_affine, interpolation='continuous') rough image
        lr_data = resample_img(img_smoothed, target_affine=target_affine, interpolation='continuous')

        # Step 3: Add noise
        print("Adding noise...")
        lr_data = add_noise(lr_data)

        # Step 4: Save HR and LR
        lr_out_path = os.path.join(LR_OUT_DIR, f"{filename}_LR_generated.nii.gz")
        hr_out_path = os.path.join(HR_OUT_DIR, f"{filename}_HR.nii.gz")

        nib.save(lr_data, lr_out_path)
        nib.save(img_4d, hr_out_path)

        print("Saved LR:", lr_out_path)

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
