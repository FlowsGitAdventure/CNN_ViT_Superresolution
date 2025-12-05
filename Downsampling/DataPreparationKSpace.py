import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# CONFIGURATION
HR_INPUT_PATH = "C:\\Users\\floko\\Desktop\\Dev\\CNN_ViT_Superresolution\\studyforrest-data-multires7t\\sub-04\\ses-r08\\func"
LR_OUT_DIR = "./Low_Res_08_Rician"
HR_OUT_DIR = "./High_Res_08_Rician"
os.makedirs(LR_OUT_DIR, exist_ok=True)
os.makedirs(HR_OUT_DIR, exist_ok=True)

DOWN_SCALE_FACTOR = 2.0
MIN_NOISE = 0.005
MAX_NOISE = 0.02


def get_file_names():
    dir_list = os.listdir(HR_INPUT_PATH)
    return [f for f in dir_list if '.nii' in f]


def downsample_with_rician_noise(img_obj, scale_factor, noise_factor):
    """
    Downsamples a 4D fMRI file by iterating over the Time dimension to reduce RAM usage.

    1. Converts to K-Space
    2. Crops High Frequencies (Downsample)
    3. Inverse FFT to Complex Image Space
    4. Adds Complex Gaussian Noise (Real + Imag)
    5. Takes Magnitude -> Result is Rician Noise
    """
    data = img_obj.get_fdata()
    original_affine = img_obj.affine

    # Get Shapes
    x, y, z, t = data.shape
    new_x, new_y, new_z = int(x // scale_factor), int(y // scale_factor), int(z // scale_factor)

    # Pre-allocate the output array (Use float32 to save space)
    # create an empty container for the final Low-Res 4D image
    lr_data_4d = np.zeros((new_x, new_y, new_z, t), dtype=np.float32)

    # 3. Calculate Crop Indices (FIXED FOR ODD NUMBERS)
    cx, cy, cz = x // 2, y // 2, z // 2

    # Calculate start points
    start_x = cx - new_x // 2
    start_y = cy - new_y // 2
    start_z = cz - new_z // 2

    # Define slices: Start -> Start + Length
    slice_x = slice(start_x, start_x + new_x)
    slice_y = slice(start_y, start_y + new_y)
    slice_z = slice(start_z, start_z + new_z)

    print(f"  Processing {t} timeframes (Target shape: {new_x, new_y, new_z}) iteratively to save RAM...")

    for i in range(t):
        # Extract single 3D volume
        vol_3d = data[..., i]

        # --- A. FFT to K-Space (3D) ---
        k_space = np.fft.fftn(vol_3d)
        k_space = np.fft.fftshift(k_space)

        # --- B. Crop (Downsample) ---
        k_space_cropped = k_space[slice_x, slice_y, slice_z]

        # --- C. iFFT to Complex Image Space ---
        k_space_cropped = np.fft.ifftshift(k_space_cropped)
        img_complex = np.fft.ifftn(k_space_cropped)

        # --- D. Generate Complex Noise ---
        # Calculate sigma for this specific volume
        signal_mag = np.abs(img_complex)

        # Optimization: To save CPU, you can skip p99 calculation every frame
        # and just use std, or calculate p99 once globally.
        # For now, we do it per frame for accuracy.
        # ToDo: check if noise for every 3D slice is different and if this is bad

        nonzero_signal = signal_mag[signal_mag > 0]
        if len(nonzero_signal) > 0:
            p99 = np.percentile(nonzero_signal, 99)
        else:
            p99 = 1.0
        sigma = p99 * noise_factor

        # Generate noise
        noise_real = np.random.normal(0, sigma, img_complex.shape)
        noise_imag = np.random.normal(0, sigma, img_complex.shape)

        # Add noise
        img_complex_noisy = (img_complex.real + noise_real) + 1j * (img_complex.imag + noise_imag)

        # --- E. Magnitude (Rician) ---
        vol_lr = np.abs(img_complex_noisy)

        # --- F. Store in 4D container ---
        lr_data_4d[..., i] = vol_lr.astype(np.float32)

    # Create final NIfTI
    target_affine = original_affine.copy()
    target_affine[:3, :3] *= scale_factor

    return nib.Nifti1Image(lr_data_4d, target_affine, img_obj.header)


def main():
    files = get_file_names()
    for file in files:
        filename = file.split('.')[0]
        filepath = os.path.join(HR_INPUT_PATH, file)

        print(f"Processing {filename}...")
        img_4d = nib.load(filepath)
        current_noise_level = np.random.uniform(MIN_NOISE, MAX_NOISE)
        lr_img = downsample_with_rician_noise(img_4d, scale_factor=DOWN_SCALE_FACTOR, noise_factor=current_noise_level)

        lr_out_path = os.path.join(LR_OUT_DIR, f"{filename}_LR_Rician.nii.gz")
        hr_out_path = os.path.join(HR_OUT_DIR, f"{filename}_HR.nii.gz")

        # Save HR image as float32 to save space and make more precise than integer
        hr_data = img_4d.get_fdata().astype(np.float32)
        hr_img_final = nib.Nifti1Image(hr_data, img_4d.affine, img_4d.header)

        # Save
        hr_img_final.header.set_data_dtype(np.float32)
        lr_img.header.set_data_dtype(np.float32)
        nib.save(lr_img, lr_out_path)
        nib.save(img_4d, hr_out_path)
        print(f"  Saved pairs to {LR_OUT_DIR} and {HR_OUT_DIR}")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()