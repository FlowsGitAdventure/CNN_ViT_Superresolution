import os
import numpy as np
import nibabel as nib
import uuid
import time
from tqdm import tqdm

# CONFIGURATION
HR_INPUT_PATH = "C:\\Users\\floko\\Desktop\\Dev\\CNN_ViT_Superresolution\\studyforrest-data-multires7t\\data"
LR_OUT_DIR = "./Low_Res_08_NEW"
HR_OUT_DIR = "./High_Res_08_NEW"
os.makedirs(LR_OUT_DIR, exist_ok=True)
os.makedirs(HR_OUT_DIR, exist_ok=True)

DOWN_SCALE_FACTOR = 4.0
MIN_NOISE = 0.001
MAX_NOISE = 0.008


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

    # 3. Calculate Crop Indices (Robust for any scale factor)
    # Center of the original volume
    cx, cy, cz = x // 2, y // 2, z // 2

    # Calculate half-lengths of the new volume
    hx, hy, hz = new_x // 2, new_y // 2, new_z // 2

    # Calculate start and end points to ensure it's symmetric
    start_x, end_x = cx - hx, cx - hx + new_x
    start_y, end_y = cy - hy, cy - hy + new_y
    start_z, end_z = cz - hz, cz - hz + new_z

    slice_x = slice(start_x, end_x)
    slice_y = slice(start_y, end_y)
    slice_z = slice(start_z, end_z)

    # print(f"  Processing {t} timeframes (Target shape: {new_x, new_y, new_z}) iteratively to save RAM...")

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
        signal_mag = np.abs(img_complex)

        # We use the mean signal of the brain tissue (ignoring zero background)
        brain_mask = signal_mag > (np.max(signal_mag) * 0.1)
        if np.any(brain_mask):
            ref_signal = np.mean(signal_mag[brain_mask])
        else:
            ref_signal = np.mean(signal_mag)

        sigma = ref_signal * noise_factor

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
    total_start_time = time.time()

    for file in tqdm(files, desc="Processing Files", unit="file"):
        filename = file.split('.')[0]
        filepath = os.path.join(HR_INPUT_PATH, file)

        # print(f"Processing {filename}...")
        img_4d = nib.load(filepath)
        current_noise_level = np.random.uniform(MIN_NOISE, MAX_NOISE)
        lr_img = downsample_with_rician_noise(img_4d, scale_factor=DOWN_SCALE_FACTOR, noise_factor=current_noise_level)

        name_id = str(uuid.uuid4())
        filename = f'{name_id}_{filename}'
        lr_out_path = os.path.join(LR_OUT_DIR, f"{filename}_LR.nii.gz")
        hr_out_path = os.path.join(HR_OUT_DIR, f"{filename}_HR.nii.gz")

        # Save HR image as float32 to save space and make more precise than integer
        hr_data = img_4d.get_fdata().astype(np.float32)
        hr_img_final = nib.Nifti1Image(hr_data, img_4d.affine, img_4d.header)

        # Save
        hr_img_final.header.set_data_dtype(np.float32)
        lr_img.header.set_data_dtype(np.float32)
        nib.save(lr_img, lr_out_path)
        nib.save(img_4d, hr_out_path)
        # print(f"Original Timepoints: {img_4d.shape[3]} | Downsampled Timepoints: {lr_img.shape[3]}")
        # print(f"  Saved pairs to {LR_OUT_DIR} and {HR_OUT_DIR}")

    total_end_time = time.time()
    duration = total_end_time - total_start_time

    print("\n" + "=" * 30)
    print("Data preparation complete.")
    print(f"Total files processed: {len(files)}")
    print(f"Total time taken: {duration / 60:.2f} minutes")
    print(f"Average time per file: {duration / len(files):.2f} seconds")
    print("=" * 30)


if __name__ == "__main__":
    main()