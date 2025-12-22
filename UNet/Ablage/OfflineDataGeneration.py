"""
In this script the HR and LR nifti-files are getting prepared for training.
The approach is "offline", meaning the files are getting prepared and saved in the filesystem as numpy-arrays.

---Start---
1. Load one pair of LR and HR files
2. Calculate the 99th percentile of each
3  Normalize the whole images with the calculated percentiles (for better training)
4. Select even patches from the data (e.g. 16x16x16) with a sliding window
5. Save each cube as numpy array on hard drive
6. Repeat
---Finish---

Those numpy arrays are later converted into tensors and are used for training
This approach saves RAM and every image is used completely instead of only choosing random patches per image.
"""
import nibabel as nib
import os
import numpy as np
import uuid
import math


PATCH_SIZE_LR = 16
SCALE_FACTOR = 2
PATCH_SIZE_HR = PATCH_SIZE_LR * SCALE_FACTOR
STRIDE = 10

HR_INPUT_PATH = "../../Downsampling/High_Res_08_Rician"
LR_INPUT_PATH = "../../Downsampling/Low_Res_08_Rician"
HR_NUMPY_OUTDIR = "./data/High_Res_Numpy_08"
LR_NUMPY_OUTDIR = "./data/Low_Res_Numpy_08"
os.makedirs(HR_NUMPY_OUTDIR, exist_ok=True)
os.makedirs(LR_NUMPY_OUTDIR, exist_ok=True)


class PrepareData:
    def __init__(self):
        pass

    def get_file_names(self):
        hr_dir_list = os.listdir(HR_INPUT_PATH)
        hr_dir_list = [f for f in hr_dir_list if '.nii' in f]

        lr_dir_list = os.listdir(LR_INPUT_PATH)
        lr_dir_list = [f for f in lr_dir_list if '.nii' in f]
        hr_dir_list.sort()
        lr_dir_list.sort()
        return hr_dir_list[1], lr_dir_list[1]   #ToDo: Switch to every file, thats just for testing

    def main(self):
        hr_img_filename, lr_img_filename = self.get_file_names()
        hr_filepath = os.path.join(HR_INPUT_PATH, hr_img_filename)
        lr_filepath = os.path.join(LR_INPUT_PATH, lr_img_filename)
        hr_img = nib.load(hr_filepath)
        lr_img = nib.load(lr_filepath)

        hr_data = hr_img.get_fdata()
        lr_data = lr_img.get_fdata()

        max = np.percentile(lr_data, 99.0)
        if max == 0: max = 1
        hr_data = np.clip(hr_data / max, 0, 1)
        lr_data = np.clip(lr_data / max, 0, 1)

        d, h, w, time = lr_data.shape
        a,b,c,q = hr_data.shape
        patch_count = 0

        for t in range(time):
            lr_vol = lr_data[..., t]
            hr_vol = hr_data[..., t]
            for z in range(0, d - PATCH_SIZE_LR + 1, STRIDE):
                for y in range(0, h - PATCH_SIZE_LR + 1, STRIDE):
                    for x in range(0, w - PATCH_SIZE_LR + 1, STRIDE):
                        lr_patch = lr_vol[z:z+PATCH_SIZE_LR, y:y+PATCH_SIZE_LR, x:x+PATCH_SIZE_LR]

                        # To not get patches which are mainly just black I filter them out
                        if np.mean(lr_patch) < 0.05:
                            continue

                        z_hr, y_hr, x_hr = z*SCALE_FACTOR, y*SCALE_FACTOR, x*SCALE_FACTOR
                        z_hr = math.ceil((z * SCALE_FACTOR) / SCALE_FACTOR) * SCALE_FACTOR
                        y_hr = math.ceil((y * SCALE_FACTOR) / SCALE_FACTOR) * SCALE_FACTOR
                        x_hr = math.ceil((x * SCALE_FACTOR) / SCALE_FACTOR) * SCALE_FACTOR

                        z_hr_max = z_hr+PATCH_SIZE_HR
                        y_hr_max = y_hr+PATCH_SIZE_HR
                        x_hr_max = x_hr+PATCH_SIZE_HR
                        hr_patch = hr_vol[z_hr:z_hr_max, y_hr:y_hr_max, x_hr:x_hr_max]

                        if hr_patch.size == 0:
                            print(f"Warn! empty array")

                        name_id = str(uuid.uuid4())
                        save_name = f"{name_id}_{patch_count:04d}"
                        np.save(os.path.join(HR_NUMPY_OUTDIR, f"{save_name}_HR.npy"), hr_patch)
                        np.save(os.path.join(LR_NUMPY_OUTDIR, f"{save_name}_LR.npy"), lr_patch)
#

                        patch_count += 1

        print(f"File {lr_img_filename}: Extracted {patch_count} patches.")


if __name__ == "__main__":
    prepare = PrepareData()
    prepare.main()
