import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import glob
import os


class FMRISuperResDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, train=True, patch_size=(32, 32, 32)):
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.nii*")))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.nii*")))
        self.train = train
        self.patch_size = np.array(patch_size)
        self.scale_factor = 4

        if len(self.lr_paths) != len(self.hr_paths):
            print(f"Warning: Found {len(self.lr_paths)} LR and {len(self.hr_paths)} HR files. Checks naming.")

        self.index_map = []
        for i, lr_path in enumerate(self.lr_paths):
            try:
                header = nib.load(lr_path).header
                n_volumes = header.get_data_shape()[3]
                for t in range(n_volumes):
                    self.index_map.append((i, t))
            except:
                pass

        print(f"Dataset ({'Train' if train else 'Val'}) ready. {len(self.index_map)} volumes.")

    def __len__(self):
        return len(self.index_map)

    def pad_to_minimum(self, data, min_shape):
        """Ensures volume is at least as big as min_shape."""
        pad_dims = []
        for i in range(3):
            diff = min_shape[i] - data.shape[i]
            if diff > 0:
                pad_dims.append((0, diff))
            else:
                pad_dims.append((0, 0))

        if any(p[1] > 0 for p in pad_dims):
            return np.pad(data, pad_dims, mode='constant')
        return data

    def get_safe_random_crop(self, lr_vol, hr_vol):
        """
        Calculates a random crop that fits inside BOTH the LR and HR volumes.
        This handles cases where HR is not perfectly 4x larger than LR.
        """
        # 1. Enforce Minimum Size (Patch Size)
        # We must pad images if they are smaller than the patch itself
        lr_vol = self.pad_to_minimum(lr_vol, self.patch_size)
        hr_vol = self.pad_to_minimum(hr_vol, self.patch_size * 4)

        # 2. Calculate Valid Start Indices
        # We need a start index 's' (in LR space) such that:
        #   s + patch_size  <= LR_Dimension
        #   s*4 + patch_size*4 <= HR_Dimension

        start_indices = []
        for i in range(3):
            # Max index allowed by LR volume
            max_lr = lr_vol.shape[i] - self.patch_size[i]

            # Max index allowed by HR volume (converted to LR scale)
            # Logic: (HR_Dim - HR_Patch) / 4
            max_hr = (hr_vol.shape[i] - (self.patch_size[i] * 4)) // 4

            # The valid crop must satisfy BOTH
            limit = min(max_lr, max_hr)

            if limit <= 0:
                start_indices.append(0)
            else:
                start_indices.append(np.random.randint(0, limit + 1))

        sx, sy, sz = start_indices

        # 3. Perform Crop
        # LR Crop
        lr_patch = lr_vol[sx:sx + self.patch_size[0],
                   sy:sy + self.patch_size[1],
                   sz:sz + self.patch_size[2]]

        # HR Crop (aligned coordinates)
        hsx, hsy, hsz = sx * 4, sy * 4, sz * 4
        htx, hty, htz = self.patch_size * 4

        hr_patch = hr_vol[hsx:hsx + htx,
                   hsy:hsy + hty,
                   hsz:hsz + htz]

        return lr_patch, hr_patch

    def __getitem__(self, idx):
        file_idx, time_idx = self.index_map[idx]

        # Load
        lr_obj = nib.load(self.lr_paths[file_idx])
        hr_obj = nib.load(self.hr_paths[file_idx])

        lr_vol = np.array(lr_obj.dataobj[..., time_idx], dtype=np.float32)
        hr_vol = np.array(hr_obj.dataobj[..., time_idx], dtype=np.float32)

        # Normalize
        if lr_vol.max() > 0: lr_vol /= lr_vol.max()
        if hr_vol.max() > 0: hr_vol /= hr_vol.max()

        if self.train:
            lr_final, hr_final = self.get_safe_random_crop(lr_vol, hr_vol)
        else:
            # For validation, we essentially do a center crop or just return the safe crop
            # to keep it simple and crash-free for now
            lr_final, hr_final = self.get_safe_random_crop(lr_vol, hr_vol)

        return (torch.from_numpy(lr_final).unsqueeze(0),
                torch.from_numpy(hr_final).unsqueeze(0))
