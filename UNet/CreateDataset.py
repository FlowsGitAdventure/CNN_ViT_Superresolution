import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

from GetRandomData import GetRandomData


class CreateDataset(Dataset):
    def __init__(self, lr_files, hr_files, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_files = lr_files
        self.hr_files = hr_files
        assert len(self.lr_files) == len(self.hr_files), "Error: HR and LR lists have different lengths!"

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])

        try:
            lr_array = nib.load(lr_path)
            lr_array = lr_array.get_fdata()

            hr_array = nib.load(hr_path)
            hr_array = hr_array.get_fdata()

            # --- 3. Normalisierung & Tensor ---
            lr_array = (lr_array - lr_array.min()) / (lr_array.max() - lr_array.min() + 1e-8)
            hr_array = (hr_array - hr_array.min()) / (hr_array.max() - hr_array.min() + 1e-8)

            lr_tensor = torch.from_numpy(lr_array).unsqueeze(0).float()
            hr_tensor = torch.from_numpy(hr_array).unsqueeze(0).float()

            return lr_tensor, hr_tensor

        except Exception as e:
            print(f"Error loading {lr_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.lr_files))

