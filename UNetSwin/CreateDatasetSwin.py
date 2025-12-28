import os
import nibabel as nib
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class CreateDataset(Dataset):
    def __init__(self, lr_files, hr_files, lr_dir, hr_dir, time_steps_per_sample=4, train=True):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.time_per_sample = time_steps_per_sample
        self.train = train
        self.samples = []

        stride = self.time_per_sample // 2

        for lr_f, hr_f in zip(lr_files, hr_files):
            lr_path = os.path.join(self.lr_dir, lr_f)
            hr_path = os.path.join(self.hr_dir, hr_f)

            lr_obj = nib.load(lr_path)
            # hr_obj = nib.load(hr_path)
            # if not np.allclose(lr_obj.affine, hr_obj.affine, atol=1e-5):
            #     print(f"{lr_obj.affine} {hr_obj.affine}")
            #     print(f"Warning: Affine mismatch in {lr_f}!")

            shape = lr_obj.header.get_data_shape()

            if len(shape) == 4:
                for t in range(0, shape[3] - self.time_per_sample + 1, stride):
                    self.samples.append({'lr': lr_f, 'hr': hr_f, 't': (t, t + self.time_per_sample)})
            else:
                self.samples.append({'lr': lr_f, 'hr': hr_f, 't': None})

    def _normalize(self, data):
        # 1. Clip Outliers
        p99 = np.percentile(data, 99.5)
        data = np.clip(data, 0, p99)

        # Voxel-wise Temporal Z-Score
        mean = np.mean(data, axis=3, keepdims=True)
        std = np.std(data, axis=3, keepdims=True) + 1e-8
        normalized = (data - mean) / std

        return normalized

    def _pad_to_multiple(self, tensor, multiple=16):
        # tensor shape: (T, D, H, W)
        d, h, w = tensor.shape[1:]
        pad_d = (multiple - d % multiple) % multiple
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        return F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        lr_img = nib.load(os.path.join(self.lr_dir, sample['lr']))
        hr_img = nib.load(os.path.join(self.hr_dir, sample['hr']))

        t_start, t_end = sample['t']
        lr_data = lr_img.slicer[..., t_start:t_end].get_fdata()
        hr_data = hr_img.slicer[..., t_start:t_end].get_fdata()

        # Normalize and permute to (T, X, Y, Z)
        lr_tensor = torch.from_numpy(self._normalize(lr_data)).float().permute(3, 0, 1, 2)
        hr_tensor = torch.from_numpy(self._normalize(hr_data)).float().permute(3, 0, 1, 2)

        # Pad spatial dims to be divisible by Swin patch size hierarchy (16)
        lr_tensor = self._pad_to_multiple(lr_tensor, multiple=16)
        hr_tensor = self._pad_to_multiple(hr_tensor, multiple=16)

        # apply identical augmentation to both LR and HR.
        if self.train and torch.rand(1) > 0.5:
            # Flip on X, Y, or Z axis (dims 1, 2, or 3)
            axis = torch.randint(1, 4, (1,)).item()
            lr_tensor = torch.flip(lr_tensor, dims=[axis])
            hr_tensor = torch.flip(hr_tensor, dims=[axis])

        return lr_tensor, hr_tensor
