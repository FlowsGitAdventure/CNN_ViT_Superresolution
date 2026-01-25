import os
import nibabel as nib
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class UnifiedSRDataset(Dataset):
    def __init__(self, lr_files, hr_files, lr_dir, hr_dir,  lr_max_shape, hr_max_shape, time_steps_per_sample=4, normalization="cnn_minmax", train=True, pad_multiple=16):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_max_shape = lr_max_shape
        self.hr_max_shape = hr_max_shape
        self.time_per_sample = time_steps_per_sample
        self.normalization = normalization
        self.train = train
        self.pad_multiple = pad_multiple
        self.samples = []

        stride = self.time_per_sample // 2

        for lr_f, hr_f in zip(lr_files, hr_files):
            lr_obj = nib.load(os.path.join(lr_dir, lr_f))
            shape = lr_obj.header.get_data_shape()

            if len(shape) == 4:
                for t in range(0, shape[3] - time_steps_per_sample + 1, stride):
                    self.samples.append((lr_f, hr_f, (t, t + time_steps_per_sample)))
            else:
                self.samples.append((lr_f, hr_f, None))

    # Normalization methods
    def _clip(self, data, lo=1.0, hi=99.0):
        lo_v = np.percentile(data, lo)
        hi_v = np.percentile(data, hi)
        return np.clip(data, lo_v, hi_v)

    def _minmax(self, data, eps=1e-8):
        return (data - data.min()) / (data.max() - data.min() + eps)

    def _temporal_zscore(self, data, eps=1e-8):
        mean = data.mean(axis=3, keepdims=True)
        std = data.std(axis=3, keepdims=True) + eps
        return (data - mean) / std

    def _percent_signal(self, data, eps=1e-8):
        mean = data.mean(axis=3, keepdims=True)
        return (data - mean) / (mean + eps)

    # Padding
    def _pad(self, tensor):
        _, d, h, w = tensor.shape
        pd = (self.pad_multiple - d % self.pad_multiple) % self.pad_multiple
        ph = (self.pad_multiple - h % self.pad_multiple) % self.pad_multiple
        pw = (self.pad_multiple - w % self.pad_multiple) % self.pad_multiple
        return F.pad(tensor, (0, pw, 0, ph, 0, pd))

    # Main getter
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lr_f, hr_f, tspan = self.samples[idx]

        # 1. Load NIfTI
        lr_img = nib.load(os.path.join(self.lr_dir, lr_f))
        hr_img = nib.load(os.path.join(self.hr_dir, hr_f))
        spacing = hr_img.header.get_zooms()[:3]

        # 2. Extract Data & Temporal Slicing
        if tspan is not None:
            t0, t1 = tspan
            lr = lr_img.slicer[..., t0:t1].get_fdata()
            hr = hr_img.slicer[..., t0:t1].get_fdata()
        else:
            # Add a singleton fourth dimension if it's a 3D volume
            lr = lr_img.get_fdata()[..., None]
            hr = hr_img.get_fdata()[..., None]

        # 3. Shared Clipping & Normalization
        lr = self._clip(lr)
        hr = self._clip(hr)

        lr = self._minmax(lr)
        hr = self._minmax(hr)

        assert np.all(np.isfinite(lr)), f"LR has non-finite values: min={lr.min()}, max={lr.max()}"
        assert np.all(np.isfinite(hr)), f"HR has non-finite values: min={hr.min()}, max={hr.max()}"

        # 4. Convert to Tensor (Shape: [C, D, H, W])
        def to_tensor(x):
            return torch.from_numpy(x).float().permute(3, 0, 1, 2)

        lr_tensor = to_tensor(lr)
        hr_tensor = to_tensor(hr)

        # 5. Padding to Global Max (The Fix)
        # Note: Tensor shape is [C, D, H, W], so dims 1, 2, 3
        pad_d = self.lr_max_shape[0] - lr_tensor.shape[1]
        pad_h = self.lr_max_shape[1] - lr_tensor.shape[2]
        pad_w = self.lr_max_shape[2] - lr_tensor.shape[3]

        # F.pad takes (w_left, w_right, h_top, h_bottom, d_front, d_back)
        lr_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant')

        # Repeat for HR
        pad_d_hr = self.hr_max_shape[0] - hr_tensor.shape[1]
        pad_h_hr = self.hr_max_shape[1] - hr_tensor.shape[2]
        pad_w_hr = self.hr_max_shape[2] - hr_tensor.shape[3]
        hr_padded = F.pad(hr_tensor, (0, pad_w_hr, 0, pad_h_hr, 0, pad_d_hr), mode='constant')

        # 6. Create a Binary Mask (Crucial for Fourier Loss)
        # This tells the loss function which pixels are real and which are reflection
        mask = torch.zeros_like(hr_padded)
        orig_d, orig_h, orig_w = hr_tensor.shape[1:]
        mask[:, :orig_d, :orig_h, :orig_w] = 1.0

        # 7. Augmentation (Apply to image AND mask if flipping)
        if self.train and torch.rand(1) > 0.5:
            lr_padded = torch.flip(lr_padded, dims=[2])  # Flip on Height
            hr_padded = torch.flip(hr_padded, dims=[2])
            mask = torch.flip(mask, dims=[2])

        spacing_tensor = torch.tensor(spacing, dtype=torch.float32)

        return lr_padded, hr_padded, mask, spacing_tensor





