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
            hr_obj = nib.load(hr_path)
            # if not np.allclose(lr_obj.affine, hr_obj.affine, atol=1e-5):
            #     print(f"{lr_obj.affine} {hr_obj.affine}")
            #     print(f"Warning: Affine mismatch in {lr_f}!")

            shape = lr_obj.header.get_data_shape()

            if len(shape) == 4:
                for t in range(0, shape[3] - self.time_per_sample + 1, stride):
                    self.samples.append({'lr': lr_f, 'hr': hr_f, 't': (t, t + self.time_per_sample)})
            else:
                self.samples.append({'lr': lr_f, 'hr': hr_f, 't': None})

    def _normalize_temporal_zscore(self, data):
        p99 = np.percentile(data, 99.5)
        data = np.clip(data, 0, p99)
        mean = np.mean(data, axis=3, keepdims=True)
        std = np.std(data, axis=3, keepdims=True) + 1e-8
        normalized = (data - mean) / std

        return normalized

    def _normalize_min_max_subject(self, data, lower_pct=1.0, upper_pct=99.5, eps=1e-8):

        # Robust clipping (computed over entire 4D volume)
        lo = np.percentile(data, lower_pct)
        hi = np.percentile(data, upper_pct)
        data = np.clip(data, lo, hi)
        # Per-subject min–max normalization
        min_val = data.min()
        max_val = data.max()
        normalized = (data - min_val) / (max_val - min_val + eps)
        norm_params = {
            "min": min_val,
            "max": max_val,
            "lo": lo,
            "hi": hi
        }

        return normalized, norm_params

    def _pad_to_multiple(self, tensor, multiple=16):
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

        if sample['t'] is not None:
            t_start, t_end = sample['t']
            lr_data = lr_img.slicer[..., t_start:t_end].get_fdata()
            hr_data = hr_img.slicer[..., t_start:t_end].get_fdata()

            # Permute (X, Y, Z, C) -> (C, X, Y, Z)
            lr_tensor = torch.from_numpy(self._normalize(lr_data)).float().permute(3, 0, 1, 2)
            hr_tensor = torch.from_numpy(self._normalize(hr_data)).float().permute(3, 0, 1, 2)
        else:
            # Handling 3D files
            lr_data = lr_img.get_fdata()
            hr_data = hr_img.get_fdata()
            lr_tensor = torch.from_numpy(self._normalize(lr_data)).float().unsqueeze(0)
            hr_tensor = torch.from_numpy(self._normalize(hr_data)).float().unsqueeze(0)

        lr_tensor = self._pad_to_multiple(lr_tensor)
        hr_tensor = self._pad_to_multiple(hr_tensor)

        # Augmentation
        if self.train and torch.rand(1) > 0.5:
            lr_tensor = torch.flip(lr_tensor, dims=[1])
            hr_tensor = torch.flip(hr_tensor, dims=[1])

        return lr_tensor, hr_tensor


import os
import nibabel as nib
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class UnifiedSRDataset(Dataset):
    def __init__(self, lr_files, hr_files, lr_dir, hr_dir, time_steps_per_sample=4, normalization="cnn_minmax", train=True, pad_multiple=16):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
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
    def _clip(self, data, lo=1.0, hi=99.5):
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

        lr_img = nib.load(os.path.join(self.lr_dir, lr_f))
        hr_img = nib.load(os.path.join(self.hr_dir, hr_f))

        if tspan is not None:
            t0, t1 = tspan
            lr = lr_img.slicer[..., t0:t1].get_fdata()
            hr = hr_img.slicer[..., t0:t1].get_fdata()
        else:
            lr = lr_img.get_fdata()[..., None]
            hr = hr_img.get_fdata()[..., None]

        # shared clipping
        lr = self._clip(lr)
        hr = self._clip(hr)

        # Normalization selection
        if self.normalization == "cnn_minmax":
            lr = self._minmax(lr)
            hr = self._minmax(hr)

        elif self.normalization == "temporal_zscore":
            lr = self._temporal_zscore(lr)
            hr = self._temporal_zscore(hr)

        elif self.normalization == "percent_signal":
            lr = self._percent_signal(lr)
            hr = self._percent_signal(hr)

        elif self.normalization == "dual_cnn_vit":
            lr_cnn = self._minmax(lr)
            hr_cnn = self._minmax(hr)
            lr_vit = self._temporal_zscore(lr)
            hr_vit = self._temporal_zscore(hr)
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        # Tensor conversion
        def to_tensor(x):
            return torch.from_numpy(x).float().permute(3, 0, 1, 2)

        if self.normalization == "dual_cnn_vit":
            lr_out = {
                "cnn": self._pad(to_tensor(lr_cnn)),
                "vit": self._pad(to_tensor(lr_vit)),
            }
            hr_out = {
                "cnn": self._pad(to_tensor(hr_cnn)),
                "vit": self._pad(to_tensor(hr_vit)),
            }
        else:
            lr_out = self._pad(to_tensor(lr))
            hr_out = self._pad(to_tensor(hr))

        # Augmentation
        if self.train and torch.rand(1) > 0.5:
            if isinstance(lr_out, dict):
                for k in lr_out:
                    lr_out[k] = torch.flip(lr_out[k], dims=[1])
                    hr_out[k] = torch.flip(hr_out[k], dims=[1])
            else:
                lr_out = torch.flip(lr_out, dims=[1])
                hr_out = torch.flip(hr_out, dims=[1])

        return lr_out, hr_out





