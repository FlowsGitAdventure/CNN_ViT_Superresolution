# import os
# import nibabel as nib
# import torch
# from torch.utils.data import Dataset, DataLoader
#
# from GetRandomData import GetRandomData
#
#
# class CreateDataset(Dataset):
#     def __init__(self, lr_files, hr_files, lr_dir, hr_dir):
#         self.lr_dir = lr_dir
#         self.hr_dir = hr_dir
#         self.lr_files = lr_files
#         self.hr_files = hr_files
#         assert len(self.lr_files) == len(self.hr_files), "Error: HR and LR lists have different lengths!"
#
#     def __len__(self):
#         return len(self.lr_files)
#
#     def __getitem__(self, idx):
#         lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
#         hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
#
#         try:
#             lr_array = nib.load(lr_path)
#             lr_array = lr_array.get_fdata()
#
#             hr_array = nib.load(hr_path)
#             hr_array = hr_array.get_fdata()
#
#             # --- 3. Normalisierung & Tensor ---
#             lr_array = (lr_array - lr_array.min()) / (lr_array.max() - lr_array.min() + 1e-8)
#             hr_array = (hr_array - hr_array.min()) / (hr_array.max() - hr_array.min() + 1e-8)
#
#             # 1. Convert to Tensor (currently 4D: X, Y, Z, T)
#             lr_tensor = torch.from_numpy(lr_array).float()
#             hr_tensor = torch.from_numpy(hr_array).float()
#
#             # 2. Check dimensions and Permute
#             # We want (T, X, Y, Z) to fit (Batch, Channel, Depth, Height, Width)
#             if lr_tensor.ndim == 4:
#                 lr_tensor = lr_tensor.permute(3, 0, 1, 2)
#                 hr_tensor = hr_tensor.permute(3, 0, 1, 2)
#             elif lr_tensor.ndim == 3:
#                 # If the file is only 3D, add the channel dimension manually
#                 lr_tensor = lr_tensor.unsqueeze(0)
#                 hr_tensor = hr_tensor.unsqueeze(0)
#
#             return lr_tensor, hr_tensor
#
#         except Exception as e:
#             print(f"Error loading {lr_path}: {e}")
#             return self.__getitem__((idx + 1) % len(self.lr_files))


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
        p99 = np.percentile(data, 99.5)
        data = np.clip(data, 0, p99)

        # 2. Voxel-wise Temporal Z-Score: Highlights the BOLD signal
        # mean and std calculated along the T axis (axis 3)
        mean = np.mean(data, axis=3, keepdims=True)
        std = np.std(data, axis=3, keepdims=True) + 1e-8
        normalized = (data - mean) / std

        return normalized

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
            # Load temporal window
            lr_data = lr_img.slicer[..., t_start:t_end].get_fdata()
            hr_data = hr_img.slicer[..., t_start:t_end].get_fdata()

            lr_tensor = torch.from_numpy(self._normalize(lr_data)).float().permute(3, 0, 1, 2)
            hr_tensor = torch.from_numpy(self._normalize(hr_data)).float().permute(3, 0, 1, 2)
        else:
            lr_data = lr_img.get_fdata()
            hr_data = hr_img.get_fdata()
            lr_vol = torch.from_numpy(self._normalize(lr_data)).float().unsqueeze(0)
            hr_vol = torch.from_numpy(self._normalize(hr_data)).float().unsqueeze(0)

            lr_tensor = lr_vol.repeat(self.time_per_sample, 1, 1, 1)
            hr_tensor = hr_vol.repeat(self.time_per_sample, 1, 1, 1)


        lr_tensor = self._pad_to_multiple(lr_tensor)
        hr_tensor = self._pad_to_multiple(hr_tensor)

        # Augmentation
        if self.train and torch.rand(1) > 0.5:
            # Flip on the X-axis (dim 1 because dim 0 is Channels)
            lr_tensor = torch.flip(lr_tensor, dims=[1])
            hr_tensor = torch.flip(hr_tensor, dims=[1])

        return lr_tensor, hr_tensor


# if __name__ == "__main__":
#    lr_filepath = '../Downsampling/Low_Res_08_Rician'
#    hr_filepath = '../Downsampling/High_Res_08_Rician'
#    data_selector = GetRandomData(lr_filepath, hr_filepath, 16, 4, False)
#    train_files, validation_files, hr_train_files, hr_validation_files = data_selector.get_data()
#    train_dataset = CreateDataset(train_files, hr_train_files, lr_filepath, hr_filepath)



