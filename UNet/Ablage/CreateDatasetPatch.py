import os
import random
import math

import torch
from torch.utils.data import Dataset
import numpy as np


class RandomDataSelector:
    def __init__(self):
        self.n_train = 50
        self.n_validation = 10

        self.lr_filepath = "../data/Low_Res_Numpy_08"
        self.hr_filepath = "../data/High_Res_Numpy_08"

    def get_data(self):
        # list all data
        hr_files = os.listdir(self.hr_filepath)
        lr_files = os.listdir(self.lr_filepath)

        # Select random LR samples
        random.seed(24)
        random.shuffle(lr_files)
        train_files = lr_files[:self.n_train]
        validation_files = lr_files[self.n_train : self.n_train + self.n_validation]
        print(f"Training data: {len(train_files)}")
        print(f"Validation data: {len(validation_files)}")

        # Select corresponding HR samples
        hr_lookup = {os.path.basename(f).split('_')[0]: f for f in hr_files}
        hr_train_files = []

        for lr_path in train_files:
            lr_id = os.path.basename(lr_path).split('_')[0]
            if lr_id in hr_lookup:
                hr_train_files.append(hr_lookup[lr_id])
            else:
                print(f"Couldn't find HR file for '{lr_id}'")

        hr_validation_files = [
            hr_lookup[os.path.basename(f).split('_')[0]]
            for f in validation_files
            if os.path.basename(f).split('_')[0] in hr_lookup
        ]
        print(train_files[0])

        return train_files, validation_files, hr_train_files, hr_validation_files


class NumpyPatchDataset(Dataset):
    def __init__(self, lr_files, hr_files, lr_dir, hr_dir, expected_shape=(16, 16, 16)):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_files = lr_files
        self.hr_files = hr_files
        self.expected_shape = expected_shape
        assert len(self.lr_files) == len(self.hr_files), "Error: HR and LR lists have different lengths!"

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])

        try:
            lr_array = np.load(lr_path).astype(np.float32)
            hr_array = np.load(hr_path).astype(np.float32)

            # --- 3. Normalisierung & Tensor ---
            lr_array = (lr_array - lr_array.min()) / (lr_array.max() - lr_array.min() + 1e-8)
            hr_array = (hr_array - hr_array.min()) / (hr_array.max() - hr_array.min() + 1e-8)

            lr_tensor = torch.from_numpy(lr_array).unsqueeze(0)
            hr_tensor = torch.from_numpy(hr_array).unsqueeze(0)

            return lr_tensor, hr_tensor

        except Exception as e:
            print(f"Error loading {lr_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.lr_files))
