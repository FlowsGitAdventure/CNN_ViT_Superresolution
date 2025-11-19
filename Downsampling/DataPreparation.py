import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


class DataPreparation:
    def __init__(self):
        pass

    def load_nifti(path):
        img = nib.load(path)
        data = img.get_fdata(dtype=np.float32)  # shape: (X, Y, Z, T) for 4D
        affine = img.affine
        zooms = img.header.get_zooms()  # voxel sizes (x,y,z,dt)
        return data, affine, zooms

    def add_noise(vol, snr_db=20.0):
        # add gaussian noise with specified SNR (dB)
        signal_power = np.mean(vol ** 2)
        sigma = np.sqrt(signal_power / (10 ** (snr_db / 10.0)))
        noise = np.random.normal(scale=sigma, size=vol.shape).astype(vol.dtype)
        return vol + noise