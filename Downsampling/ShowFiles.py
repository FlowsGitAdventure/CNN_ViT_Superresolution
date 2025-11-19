import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

img = nib.load("C:\\Users\\floko\\Desktop\\Dev\\CNN_ViT_Superresolution\\studyforrest-data-multires7t\\sub-04\\ses-r08\\func\\sub-04_ses-r08_task-coverage_rec-dico_bold.nii.gz")
data = img.get_fdata()

plt.imshow(data[:, :, 5, 0], cmap='gray')
plt.show()

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)  # shape: (X, Y, Z, T) for 4D
    affine = img.affine
    zooms = img.header.get_zooms()  # voxel sizes (x,y,z,dt)
    return data, affine, zooms


import scipy.ndimage as ndi


def psf_blur_and_downsample(hr_vol, hr_voxel, lr_voxel, order=1, psf_sigma=None):
    # hr_vol: 3D or 4D (X,Y,Z[,T]) volume (assume spatial first dims)
    # hr_voxel/lr_voxel: tuples (sx,sy,sz)
    # psf_sigma: if None, compute from voxel ratio

    if hr_vol.ndim == 4:
        time_axis = True
        frames = hr_vol.shape[3]
    else:
        time_axis = False

    # compute sigma (voxels) approximating PSF to match LR voxel size
    if psf_sigma is None:
        # assume gaussian PSF ~ ratio of voxel sizes / 2.355 (FWHM->sigma) heuristic
        ratio = tuple(lr / hr for lr, hr in zip(lr_voxel, hr_voxel))
        # convert to HR voxel units; set sigma to half of ratio*something
        psf_sigma = tuple(max(0.5, r / 2.0) for r in ratio)

    def process_single(vol3):
        blurred = ndi.gaussian_filter(vol3, sigma=psf_sigma, mode='mirror')
        zoom_factors = tuple(hr / lr for hr, lr in zip(hr_voxel, lr_voxel))
        # downsample by inverse zoom (we want target voxel size lr_voxel)
        ds = ndi.zoom(blurred, zoom=(1 / zoom_factors[0], 1 / zoom_factors[1], 1 / zoom_factors[2]), order=order)
        return ds

    if time_axis:
        lr_list = [process_single(hr_vol[..., t]) for t in range(frames)]
        lr = np.stack(lr_list, axis=3)
    else:
        lr = process_single(hr_vol)
    return lr


def add_noise(vol, snr_db=20.0):
    # add gaussian noise with specified SNR (dB)
    signal_power = np.mean(vol ** 2)
    sigma = np.sqrt(signal_power / (10 ** (snr_db / 10.0)))
    noise = np.random.normal(scale=sigma, size=vol.shape).astype(vol.dtype)
    return vol + noise

def extract_patches_3dtime(hr_vol, lr_vol, spatial_patch, temporal_window, stride_spatial, stride_time):
    # hr_vol: (X_hr,Y_hr,Z_hr,T)
    # lr_vol: (X_lr,Y_lr,Z_lr,T)
    # assume coordinate alignment already handled
    X_hr,Y_hr,Z_hr,T = hr_vol.shape
    X_lr,Y_lr,Z_lr,_ = lr_vol.shape
    ph,pw,pd = spatial_patch  # in HR voxels
    tw = temporal_window
    patches = []
    # choose indices in HR space, compute corresponding LR indices by ratio
    ratio_x = X_hr / X_lr
    ratio_y = Y_hr / Y_lr
    ratio_z = Z_hr / Z_lr
    rx,ry,rz = ratio_x,ratio_y,ratio_z
    for t0 in range(0, T - tw + 1, stride_time):
        for x in range(0, X_hr - ph + 1, stride_spatial):
            for y in range(0, Y_hr - pw + 1, stride_spatial):
                for z in range(0, Z_hr - pd + 1, stride_spatial):
                    hr_patch = hr_vol[x:x+ph, y:y+pw, z:z+pd, t0:t0+tw]
                    # corresponding lr coordinates (floor div)
                    lx = int(x / rx)
                    ly = int(y / ry)
                    lz = int(z / rz)
                    lph = int(np.ceil(ph / rx))
                    lpw = int(np.ceil(pw / ry))
                    lpd = int(np.ceil(pd / rz))
                    lr_patch = lr_vol[lx:lx+lph, ly:ly+lpw, lz:lz+lpd, t0:t0+tw]
                    patches.append((lr_patch, hr_patch))
    return patches

import h5py

def save_patches_h5(pairs, out_path):
    # pairs: list of (lr_patch, hr_patch)
    with h5py.File(out_path, 'w') as f:
        N = len(pairs)
        # determine shapes dynamically
        lr0, hr0 = pairs[0]
        f.create_dataset('lr', shape=(N,)+lr0.shape, dtype='float32', compression='gzip')
        f.create_dataset('hr', shape=(N,)+hr0.shape, dtype='float32', compression='gzip')
        for i,(l,h) in enumerate(pairs):
            f['lr'][i] = l
            f['hr'][i] = h

import torch
from torch.utils.data import Dataset, DataLoader

class H5SRDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        import h5py
        self.f = h5py.File(h5_path, 'r')
        self.lr = self.f['lr']
        self.hr = self.f['hr']
        self.transform = transform
    def __len__(self):
        return self.lr.shape[0]
    def __getitem__(self, idx):
        lr = torch.from_numpy(self.lr[idx]).float()  # shape (T, X, Y, Z) or (X,Y,Z,T)
        hr = torch.from_numpy(self.hr[idx]).float()
        # reorder to network format: (C=1, T, D, H, W) for example:
        # assume input is (X,Y,Z,T)
        if lr.ndim == 4:
            lr = lr.permute(3,0,1,2).unsqueeze(0)  # (1,T,D,H,W)
            hr = hr.permute(3,0,1,2).unsqueeze(0)
        if self.transform:
            lr, hr = self.transform(lr, hr)
        return lr, hr
