import torch
import nibabel as nib
import numpy as np
import os
import math
import warnings
from UNetSuperRes import UNet

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def predict_volume_patched(model, vol, device, patch_size=(32, 32, 32)):
    """
    Runs inference on a 3D volume using patch-based processing.
    """
    D, H, W = vol.shape
    scale = 4
    target_D, target_H, target_W = D * scale, H * scale, W * scale

    # Pre-allocate output volume in RAM (This is just one 3D volume, so it fits fine)
    output_vol = np.zeros((target_D, target_H, target_W), dtype=np.float32)

    pd, ph, pw = patch_size

    for d in range(0, D, pd):
        for h in range(0, H, ph):
            for w in range(0, W, pw):
                d_end = min(d + pd, D)
                h_end = min(h + ph, H)
                w_end = min(w + pw, W)

                chunk = vol[d:d_end, h:h_end, w:w_end]

                cd, ch, cw = chunk.shape
                pad_d = pd - cd
                pad_h = ph - ch
                pad_w = pw - cw

                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    chunk = np.pad(chunk, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')

                input_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    out_tensor = model(input_tensor)

                out_chunk = out_tensor.squeeze().cpu().numpy()

                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    valid_d = cd * scale
                    valid_h = ch * scale
                    valid_w = cw * scale
                    out_chunk = out_chunk[:valid_d, :valid_h, :valid_w]

                out_d, out_h, out_w = d * scale, h * scale, w * scale

                output_vol[out_d:out_d + out_chunk.shape[0],
                out_h:out_h + out_chunk.shape[1],
                out_w:out_w + out_chunk.shape[2]] = out_chunk

                del input_tensor, out_tensor

    return output_vol


def predict_single_file(lr_path, model_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference Device: {device}")

    # 1. Load Model
    try:
        model = UNet(in_channels=1, out_channels=1, base_filters=32).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Standard load failed ({e}), trying fallback...")
        model = UNet(in_channels=1, out_channels=1, base_filters=64).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    # 2. Setup Files
    print(f"Loading NIfTI: {lr_path}")
    img_obj = nib.load(lr_path)

    # We load data carefully. If it's huge, we might need to verify memory,
    # but LR data is usually small enough.
    lr_data = img_obj.get_fdata().astype(np.float32)


    # Normalize
    global_max = lr_data.max()
    if global_max > 0: lr_data /= global_max

    # Detect Dimensions
    if len(lr_data.shape) == 4:
        H, W, D, T = lr_data.shape
    else:
        H, W, D = lr_data.shape
        T = 1

    # 3. Setup Disk-Based Array (Memmap)
    # We create a temporary file to hold the HR data on DISK instead of RAM
    scale = 4
    target_shape = (H * scale, W * scale, D * scale, T)

    temp_mmap_path = os.path.join(os.path.dirname(output_path), "temp_hr_data.dat")
    if os.path.exists(temp_mmap_path): os.remove(temp_mmap_path)

    print(f"Allocating disk buffer: {temp_mmap_path} {target_shape}")
    # Create a memory-mapped array. This acts like a numpy array but lives on the hard drive.
    final_data = np.memmap(temp_mmap_path, dtype=np.float32, mode='w+', shape=target_shape)

    print(f"Processing {T} volumes...")

    for t in range(T):
        print(f"Reconstructing Volume {t + 1}/{T}...", end='\r')

        vol = lr_data[..., t] if T > 1 else lr_data

        # Run inference
        hr_vol = predict_volume_patched(model, vol, device, patch_size=(32, 32, 32))

        # Write directly to disk buffer
        if T > 1:
            final_data[..., t] = hr_vol
        else:
            final_data[:] = hr_vol

        # Optional: Flush to disk periodically
        if t % 10 == 0:
            final_data.flush()

    print("\nInference complete. Saving NIfTI header...")

    # 4. Save Final File
    # Important: We force .nii (uncompressed) to avoid GZIP memory explosion
    if output_path.endswith(".gz"):
        output_path = output_path.replace(".gz", "")
        print(f"NOTE: Saving as uncompressed .nii to save RAM: {output_path}")

    # Adjust Affine
    new_affine = img_obj.affine.copy()
    new_affine[:3, :3] /= float(scale)

    # Create NIfTI image referencing the memmap
    new_img = nib.Nifti1Image(final_data, new_affine)
    nib.save(new_img, output_path)

    # Cleanup
    print("Cleaning up temporary files...")
    del final_data  # Close the pointer
    del new_img

    # Delete the temp .dat file?
    # Usually we want to delete it, but since nib.save reads from it,
    # we must ensure save is done.
    # Since we passed the memmap to Nifti1Image, nib.save reads from it.
    # Now it is safe to remove.
    try:
        os.remove(temp_mmap_path)
    except PermissionError:
        print(f"Warning: Could not delete temp file {temp_mmap_path}. Please delete manually.")

    print(f"Done! Saved: {output_path}")


# Example Usage
if __name__ == "__main__":
    predict_single_file(
        lr_path="../data/Low_Res_08/nii_LR_generated.nii.gz",
        model_path="../checkpoints/unet_final.pth",
        output_path="../../Downsampling/output_hr.nii.gz"
    )