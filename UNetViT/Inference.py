import torch
import nibabel as nib
import numpy as np
import os
import torch.nn.functional as F
from UNetHybridSuperResolution import UNet
import matplotlib.pyplot as plt
import cv2

# --- CONFIGURATION ---
# Scale factors must match your UNet upscale logic (D=2, H=2, W=1)
SCALE_D = 2
SCALE_H = 2
SCALE_W = 1
BASE_FILTERS = 16


def load_and_preprocess(data_chunk, multiple=16):
    """
    Normalizes data to [0, 1] and applies replicate padding.
    """
    # 1. Robust clipping
    p_low, p_high = np.percentile(data_chunk, [0.5, 99.5])
    data_chunk = np.clip(data_chunk, p_low, p_high)

    # 2. Capture stats for inverse normalization
    data_min, data_max = data_chunk.min(), data_chunk.max()
    eps = 1e-8

    # 3. Min-max normalize
    normalized_data = (data_chunk - data_min) / (data_max - data_min + eps)

    # 4. Prepare Tensor: (X, Y, Z, T) -> (B=1, C=T, D=X, H=Y, W=Z)
    tensor = torch.from_numpy(normalized_data).float().permute(3, 0, 1, 2).unsqueeze(0)

    # 5. Pad to multiple of 16 using replicate mode
    d, h, w = tensor.shape[2:]
    pd = (multiple - d % multiple) % multiple
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple

    tensor_padded = F.pad(tensor, (0, pw, 0, ph, 0, pd), mode="replicate")

    return tensor_padded, (pd, ph, pw), (data_min, data_max), normalized_data.shape[:3]


def show_result():
    print("--- Diagnostic: Loading Files for Visualization ---")
    # 1. Load the data
    img_lr = nib.load(LR_INPUT)
    img_sr = nib.load(OUTPUT)
    img_gt = nib.load(GT_FILE)

    data_lr = img_lr.get_fdata()
    data_sr = img_sr.get_fdata()
    data_gt = img_gt.get_fdata()

    # DIAGNOSTIC PRINT: If these shapes are the same, your LR_INPUT is not actually LR!
    print(f"LR Shape: {data_lr.shape}")
    print(f"SR Shape: {data_sr.shape}")
    print(f"GT Shape: {data_gt.shape}")

    # 2. Slice Selection (W-axis / Depth)
    # Ensure we are looking at the same anatomical slice
    slice_idx = data_lr.shape[2] // 2

    def prep_view(data, s_idx):
        # Extract 2D slice from first timepoint if 4D
        if data.ndim == 4:
            v = data[:, :, s_idx, 0]
        else:
            v = data[:, :, s_idx]
        return v

    raw_lr = prep_view(data_lr, slice_idx)
    raw_sr = prep_view(data_sr, slice_idx)
    raw_gt = prep_view(data_gt, slice_idx)

    # 3. Alignment Logic
    def align_to_sr(img, target_shape):
        # Resize to match SR pixel grid (this upsamples LR for comparison)
        if img.shape != target_shape:
            # Note: cv2.resize expects (Width, Height), which is (shape[1], shape[0])
            img = cv2.resize(img, (target_shape[1], target_shape[0]),
                             interpolation=cv2.INTER_NEAREST)  # Use NEAREST to see the pixels
        return np.rot90(img)

    view_sr = np.rot90(raw_sr)
    view_lr = align_to_sr(raw_lr, raw_sr.shape)
    view_gt = align_to_sr(raw_gt, raw_sr.shape)

    # 4. Calculate Error
    diff_map = np.abs(view_sr - view_gt)

    # 5. Plotting with a "Zoom" feature to see pixel differences
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    titles = [f"LR (Upsampled)\n{raw_lr.shape}", f"SR (Output)\n{raw_sr.shape}", "Ground Truth",
              "Difference (SR vs GT)"]
    views = [view_lr, view_sr, view_gt, diff_map]

    for i in range(4):
        if i < 3:
            # Use fixed percentiles to ensure fair contrast
            v_min, v_max = np.percentile(view_gt, [1, 99])
            axes[i].imshow(views[i], cmap='gray', vmin=v_min, vmax=v_max)
        else:
            axes[i].imshow(views[i], cmap='hot', vmin=0, vmax=np.percentile(diff_map, 98))

        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("comparison_result.png")
    print("Comparison saved as comparison_result.png")
    plt.show()


@torch.inference_mode()
def run_temporal_inference(model_path, input_path, output_path, device='cuda'):
    # 1. Load Checkpoint and Detect Channels/Filters
    checkpoint = torch.load(model_path, map_location=device)

    # Using the robust key detection we discussed
    first_key = next(k for k in checkpoint.keys() if 'weight' in k)
    trained_base_filters = checkpoint[first_key].shape[0]
    trained_in_channels = checkpoint[first_key].shape[1]

    # 2. Initialize Model
    model = UNet(in_channels=trained_in_channels,
                 out_channels=trained_in_channels,
                 base_filters=trained_base_filters).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. Load NIfTI
    img = nib.load(input_path)
    full_data = img.get_fdata()
    affine, header = img.affine, img.header
    total_timepoints = full_data.shape[3]

    all_reconstructed_chunks = []

    for t in range(0, total_timepoints - trained_in_channels + 1, trained_in_channels):
        chunk = full_data[..., t:t + trained_in_channels]

        # input_tensor is PADDED here (e.g., 208 -> 224)
        input_tensor, pads, stats, orig_lr_shape = load_and_preprocess(chunk)
        input_tensor = input_tensor.to(device)
        pd, ph, pw = pads  # Captured padding amounts

        # --- THE FIX ---
        # Calculate the shape the model EXPECTS based on the PADDED input
        # If input was 224, and scale is 2, the model output will be 448 internally.
        padded_d = input_tensor.shape[2] * SCALE_D
        padded_h = input_tensor.shape[3] * SCALE_H
        padded_w = input_tensor.shape[4] * SCALE_W

        # We pass the PADDED target shape so the model's .view() doesn't crash
        internal_target_shape = (1, trained_in_channels, padded_d, padded_h, padded_w)

        with torch.amp.autocast(device_type='cuda'):
            # Model processes the padded tensor and returns a padded output
            output = model(input_tensor, internal_target_shape)

        # 4. Inverse Normalization
        recon_chunk = output.cpu().numpy()[0]  # Shape: (C, D_pad, H_pad, W_pad)
        data_min, data_max = stats
        recon_chunk = recon_chunk * (data_max - data_min) + data_min

        # 5. CROP the output back to the true HR size
        # We remove the padding (multiplied by the scale factor)
        true_hr_d = orig_lr_shape[0] * SCALE_D
        true_hr_h = orig_lr_shape[1] * SCALE_H
        true_hr_w = orig_lr_shape[2] * SCALE_W

        recon_chunk = recon_chunk[:, :true_hr_d, :true_hr_h, :true_hr_w]

        # Reorder to (X, Y, Z, T) for NIfTI
        recon_chunk = recon_chunk.transpose(1, 2, 3, 0)
        all_reconstructed_chunks.append(recon_chunk)

    # 6. Reassemble and Save
    final_output = np.concatenate(all_reconstructed_chunks, axis=3)

    # Adjust affine for the NEW resolution
    new_affine = affine.copy()
    new_affine[0, 0] /= SCALE_D
    new_affine[1, 1] /= SCALE_H
    new_affine[2, 2] /= SCALE_W

    new_img = nib.Nifti1Image(final_output, new_affine, header)
    nib.save(new_img, output_path)
    print(f"Reconstruction complete: {output_path}")
    show_result()


if __name__ == "__main__":
    MODEL_PATH = './CheckpointHybrid/super-resolution-finale.pth'
    GT_FILE = '/fast_storage/flk7161/data/hr_Anisotropic/1b80b162-ae30-42b1-9ad0-9677ee768f81_sub-04_ses-r08_task-orientation_rec-dico_run-08_bold_HR.nii.gz'
    LR_INPUT = '/fast_storage/flk7161/data/lr_Anisotropic/1b80b162-ae30-42b1-9ad0-9677ee768f81_sub-04_ses-r08_task-orientation_rec-dico_run-08_bold_LR.nii.gz'
    OUTPUT = '/fast_storage/flk7161/inference/reconstructed_HR.nii.gz'

    if os.path.exists(MODEL_PATH):
        run_temporal_inference(MODEL_PATH, LR_INPUT, OUTPUT)
    else:
        print("Checkpoint not found!")
