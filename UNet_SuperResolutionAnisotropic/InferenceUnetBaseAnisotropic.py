import torch
import nibabel as nib
import numpy as np
import os
import torch.nn.functional as F
from UNetAnisotropic import UNet
import matplotlib.pyplot as plt
import cv2

# --- CONFIGURATION ---
# Scale factors must match your UNet upscale logic (D=2, H=2, W=1)
SCALE_D = 2
SCALE_H = 2
SCALE_W = 1
BASE_FILTERS = 32


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
    # 1. Load the data
    data_lr = nib.load(LR_INPUT).get_fdata()
    data_sr = nib.load(OUTPUT).get_fdata()
    data_gt = nib.load(GT_FILE).get_fdata()

    # 2. Slice Selection (W-axis)
    # W-scale is 1, so slice_idx is the same for all
    slice_idx = data_lr.shape[2] // 2

    def prep_view(data, s_idx):
        # Extract 2D slice
        v = data[:, :, s_idx, 0] if data.ndim == 4 else data[:, :, s_idx]
        # Return raw for now; we rotate after resizing to avoid shape confusion
        return v

    # Initial extraction
    raw_lr = prep_view(data_lr, slice_idx)
    raw_sr = prep_view(data_sr, slice_idx)
    raw_gt = prep_view(data_gt, slice_idx)

    # 3. Master Shape Alignment
    # We want everything to match the Super-Res shape
    target_h, target_w = raw_sr.shape

    def align_to_sr(img, target_shape):
        # If transposed (e.g., 208, 160 vs 160, 208), flip it
        if img.shape[0] == target_shape[1] and img.shape[1] == target_shape[0]:
            img = img.T

        # Resize to match SR pixel grid
        if img.shape != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

        # Now apply the rotation for anatomical viewing
        return np.rot90(img)

    view_sr = np.rot90(raw_sr)
    view_lr = align_to_sr(raw_lr, raw_sr.shape)
    view_gt = align_to_sr(raw_gt, raw_sr.shape)

    # 4. Difference Map (Broadcasting will work now!)
    diff_map = np.abs(view_sr - view_gt)

    # 5. Plotting (4 Panes)
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    titles = ["Raw LR", "Raw SR", "Ground Truth", "Absolute Error"]
    views = [view_lr, view_sr, view_gt, diff_map]

    for i in range(4):
        # Independent scaling for first 3, Heatmap scaling for 4th
        if i < 3:
            v_min, v_max = np.percentile(views[i], [2, 98])
            axes[i].imshow(views[i], cmap='gray', vmin=v_min, vmax=v_max)
        else:
            # Heatmap uses 'inferno' to show error density
            axes[i].imshow(views[i], cmap='inferno', vmin=0, vmax=np.percentile(views[i], 99))

        axes[i].set_title(titles[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("debug_comparison_fixed.png")
    plt.show()


@torch.inference_mode()
def run_temporal_inference(model_path, input_path, output_path, device='cuda'):
    # 1. Load Checkpoint and Detect Channels
    checkpoint = torch.load(model_path, map_location=device)

    # DYNAMIC FIX: Extract input channels from the first layer of the checkpoint
    # Checkpoint shape: [out_filters, in_channels, k, k, k]
    trained_in_channels = checkpoint['down_conv1.conv.double_conv.0.weight'].shape[1]
    print(f"Detected {trained_in_channels} channels from checkpoint. Adjusting model...")

    # 2. Initialize Model with the CORRECT channel count
    model = UNet(in_channels=trained_in_channels,
                 out_channels=trained_in_channels,
                 base_filters=BASE_FILTERS).to(device)

    model.load_state_dict(checkpoint)
    model.eval()

    # 3. Load NIfTI
    img = nib.load(input_path)
    full_data = img.get_fdata()
    affine, header = img.affine, img.header
    total_timepoints = full_data.shape[3]

    all_reconstructed_chunks = []

    # 4. Temporal Sliding Window Inference
    # Ensure we use the detected channel count for the stride
    for t in range(0, total_timepoints - trained_in_channels + 1, trained_in_channels):
        print(f"Processing volumes {t} to {t + trained_in_channels}...")
        chunk = full_data[..., t:t + trained_in_channels]

        input_tensor, pads, stats, orig_lr_shape = load_and_preprocess(chunk)
        input_tensor = input_tensor.to(device)

        # 5. Calculate HR Target Shape for internal alignment
        target_d = orig_lr_shape[0] * SCALE_D
        target_h = orig_lr_shape[1] * SCALE_H
        target_w = orig_lr_shape[2] * SCALE_W
        hr_target_shape = (1, trained_in_channels, target_d, target_h, target_w)

        # 6. Model Prediction with Autocast
        with torch.amp.autocast('cuda'):
            output = model(input_tensor, hr_target_shape)

        # 7. Inverse Normalization using captured stats
        recon_chunk = output.cpu().numpy()[0].transpose(1, 2, 3, 0)
        data_min, data_max = stats
        recon_chunk = recon_chunk * (data_max - data_min) + data_min

        all_reconstructed_chunks.append(recon_chunk)

    # 8. Reassemble 4D Volume
    final_output = np.concatenate(all_reconstructed_chunks, axis=3)

    # 9. Update Affine for correct spatial resolution
    new_affine = affine.copy()
    new_affine[0, 0] /= SCALE_D
    new_affine[1, 1] /= SCALE_H
    new_affine[2, 2] /= SCALE_W

    # 10. Save HR Output
    new_img = nib.Nifti1Image(final_output, new_affine, header)
    nib.save(new_img, output_path)
    print(f"Reconstruction complete: {output_path}")
    show_result()


if __name__ == "__main__":
    MODEL_PATH = './CheckpointBaseUNetAnisotropic/super-resolution-finale.pth'
    GT_FILE = '/fast_storage/flk7161/data/hr_Anisotropic/1b80b162-ae30-42b1-9ad0-9677ee768f81_sub-04_ses-r08_task-orientation_rec-dico_run-08_bold_HR.nii.gz'
    LR_INPUT = '/fast_storage/flk7161/data/lr_Anisotropic/1b80b162-ae30-42b1-9ad0-9677ee768f81_sub-04_ses-r08_task-orientation_rec-dico_run-08_bold_LR.nii.gz'
    OUTPUT = '/fast_storage/flk7161/inference/reconstructed_HR.nii.gz'

    if os.path.exists(MODEL_PATH):
        run_temporal_inference(MODEL_PATH, LR_INPUT, OUTPUT)
    else:
        print("Checkpoint not found!")
