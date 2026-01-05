import torch
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from UNet import UNet


def load_and_preprocess(data_chunk, multiple=16):
    """
    Captures original stats for de-normalization.
    """
    # 1. Clip and capture stats BEFORE normalization
    p99 = np.percentile(data_chunk, 99.5)
    data_chunk = np.clip(data_chunk, 0, p99)

    # We use these later to restore the intensity range
    orig_mean = np.mean(data_chunk)
    orig_std = np.std(data_chunk) + 1e-8

    # 2. Normalize
    normalized_data = (data_chunk - orig_mean) / orig_std

    # 3. Convert to Tensor: [T, X, Y, Z] -> [1, T, X, Y, Z]
    tensor = torch.from_numpy(normalized_data).float().permute(3, 0, 1, 2).unsqueeze(0)

    # 4. PAD TO MULTIPLE OF 16
    d, h, w = tensor.shape[2:]
    pad_d = (multiple - d % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    tensor_padded = F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)

    # Return the stats along with the tensor and padding information
    return tensor_padded, (pad_d, pad_h, pad_w), (orig_mean, orig_std)


def show_result():
    # 1. Load the data
    data_sr = nib.load(RESULT_FILE).get_fdata()
    data_lr = nib.load(TEST_FILE).get_fdata()
    data_gt = nib.load(GT_FILE).get_fdata()

    # 2. Select matching slices
    # If LR is 16 slices and SR is 64 slices, index 7 in LR is index 28 in SR
    slice_lr = 7
    scale_factor = data_sr.shape[2] // data_lr.shape[2]  # Should be 4
    slice_sr = slice_lr * scale_factor

    # 3. Extract the 2D planes (assuming time dimension is last or squeezed)
    # We take the first timepoint [..., 0] if 4D
    view_lr = data_lr[:, :, slice_lr, 0] if data_lr.ndim == 4 else data_lr[:, :, slice_lr]
    view_sr = data_sr[:, :, slice_sr, 0] if data_sr.ndim == 4 else data_sr[:, :, slice_sr]
    view_gt = data_gt[:, :, slice_sr, 0] if data_gt.ndim == 4 else data_gt[:, :, slice_sr]

    # 4. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    print(f"SR Range: min={data_sr.min():.4f}, max={data_sr.max():.4f}, mean={data_sr.mean():.4f}")

    axes[0].imshow(np.rot90(view_lr), cmap='gray')
    axes[0].set_title(f"Low-Res (Slice {slice_lr})")
    axes[0].axis('off')

    axes[1].imshow(np.rot90(view_sr), cmap='gray')
    axes[1].set_title(f"Super-Res (Slice {slice_sr})")
    axes[1].axis('off')

    axes[2].imshow(np.rot90(view_gt), cmap='gray')
    axes[2].set_title(f"Ground Truth (Slice {slice_sr})")
    axes[2].axis('off')

    plt.tight_layout()

    # 5. Save the result
    output_img_path = "sr_comparison_result.png"
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    print(f"Comparison image saved to {output_img_path}")
    plt.show()


def save_as_nifti(data, affine, header, output_path):
    """Saves a numpy array back to NIfTI format."""
    # Ensure the data is in the correct orientation
    new_img = nib.Nifti1Image(data, affine, header)
    nib.save(new_img, output_path)
    print(f"Successfully saved to: {output_path}")


@torch.inference_mode()
def run_temporal_inference(model_path, input_path, output_path, device='cuda'):
    # 1. Load Checkpoint to get the required channel count
    checkpoint = torch.load(model_path, map_location=device)
    expected_channels = checkpoint['down_conv1.conv.double_conv.0.weight'].shape[1]
    print(f"Model requires chunks of {expected_channels} timepoints.")

    # 2. Setup Model
    model = UNet(in_channels=expected_channels, out_channels=expected_channels, base_filters=32).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. Load full 4D fMRI volume
    img = nib.load(input_path)
    full_data = img.get_fdata()  # Shape: (X, Y, Z, T)
    affine, header = img.affine, img.header

    total_timepoints = full_data.shape[3]
    print(f"Total timepoints in file: {total_timepoints}")

    # 4. Temporal Loop
    all_reconstructed_chunks = []

    # Loop through time in steps of 'expected_channels'
    for t in range(0, total_timepoints - expected_channels + 1, expected_channels):
        print(f"Processing volumes {t} to {t + expected_channels}...")

        # Slice the raw data chunk
        chunk = full_data[..., t:t + expected_channels]

        # --- UPDATE 1: Capture stats and receive them from load_and_preprocess ---
        # Your updated load_and_preprocess now returns 'stats' (mean, std)
        input_tensor, pads, stats = load_and_preprocess(chunk)
        orig_mean, orig_std = stats
        input_tensor = input_tensor.to(device)

        with torch.amp.autocast('cuda'):
            output = model(input_tensor)

        # Crop the result (considering the 4x upscale factor)
        pd, ph, pw = pads
        d_end = output.shape[2] - (pd * 4) if pd > 0 else None
        h_end = output.shape[3] - (ph * 4) if ph > 0 else None
        w_end = output.shape[4] - (pw * 4) if pw > 0 else None

        output = output[:, :, :d_end, :h_end, :w_end]

        # Convert back to (X, Y, Z, T)
        # Remove batch, transpose (T, X, Y, Z) -> (X, Y, Z, T)
        recon_chunk = output.cpu().numpy()[0].transpose(1, 2, 3, 0)

        # --- UPDATE 2: Apply De-normalization before appending ---
        # Reverse the Z-score: (Result * Std) + Mean
        recon_chunk = (recon_chunk * orig_std) + orig_mean

        # Clip negative values to zero to ensure a clean black background
        recon_chunk = np.clip(recon_chunk, 0, None)

        all_reconstructed_chunks.append(recon_chunk)

    # 5. Restack and Save
    final_output = np.concatenate(all_reconstructed_chunks, axis=3)

    # Update header for new spatial dimensions (SR is 4x larger)
    new_img = nib.Nifti1Image(final_output, affine, header)
    nib.save(new_img, output_path)
    print(f"Full 4D Super-Resolution volume saved to {output_path}")
    show_result()


if __name__ == "__main__":
    MODEL_CHECKPOINT = './CheckpointBaseUNet/super-resolution-finale.pth'
    GT_FILE = '/fast_storage/flk7161/data/hr/1d47cd9f-2438-42cd-9ee1-2f2cddf75148_sub-17_ses-r08_task-coverage_bold_HR.nii.gz'
    TEST_FILE = '/fast_storage/flk7161/data/lr/1d47cd9f-2438-42cd-9ee1-2f2cddf75148_sub-17_ses-r08_task-coverage_bold_LR.nii.gz'
    RESULT_FILE = '/fast_storage/flk7161/inference/reconstructed_fMRI.nii.gz'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(MODEL_CHECKPOINT):
        run_temporal_inference(MODEL_CHECKPOINT, TEST_FILE, RESULT_FILE, DEVICE)
    else:
        print("Model checkpoint not found!")
