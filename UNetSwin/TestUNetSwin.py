import torch
import torch.nn as nn
import time
# Replace with your actual filename for the MONAI Swin model
from UNetSwinSuperResolution import MonaiSwinEncoderSR


def test_swin_memory_and_shape():
    # 1. Setup Mock Hyperparameters
    BATCH_SIZE = 1
    TIME_STEPS = 4
    DEPTH, HEIGHT, WIDTH = 16, 16, 16
    BASE_FILTERS = 24  # MONAI Swin default in your class

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"--- Starting Swin-MONAI Test on {DEVICE} ---")

    # 2. Initialize Model
    try:
        model = MonaiSwinEncoderSR(
            in_channels=1,
            out_channels=1,
            base_filters=BASE_FILTERS,
            window_size=(4, 4, 4)
        ).to(DEVICE)
        print("✅ Swin Model initialized.")
    except Exception as e:
        print(f"❌ Init Failed: {e}")
        return

    # 3. Create Mock Data (B, T, D, H, W)
    mock_input = torch.randn(BATCH_SIZE, TIME_STEPS, DEPTH, HEIGHT, WIDTH).to(DEVICE)
    print(f"Input Shape: {mock_input.shape}")

    # 4. Memory Profiling & Forward Pass
    try:
        if DEVICE == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / 1024 ** 2

        start_time = time.time()

        model.eval()
        with torch.no_grad():
            output = model(mock_input)

        end_time = time.time()

        if DEVICE == 'cuda':
            peak_mem = torch.cuda.max_memory_reserved() / 1024 ** 2
            print(f"Peak VRAM Usage: {peak_mem:.2f} MB")

        print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")
        print(f"Output Shape: {output.shape}")

        # 5. Validation Logic
        # Your final_upsample has two 2x steps (8 -> 16 -> 32)
        # So output should be 32x32x32
        expected_spatial = (32, 32, 32)
        actual_spatial = output.shape[-3:]

        if actual_spatial == expected_spatial:
            print(f"✅ Success: Spatial dimensions are correctly upsampled to {actual_spatial}!")
        else:
            print(f"⚠️ Warning: Shape mismatch. Expected {expected_spatial}, got {actual_spatial}")
            print("Check if your final_upsample has one or two ConvTranspose3d layers.")

        # 6. Channel Check
        if output.ndim == 6 and output.shape[2] == 1:
            print("✅ Success: Channel dimension (C=1) is correctly preserved.")
        else:
            print(f"⚠️ Warning: Unexpected channel structure. Shape: {output.shape}")

    except Exception as e:
        print(f"❌ Test Failed during Forward Pass!")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_swin_memory_and_shape()