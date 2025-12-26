import torch
import traceback
# Ensure your file name matches where your new Bottleneck model is defined
from UNetSuperResBottle import UNet


def test_bottleneck_unet():
    # 1. Setup Mock Hyperparameters
    # Note: For fMRI, your model now expects B, T, D, H, W
    BATCH_SIZE = 2  # Testing with B > 1 to verify batch reshaping logic
    TIME_STEPS = 4
    DEPTH, HEIGHT, WIDTH = 16, 16, 16
    BASE_FILTERS = 16

    # out_channels should be 1 if you are predicting a single BOLD intensity
    # but we'll use TIME_STEPS if your architecture outputs a full sequence
    IN_CHANNELS = 1  # CNN shared weights approach (T is a separate dim)
    OUT_CHANNELS = 1

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"--- Testing ViT-Bottleneck U-Net on {DEVICE} ---")

    # 2. Initialize Model
    try:
        # Note: If your model treats T as channels at the start, use in_channels=TIME_STEPS
        model = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, base_filters=BASE_FILTERS).to(DEVICE)
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        traceback.print_exc()
        return

    # 3. Create Mock Data (B, T, D, H, W)
    # This simulates Z-normalized fMRI data
    mock_input = torch.randn(BATCH_SIZE, TIME_STEPS, DEPTH, HEIGHT, WIDTH).to(DEVICE)
    print(f"Input Shape: {mock_input.shape}")

    # 4. Forward Pass
    try:
        model.eval()
        with torch.no_grad():
            output = model(mock_input)

        print(f"Output Shape: {output.shape}")

        # 5. Validation Checks
        # Your target is usually 2x spatial upsampling.
        # Check if output is 5D (B, T, D, H, W) or 6D (B, T, C, D, H, W)

        # Check spatial dimensions (last 3)
        expected_spatial = (DEPTH * 2, HEIGHT * 2, WIDTH * 2)
        actual_spatial = output.shape[-3:]

        if actual_spatial == expected_spatial:
            print(f"✅ Success: Spatial dimensions are correct {actual_spatial}!")
        else:
            print(f"⚠️ Warning: Spatial mismatch. Expected {expected_spatial}, got {actual_spatial}")

        # Check Temporal dimension preservation
        if output.shape[1] == TIME_STEPS:
            print(f"✅ Success: Temporal dimension (T={TIME_STEPS}) preserved!")
        else:
            print(f"⚠️ Warning: Temporal dimension lost. Got {output.shape[1]}")

    except Exception as e:
        print(f"❌ Forward Pass Failed: {str(e)}")
        # This will pinpoint if the error is in the Transformer reshape or UpSample cat
        traceback.print_exc()


if __name__ == "__main__":
    test_bottleneck_unet()