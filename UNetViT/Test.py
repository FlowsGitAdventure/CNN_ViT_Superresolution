import torch
from UNetHybridSuperResolution import UNet


def test_fmri_unet():
    # 1. Setup Mock Hyperparameters matching your CreateDataset
    BATCH_SIZE = 1
    TIME_STEPS = 4  # Matches your time_steps_per_sample
    DEPTH, HEIGHT, WIDTH = 16, 16, 16  # Small spatial dims for quick CPU/GPU test
    BASE_FILTERS = 16

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Testing Hybrid U-Net on {DEVICE}...")

    # 2. Initialize Model
    # in_channels = TIME_STEPS because your model treats time as the initial feature dim
    model = UNet(in_channels=TIME_STEPS, out_channels=TIME_STEPS, base_filters=BASE_FILTERS).to(DEVICE)

    # 3. Create Mock Data (Batch, Time, D, H, W)
    # This simulates a small patch of fMRI data
    mock_input = torch.randn(BATCH_SIZE, TIME_STEPS, DEPTH, HEIGHT, WIDTH).to(DEVICE)

    print(f"Input Shape: {mock_input.shape}")

    # 4. Forward Pass
    try:
        model.eval()  # Set to eval to avoid BatchNorm issues with batch size 1
        with torch.no_grad():
            output = model(mock_input)

        print(f"Output Shape: {output.shape}")

        # 5. Validation Checks
        # The output should be 2x larger spatially if you kept the final upsample block
        expected_shape = (BATCH_SIZE, TIME_STEPS, DEPTH * 2, HEIGHT * 2, WIDTH * 2)

        if output.shape == expected_shape:
            print("✅ Success: Output dimensions are correct!")
        else:
            print(f"⚠️ Warning: Shape mismatch. Expected {expected_shape}, got {output.shape}")

    except Exception as e:
        print(f"❌ Test Failed: {str(e)}")
        # This will help catch the specific line where reshaping or concatenation fails
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fmri_unet()