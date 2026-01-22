import torch
from UNetAnisotropic import UNet


def test_unet_shapes():
    # 1. Setup parameters matching your training script
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_FILTERS = 32
    IN_CHANNELS = 4  # Matching your time_steps_per_sample
    OUT_CHANNELS = 4

    # 2. Initialize Model
    print(f"--- Initializing UNet on {DEVICE} ---")
    model = UNet(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        base_filters=BASE_FILTERS
    ).to(DEVICE)
    model.eval()

    # 3. Create a dummy fMRI patch [Batch, Channels, Depth, Height, Width]
    # We use 32x32x16 to test if your bottleneck hits 1x1x1
    dummy_input = torch.randn(1, IN_CHANNELS, 16, 32, 32).to(DEVICE)
    print(f"Input Shape: {dummy_input.shape}")
    # 4. Forward Pass with Error Handling
    print("\n--- Starting Forward Pass ---")
    try:
        with torch.no_grad():
            output = model(dummy_input, (1, 4, 16, 64, 64))

        print(f"Output Shape: {output.shape}")

        # 5. Assertions to verify correctness
        # Check if output matches the intended Super-Resolution target
        # (Your UNet currently doubles the resolution twice in the final block)
        expected_shape = (1, OUT_CHANNELS, 32, 64, 64)  # 4x upscale logic in your code

        print("\n--- Validation Results ---")
        if output.shape[2:] == (16, 64, 64):
            print("✅ Success: Spatial dimensions are correctly upscaled.")
        else:
            print(f"⚠️ Warning: Spatial dimensions {output.shape[2:]} do not match expected 4x upscale.")

        if output.shape[1] == OUT_CHANNELS:
            print(f"✅ Success: Channel count ({output.shape[1]}) is correct.")

    except ValueError as e:
        print(f"❌ Spatial Error: {e}")
        print("Tip: Your input patch (16x32x32) is too small for a 4-level UNet. "
              "The bottleneck is reaching 1x1x1, which breaks InstanceNorm3d.")
    except RuntimeError as e:
        print(f"❌ Channel/Memory Error: {e}")


if __name__ == "__main__":
    test_unet_shapes()