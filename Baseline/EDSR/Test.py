import torch
import torch.nn as nn
from EDSR import EDSRBaseline


def test_edsr_baseline():
    # -------------------------
    # 1. Configuration
    # -------------------------
    BATCH_SIZE = 1
    TIME_STEPS = 4  # Number of temporal volumes (channels)
    D, H, W = 16, 32, 32  # Input spatial dimensions

    BASE_FILTERS = 32  # Channels in the residual body
    N_RESBLOCKS = 8  # Depth of the EDSR body
    SCALE_FACTOR = 2

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Initializing EDSR Baseline on {DEVICE} ---")

    # -------------------------
    # 2. Mock Model Initialization
    # -------------------------
    # Replace EDSRBaseline with your class name
    model = EDSRBaseline(
        in_channels=1,  # Internally we view as (B*T, 1, D, H, W)
        out_channels=1,
        base_filters=BASE_FILTERS,
        n_resblocks=N_RESBLOCKS,
        scale_factor=SCALE_FACTOR
    ).to(DEVICE)

    # -------------------------
    # 3. Create Mock Data
    # -------------------------
    # Dataset provides [B, T, D, H, W]
    mock_input = torch.randn(BATCH_SIZE, TIME_STEPS, D, H, W).to(DEVICE)

    # Target shape for in-plane upsampling (D=1, H=2, W=2)
    D_t = D * SCALE_FACTOR
    H_t = H * SCALE_FACTOR
    W_t = W
    target_shape = (BATCH_SIZE, TIME_STEPS, D_t, H_t, W_t)

    print(f"Input Shape:  {mock_input.shape}")
    print(f"Target Shape: {target_shape}")

    # -------------------------
    # 4. Forward Pass
    # -------------------------
    print("\n--- Starting Forward Pass ---")
    try:
        model.eval()
        with torch.no_grad():
            # EDSR should handle the temporal dimension internally like your UNet
            output = model(mock_input, target_shape)

        print(f"Output Shape: {output.shape}")

        # -------------------------
        # 5. Validation
        # -------------------------
        if output.shape == target_shape:
            print("\n✅ Success: EDSR output matches target dimensions!")

            # Memory Diagnostic
            memory_used = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
            print(f"Peak VRAM Usage: {memory_used:.2f} MB")
        else:
            print(f"\n⚠️ Shape Mismatch: Expected {target_shape}, got {output.shape}")

    except Exception as e:
        print(f"\n❌ Test Failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_edsr_baseline()