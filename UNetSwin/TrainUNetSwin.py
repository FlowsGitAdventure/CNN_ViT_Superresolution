import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# --- MODEL IMPORT ---
# Ensure this matches your new Swin filename
from UNetSwinSuperResolution import MonaiSwinEncoderSR

from GetRandomData import GetRandomData
from CreateDatasetSwin import CreateDataset
from fMRIHybridLossSwin import HybridLossSSIML1Temp as combined_loss
from SSIMSwin import calculate_ssim_score
from PSNRSwin import calculate_psnr

# Paths
LR_DIR = '/fast_storage/flk7161/data/lr'
HR_DIR = '/fast_storage/flk7161/data/hr'

# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_FILTERS = 24  # Swin works best with multiples of 12 or 24
BATCH_SIZE = 1  # Keep at 1 for Swin memory stability
START_LR = 0.0001
NUM_EPOCHS = 20  # Transformers often need more epochs to converge
DATA_RANGE = 6.0
ACCUMULATION_STEPS = 4
SAVE_DIR = './Models_Swin_ViT'
os.makedirs(SAVE_DIR, exist_ok=True)


def train(model, device, loader, optimizer, loss_fn, epoch, num_epochs, scaler):
    model.train()
    loss_log = []
    optimizer.zero_grad()
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}/{num_epochs}")

    for batch_idx, (lr, hr) in progress_bar:
        # Linear Warmup (Crucial for Transformers to prevent gradient explosion)
        if epoch == 0:
            warmup_factor = (batch_idx + 1) / len(loader)
            for param_group in optimizer.param_groups:
                param_group['lr'] = START_LR * warmup_factor

        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            out = model(lr)
            # Ensure spatial alignment for loss if there's rounding in Swin stages
            loss, loss_components = loss_fn(out, hr)
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_log.append(loss.item() * ACCUMULATION_STEPS)
        progress_bar.set_postfix(L1=f"{loss_components['L1']:.4f}", SSIM=f"{loss_components['SSIM']:.4f}")

    avg_loss = sum(loss_log) / len(loss_log)
    return avg_loss


def validation(model, device, loss_fn, loader):
    model.eval()
    total_psnr, total_ssim, total_val_loss = 0.0, 0.0, 0.0
    total_samples = 0

    with torch.inference_mode():
        for lr, hr in loader:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(lr)
                v_loss, _ = loss_fn(out, hr)

            out = out.float()
            # 6D -> 5D Squeeze check (B, T, C, D, H, W) -> (B, T, D, H, W)
            if out.ndim == 6:
                out = out.squeeze(2)

            # Dynamic Data Range for accuracy
            current_range = (hr.max() - hr.min()).item()
            if current_range < 0.1: current_range = DATA_RANGE

            batch_psnr = calculate_psnr(out, hr, data_range=current_range)
            batch_ssim = calculate_ssim_score(out, hr, data_range=current_range)

            batch_size = lr.size(0)
            total_val_loss += v_loss.item() * batch_size
            total_psnr += batch_psnr.item() * batch_size
            total_ssim += batch_ssim.item() * batch_size
            total_samples += batch_size

    return total_psnr / total_samples, total_ssim / total_samples, total_val_loss / total_samples


if __name__ == "__main__":
    # Data Setup
    data_selector = GetRandomData(LR_DIR, HR_DIR, 16, 4, is_random=True)
    train_files, val_files, hr_train_files, hr_val_files = data_selector.get_data()

    train_dataset = CreateDataset(train_files, hr_train_files, LR_DIR, HR_DIR)
    val_dataset = CreateDataset(val_files, hr_val_files, LR_DIR, HR_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model Initialization
    # NOTE: Swin handles Time via internal reshaping, so in_channels=1 (Shared Weights)
    model = MonaiSwinEncoderSR(in_channels=1, out_channels=1, base_filters=BASE_FILTERS).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=START_LR, weight_decay=1e-5)
    criterion = combined_loss(data_range=DATA_RANGE)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    metrics = {"train_loss": [], "val_loss": [], "psnr": [], "ssim": []}
    best_psnr = 0.0

    print(f"Starting SwinUNet Training on {DEVICE}...")

    for epoch in range(NUM_EPOCHS):
        t_loss = train(model, DEVICE, train_loader, optimizer, criterion, epoch, NUM_EPOCHS, scaler)
        torch.cuda.empty_cache()

        v_psnr, v_ssim, v_loss = validation(model, DEVICE, criterion, val_loader)
        torch.cuda.empty_cache()

        metrics["train_loss"].append(t_loss)
        metrics["val_loss"].append(v_loss)
        metrics["psnr"].append(v_psnr)
        metrics["ssim"].append(v_ssim)

        print(f"Epoch {epoch} | Val PSNR: {v_psnr:.2f} | Val SSIM: {v_ssim:.4f}")

        scheduler.step(v_psnr)

        if v_psnr > best_psnr:
            best_psnr = v_psnr
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_swin_model.pth")

    # --- PLOTTING ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Train")
    plt.plot(metrics["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics["psnr"], label="PSNR")
    plt.title("PSNR Progress")
    plt.savefig('swin_training_results.png')
    plt.show()