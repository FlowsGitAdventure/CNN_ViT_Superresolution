import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# --- MODEL IMPORT ---
from UNet import UNet
from GetRandomData import GetRandomData
from CreateDataset import UnifiedSRDataset
from CombinedSSIML1Loss import CombinedSSIML1Loss as combined_loss
from VolumetricSRLoss import VolumetricSRLoss
from PeakSignalNoiseRatio import calculate_psnr
from StructuralSimilarity import calculate_ssim_score

LR_DIR = '/fast_storage/flk7161/data/lr'
HR_DIR = '/fast_storage/flk7161/data/hr'

# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_FILTERS = 32
BATCH_SIZE = 1
START_LR = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 10
DATA_RANGE = 1.0
ACCUMULATION_STEPS = 8
SAVE_DIR = './CheckpointBaseUNet'
os.makedirs(SAVE_DIR, exist_ok=True)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, device, loader, optimizer, loss_fn, epoch, num_epochs, scaler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_log = []

    progress_bar = tqdm(enumerate(loader), total=len(loader),
                        desc=f"Epoch {epoch}/{num_epochs}")

    for batch_idx, (lr, hr) in progress_bar:

        if epoch == 0:
            warmup_factor = (batch_idx + 1) / len(loader)
            for pg in optimizer.param_groups:
                pg['lr'] = START_LR * warmup_factor

        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(lr)
            loss, loss_components = loss_fn(out, hr)
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_log.append(loss.item() * ACCUMULATION_STEPS)
        progress_bar.set_postfix(
            L1=f"{loss_components['Charb']:.4f}",
            Temp=f"{loss_components['SSIM3D']:.4f}",
            SSIM=f"{loss_components['Grad']:.4f}",
            Edge=f"{loss_components['LowF']:.4f}"
        )

    if len(loader) % ACCUMULATION_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    if epoch == 0:
        for pg in optimizer.param_groups:
            pg['lr'] = START_LR

    avg_loss = sum(loss_log) / len(loss_log)
    return avg_loss


def validation(model, device, loss_fn, loader):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for lr, hr in loader:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

            # Forward Step
            with torch.amp.autocast('cuda'):
                out = model(lr)
                v_loss, _ = loss_fn(out, hr)

            out = out.float()
            if out.ndim == 6:
                out = out.squeeze(2)

            # ToDo: Check if better or worse, current_range = hr.max() - hr.min()
            # current_range = hr.max() - hr.min()
            # if current_range == 0:
            #     current_range = 1.0  # Prevent div by zero

            # Metric Calculation
            batch_psnr = calculate_psnr(out, hr, data_range=DATA_RANGE)
            batch_ssim = calculate_ssim_score(out, hr, data_range=DATA_RANGE)

            batch_size = lr.size(0)
            total_loss += v_loss.item() * batch_size
            total_psnr += batch_psnr.item() * batch_size
            total_ssim += batch_ssim.item() * batch_size
            total_samples += batch_size

    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    avg_loss = total_loss / total_samples
    print(f'Epoch {epoch} | Val Loss: {avg_loss:.4f} | Avg PSNR: {avg_psnr:.2f} | Avg SSIM: {avg_ssim:.4f}')
    return avg_psnr, avg_ssim, avg_loss


if __name__ == "__main__":
    # Data Setup
    data_selector = GetRandomData(LR_DIR, HR_DIR, 40, 4, is_random=True)
    train_files, val_files, hr_train_files, hr_val_files = data_selector.get_data()

    train_dataset = UnifiedSRDataset(train_files, hr_train_files, LR_DIR, HR_DIR, normalization="cnn_minmax")
    val_dataset = UnifiedSRDataset(val_files, hr_val_files, LR_DIR, HR_DIR, normalization="cnn_minmax", train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training on: {DEVICE}")

    sample_lr, _ = train_dataset[0]
    channels_in = sample_lr.shape[0]
    print(f"Train with {sample_lr.shape}")
    model = UNet(in_channels=channels_in, out_channels=channels_in, base_filters=BASE_FILTERS).to(DEVICE)
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=START_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    criterion = criterion = VolumetricSRLoss(data_range=1.0, lambda_charb=0.7, lambda_ssim=1.0, lambda_grad=0.5,
                                             lambda_lowfreq=0.1)

    # ToDo: Control loss factors as hyperparameters (not pre set)

    scaler = torch.amp.GradScaler("cuda")  # ToDo: Here also hyperparameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        threshold=0.01,
        mode='max',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
    )

    metrics = {"train_loss": [], "val_loss": [], "psnr": [], "ssim": []}

    best_psnr = 25.0

    print(f"Starting Training on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        criterion.epoch = epoch

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

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/super-resolution_ep{epoch + 1}.pth")

        if v_psnr > best_psnr:
            best_psnr = v_psnr
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")

    torch.save(model.state_dict(), f"{SAVE_DIR}/super-resolution-finale.pth")
    print(f"Model saved to {SAVE_DIR}")

    # Visualizing results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics["train_loss"], label="Train")
    plt.plot(metrics["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics["ssim"], label="SSIM")
    plt.title("SSIM Progress")
    plt.savefig('training_resultsBaseUNet.png')
    plt.show()
