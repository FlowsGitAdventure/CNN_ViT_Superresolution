import torch
import torch.optim as optim
from torch._utils import _get_async_or_non_blocking
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data._utils import pin_memory
from tqdm import tqdm
import os
from UNetHybridSuperResolution import UNet

from GetRandomData import GetRandomData
from CreateDatasetHybridViT import CreateDataset
from fMRIHybridLoss import HybridLossSSIML1Temp as combined_loss

from PeakSignalNoiseRatio import calculate_psnr
from StructuralSimilarity import calculate_ssim_score


LR_DIR = '../Downsampling/Low_Res_08_Rician'
HR_DIR = '../Downsampling/High_Res_08_Rician'
LR_DIR = '/fast_storage/flk7161/data/lr'
HR_DIR = '/fast_storage/flk7161/data/hr'


# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_FILTERS = 16   # Former: 32
BATCH_SIZE = 1
START_LR = 0.0001
NUM_EPOCHS = 10
DATA_RANGE = 6.0 # Estimated for Z-scores
ACCUMULATION_STEPS = 4
SAVE_DIR = './Models_CNN_ViT'
os.makedirs(SAVE_DIR, exist_ok=True)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, device, loader, optimizer, loss_fn, epoch, num_epochs, scaler):
    model.train()
    loss_log = []
    optimizer.zero_grad()

    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}/{num_epochs}")

    for batch_idx, (lr, hr) in progress_bar:
        # 1. Linear Warmup for the first epoch only
        if epoch == 0:
            warmup_factor = (batch_idx + 1) / len(loader)
            for param_group in optimizer.param_groups:
                param_group['lr'] = START_LR * warmup_factor

        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

        # 2. Mixed Precision Forward Pass
        with torch.amp.autocast('cuda'):
            out = model(lr)
            loss, loss_components = loss_fn(out, hr)
            loss = loss / ACCUMULATION_STEPS

        # 3. Scaled Backward Pass
        scaler.scale(loss).backward()

        # 4. Step optimizer after accumulation
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_log.append(loss.item() * ACCUMULATION_STEPS)
        progress_bar.set_postfix(L1=f"{loss_components['L1']:.4f}",
                                 Temp=f"{loss_components['Temp']:.4f}")

    # 5. Reset LR to START_LR after warmup epoch to be safe
    if epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = START_LR

    avg_loss = sum(loss_log) / len(loss_log)
    print(f'Epoch {epoch} | Training Loss: {avg_loss:.4f}')
    return avg_loss, loss_log


def validation(model, device, loss_fn, loader):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    with torch.inference_mode():
        for lr, hr in loader:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                out = model(lr)

            out = out.float()
            # Use the Z-score compatible data range
            batch_psnr = calculate_psnr(out, hr, data_range=DATA_RANGE)
            batch_ssim = calculate_ssim_score(out, hr, data_range=DATA_RANGE)

            batch_size = lr.size(0)
            total_psnr += batch_psnr.item() * batch_size
            total_ssim += batch_ssim.item() * batch_size
            total_samples += batch_size

    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    print(f'Epoch {epoch} | Avg PSNR: {avg_psnr} | Avg SSIM: {avg_ssim}')
    return avg_psnr, avg_ssim


if __name__ == "__main__":
    # Load dataset
    data_selector = GetRandomData(LR_DIR, HR_DIR, 16, 4, is_random=True)
    train_files, validation_files, hr_train_files, hr_validation_files = data_selector.get_data()
    print("random data selected")
    train_dataset = CreateDataset(train_files, hr_train_files, LR_DIR, HR_DIR)
    print("Train Dataset created")
    validation_dataset = CreateDataset(validation_files, hr_validation_files, LR_DIR, HR_DIR)
    print("Validation Dataset created")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    print("Dataloader created")

    print(f"Training on: {DEVICE}")

    # prepare training
    sample_lr, _ = train_dataset[0]
    model = UNet(in_channels=sample_lr.shape[0], out_channels=sample_lr.shape[0], base_filters=BASE_FILTERS).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=START_LR)
    criterion = combined_loss()
    scaler = torch.amp.GradScaler("cuda")

    total_train_loss = []

    total_val_loss = []
    total_psnr_metric = []
    total_ssim_metric = []

    # Initializing the Scheduler
    # 'max' because we want to maximize PSNR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=2
    )

    best_psnr = 25.0

    # Trainings loop
    print("Train loop start")
    for epoch in range(NUM_EPOCHS):
        train_loss, _ = train(model, DEVICE, train_loader, optimizer, criterion, epoch, NUM_EPOCHS, scaler)
        torch.cuda.empty_cache()
        avg_psnr, avg_ssim = validation(model, DEVICE, criterion, val_loader)
        torch.cuda.empty_cache()

        total_train_loss.append(train_loss)
        total_psnr_metric.append(avg_psnr)
        total_ssim_metric.append(avg_ssim)

        scheduler.step(avg_psnr)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/super-resolution_ep{epoch + 1}.pth")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print(f"--- New Best PSNR: {best_psnr:.2f}! Model Saved ---")

    torch.save(model.state_dict(), f"{SAVE_DIR}/super-resolution-finale.pth")
    print(f"Model saved to {SAVE_DIR}")

    # Visualizing results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, NUM_EPOCHS + 1), total_train_loss, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), total_val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, NUM_EPOCHS + 1), total_ssim_metric, label='Structural Similarity Metric')
    plt.plot(range(1, NUM_EPOCHS + 1), total_psnr_metric, label='Peak Signal To Noise Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR and SSIM Over Epochs')
    plt.title('Metrics: PSNR and SSIM')
    plt.legend()

    plt.savefig('training_validation_plots.png')
    print("Training and validation plots saved as 'training_validation_plots.png'")
    plt.show()
