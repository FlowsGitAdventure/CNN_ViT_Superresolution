import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
import os

# --- MODEL IMPORT ---
from EDSR import EDSRBaseline
from GetRandomData import GetRandomData
from CreateDataset import UnifiedSRDataset
from lossesEDSR.KSpaceGradCharLoss import FourierCharbonier
from PeakSignalNoiseRatio import calculate_psnr
from StructuralSimilarity import calculate_ssim_score


LR_DIR = '/fast_storage/flk7161/data/lr_Anisotropic'
HR_DIR = '/fast_storage/flk7161/data/hr_Anisotropic'


# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_FILTERS = 32   # Former: 32
BATCH_SIZE = 1
START_LR = 1e-4
WEIGHT_DECAY = 1e-3
NUM_EPOCHS = 3
DATA_RANGE = 1.0
ACCUMULATION_STEPS = 8
SAVE_DIR = './CheckpointEDSR'
os.makedirs(SAVE_DIR, exist_ok=True)


def get_max_dimensions_HR(file_list):
    max_d, max_h, max_w = 0, 0, 0
    for f in file_list:
        file = os.path.join(HR_DIR, f)
        shape = nib.load(file).header.get_data_shape()
        max_d = max(max_d, shape[0])
        max_h = max(max_h, shape[1])
        max_w = max(max_w, shape[2])
    return (max_d, max_h, max_w)


def get_max_dimensions_LR(file_list):
    max_d, max_h, max_w = 0, 0, 0
    for f in file_list:
        file = os.path.join(LR_DIR, f)
        shape = nib.load(file).header.get_data_shape()
        max_d = max(max_d, shape[0])
        max_h = max(max_h, shape[1])
        max_w = max(max_w, shape[2])
    return (max_d, max_h, max_w)


def crop_to_valid_fov(x, ref):
    Dx, Hx, Wx = x.shape[-3:]
    Dr, Hr, Wr = ref.shape[-3:]

    assert Dx >= Dr and Hx >= Hr and Wx >= Wr, \
        f"Cannot crop {x.shape} to {ref.shape}"

    return x[..., :Dr, :Hr, :Wr]


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, device, loader, optimizer, loss_fn, epoch, num_epochs, scaler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_log = []

    progress_bar = tqdm(enumerate(loader), total=len(loader),
                        desc=f"Epoch {epoch}/{num_epochs}")

    for batch_idx, (lr, hr, mask, spacing) in progress_bar:
        # Warmup for the first epoch to stabilize training
        if epoch == 0:
            warmup_factor = (batch_idx + 1) / len(loader)
            for pg in optimizer.param_groups:
                pg['lr'] = START_LR * warmup_factor

        # Move all tensors to device
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        spacing = spacing.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            lr = lr.squeeze(1)
            out = model(lr, hr.shape)
            out = crop_to_valid_fov(out, hr)

            loss, loss_components = loss_fn(out, hr, spacing)
            loss = loss / ACCUMULATION_STEPS

        # 3. Scaled Backward Pass
        scaler.scale(loss).backward()

        # 4. Step optimizer after accumulation
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Logging
            loss_log.append(loss.item() * ACCUMULATION_STEPS)
            progress_bar.set_postfix(
                Total=f"{loss_components['Total']:.4f}",
                Charb=f"{loss_components['Charb']:.4f}",
                Grad=f"{loss_components['Grad']:.4f}",
                KSpace=f"{loss_components['KSpace']:.4f}"
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


def validation(model, device, criterion, loader):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for lr, hr, mask, spacing in loader:
            lr, hr = lr.to(device), hr.to(device)
            mask, spacing = mask.to(device), spacing.to(device)

            with torch.amp.autocast('cuda'):
                out = model(lr, hr.shape)
                out = crop_to_valid_fov(out, hr)

                v_loss, _ = criterion(out, hr, spacing)

            out = out.float()
            batch_psnr = calculate_psnr(out, hr, mask, data_range=DATA_RANGE)
            batch_ssim = calculate_ssim_score(out, hr, mask, data_range=DATA_RANGE)

            batch_size = lr.size(0)
            total_loss += v_loss.item() * batch_size
            total_psnr += batch_psnr * batch_size
            total_ssim += batch_ssim * batch_size
            total_samples += batch_size

        avg_psnr = total_psnr / total_samples
        avg_ssim = total_ssim / total_samples
        avg_loss = total_loss / total_samples

        return avg_psnr, avg_ssim, avg_loss


if __name__ == "__main__":
    # Data Setup
    data_selector = GetRandomData(LR_DIR, HR_DIR, 10, 1, is_random=True)
    train_files, val_files, hr_train_files, hr_val_files = data_selector.get_data()

    # Example usage before creating the Dataset
    MAX_LR_SHAPE = get_max_dimensions_LR(train_files)
    MAX_HR_SHAPE = get_max_dimensions_HR(hr_train_files)

    train_dataset = UnifiedSRDataset(train_files, hr_train_files, LR_DIR, HR_DIR, MAX_LR_SHAPE, MAX_HR_SHAPE, normalization="cnn_minmax")
    val_dataset = UnifiedSRDataset(val_files, hr_val_files, LR_DIR, HR_DIR, MAX_LR_SHAPE, MAX_HR_SHAPE, normalization="cnn_minmax", train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training on: {DEVICE}")
    model = EDSRBaseline(in_channels=1, out_channels=1, base_filters=BASE_FILTERS).to(DEVICE)
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=START_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    criterion = FourierCharbonier(NUM_EPOCHS)

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

    best_psnr = 30.0

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
    plt.savefig('training_results_EDSR.png')
    plt.show()
