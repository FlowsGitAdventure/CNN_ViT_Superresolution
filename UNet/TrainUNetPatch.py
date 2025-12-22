import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from UNetSuperRes import UNet

from GetRandomData import GetRandomData
from CreateDataset import CreateDataset
from CombinedSSIML1Loss import CombinedSSIML1Loss as ssim_l1

from PeakSignalNoiseRatio import calculate_psnr
from StructuralSimilarity import calculate_ssim_score


LR_DIR = '../Downsampling/Low_Res_08_Rician'
HR_DIR = '../Downsampling/High_Res_08_Rician'


# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_FILTERS = 16   # Former: 32
BATCH_SIZE = 1
START_LR = 0.1
NUM_EPOCHS = 10
SAVE_DIR = './checkpoints1'
os.makedirs(SAVE_DIR, exist_ok=True)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, device, loader, optimizer, loss_fn, epoch, num_epochs):
    print(f"train start epoch {epoch}")
    model.train()
    print("1")
    loss_log = []
    print("2")

    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}/{num_epochs}")
    print("3")
    for batch_idx, (lr, hr) in enumerate(loader):
        print(f"Train on {batch_idx}")
        lr, hr = lr.to(device), hr.to(device)
        print("Data loaded to device")

        # Forward step
        print("Forward step")
        out = model(lr)
        loss, _ = loss_fn(out, hr)
        # loss = loss_fn.forward(out, hr)

        # Backward step
        print("Baackward step")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())

        # Progress
        progress_bar.set_postfix(loss=loss)

    avg_loss = sum(loss_log) / len(loss_log)
    print(f'Epoch {epoch} | Training Loss: {avg_loss:.4f}')
    return avg_loss, loss_log


def validation(model, device, loss_fn, loader):
    model.eval()
    loss_log = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)

            # Forward Step
            out = model(lr)

            loss, _ = loss_fn(out, hr)
            loss_log.append(loss.item())

            batch_psnr = calculate_psnr(out, hr, data_range=1.0)
            batch_ssim = calculate_ssim_score(out, hr, data_range=1.0)

            batch_size = lr.size(0)
            total_psnr += batch_psnr.item() * batch_size
            total_ssim += batch_ssim.item() * batch_size
            total_samples += batch_size

    avg_loss = sum(loss_log) / len(loss_log)
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    print(f'Epoch {epoch} || Validation Loss: {avg_loss:.4f} || Avg PSNR: {avg_psnr} || Avg SSIM: {avg_ssim}')
    return avg_loss, avg_psnr, avg_ssim, loss_log


if __name__ == "__main__":
    # Load dataset
    data_selector = GetRandomData(LR_DIR, HR_DIR, 16, 4, is_random=True)
    train_files, validation_files, hr_train_files, hr_validation_files = data_selector.get_data()
    print("random data selected")
    train_dataset = CreateDataset(train_files, hr_train_files, LR_DIR, HR_DIR)
    print("Train Dataset created")
    validation_dataset = CreateDataset(validation_files, hr_validation_files, LR_DIR, HR_DIR)
    print("Validation Dataset created")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(validation_dataset, batch_size=10, shuffle=False, num_workers=0, pin_memory=True)
    print("Dataloader created")

    print(f"Training on: {DEVICE}")

    # prepare training
    sample_lr, _ = train_dataset[0]
    channels_in = sample_lr.shape[0]
    print(f"Train with {sample_lr.shape}")
    model = UNet(in_channels=channels_in, out_channels=channels_in, base_filters=BASE_FILTERS).to(DEVICE)
    print("model loaded")
    learning_rate = START_LR
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # criterion = nn.L1Loss()
    criterion = ssim_l1(lambda_l1=1.0, lambda_ssim=0.05)

    total_train_loss = []

    total_val_loss = []
    total_psnr_metric = []
    total_ssim_metric = []

    # Trainings loop
    print("Train loop start")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_loss_log = train(model=model, device=DEVICE, loss_fn=criterion, optimizer=optimizer, loader=train_loader, num_epochs=NUM_EPOCHS, epoch=epoch)
        val_loss, avg_psnr_metric, avg_ssim_metric, val_loss_log = validation(model, DEVICE, criterion, val_loader)

        total_train_loss.append(train_loss)
        total_val_loss.append(val_loss)
        total_psnr_metric.append(avg_psnr_metric)
        total_ssim_metric.append(avg_ssim_metric)

        if epoch > 3:
            if train_loss >= (sum(total_train_loss[-3:]) / 3):
                learning_rate /= 10
                update_lr(optimizer, learning_rate)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/super-resolution_ep{epoch + 1}.pth")

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
