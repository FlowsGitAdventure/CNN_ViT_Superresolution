import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from UNetSuperRes import UNet
from UNet.trash.FMRISuperResData import FMRISuperResDataset


def train():
    LR_DIR = './data/Low_Res_08'
    HR_DIR = './data/High_Res_08'

    # Config for CPU/Stability
    BATCH_SIZE = 2
    PATCH_SIZE = (32, 32, 32)  # Keeps memory low (~250MB per batch layer)
    BASE_FILTERS = 32  # Keeps model light
    NUM_EPOCHS = 20

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = './checkpoints'
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Training on: {DEVICE}")

    # Initialize Dataset
    train_dataset = FMRISuperResDataset(LR_DIR, HR_DIR, train=True, patch_size=PATCH_SIZE)
    # Validation also uses safe crop now to prevent crashes
    val_dataset = FMRISuperResDataset(LR_DIR, HR_DIR, train=False, patch_size=PATCH_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=1, out_channels=1, base_filters=BASE_FILTERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    print(f"Starting... (Batch={BATCH_SIZE}, Patch={PATCH_SIZE})")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)

            optimizer.zero_grad()
            out = model(lr)
            loss = criterion(out, hr)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} Avg Loss: {train_loss / len(train_loader):.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/unet_ep{epoch + 1}.pth")

    torch.save(model.state_dict(), f"{SAVE_DIR}/unet_final.pth")


if __name__ == "__main__":
    train()
