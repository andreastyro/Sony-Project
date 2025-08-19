import os
from pathlib import Path
from torch.utils.data import Subset, DataLoader, random_split
from frame_dataset import Frame_Dataset
from torch.cuda.amp import GradScaler, autocast
from unet import UNet
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, models
from torchvision.transforms import ToPILImage
from torchvision.models import vgg19
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

checkpoint_dir = Path(__file__).resolve().parent.parent / "Checkpoints"
checkpoint_file = checkpoint_dir / "latest.pt"
os.makedirs(checkpoint_dir, exist_ok=True)

root_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(root_dir, "Plots")
os.makedirs(save_dir, exist_ok=True)


to_pil = ToPILImage()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------- Split Dataset -----------------------------------

mode = "predict"
game = "tlou"


frame_gap = 5
frames = frame_gap - 1

dataset = Frame_Dataset(frame_gap=frame_gap, mode=mode, game=game)

action_dim = dataset.action_dimensions

total_len  = len(dataset)
train_len  = int(0.70 * total_len)          # 70 %
val_len    = int(0.15 * total_len)          # 15 %
test_len   = total_len - train_len - val_len  # remaining 15 %

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# ----------------------------------------------------------------------------------

learning_rate = 1e-4

model = UNet(frames=frames, action_dim=action_dim * (frame_gap+1), mode=mode, stochastic=True, noise_sigma=0.1).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

start_epoch = 0
epochs = 10

avg_train_loss = 0
avg_val_loss = 0

train_loss = 0
val_loss = 0

train_losses = []
val_losses = []

train_psnr = []
val_psnr = []

train_ssim = []
val_ssim = []

checkpoint = False

"""

if checkpoint_file.exists():
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    #print(checkpoint)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from epoch {start_epoch}")
"""
for epoch in range(start_epoch, epochs):

    train_loss = 0
    val_loss = 0

    psnr_value = 0
    ssim_value = 0

    model.train()

    for (x_train_frames, x_train_actions), y_train_frames in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        #print("y_train shape:", y_train.shape)

        #B, T, C, H, W = y_train.shape
        #y_train = y_train.view(B * T, C, H, W) # Reduce 5 dimensions to 4

        x_train_frames = x_train_frames.to(device)
        x_train_actions = x_train_actions.to(device)
        y_train_frames = y_train_frames.to(device)

        optimizer.zero_grad()

        y_pred = model(x_train_frames, x_train_actions)

        B, C, H, W = y_pred.shape
        T = C // 3

        y_train_frames = y_train_frames.view(B, T, 3, H, W).reshape(B * T, 3, H, W)
        y_pred  = y_pred.view(B, T, 3, H, W).reshape(B * T, 3, H, W)

        adv_loss = criterion(y_train_frames, y_pred)

        loss = adv_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        psnr_score = psnr(y_pred, y_train_frames)
        psnr_value += psnr_score.item()

        ssim_score = ssim(y_pred, y_train_frames)
        ssim_value += ssim_score.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_psnr = psnr_value / len(train_loader)
    avg_train_ssim = ssim_value / len(train_loader)
    
    train_losses.append(avg_train_loss)
    train_psnr.append(avg_train_psnr)
    train_ssim.append(avg_train_ssim)

    print(f"Train Epoch [{epoch+1}/{epochs}] | MSE Loss: ", avg_train_loss, " | PSNR: ", avg_train_psnr,  " | SSIM: ", avg_train_ssim)

    psnr_value = 0
    ssim_value = 0

    with torch.no_grad():
        
        model.eval()

        for (x_val_frames, x_val_actions), y_val_frames in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):


            x_val_frames = x_val_frames.to(device)
            x_val_actions = x_val_actions.to(device)
            y_val_frames = y_val_frames.to(device)

            #print(y_val.shape)
            #print(y_pred.shape)

            y_pred = model(x_val_frames, x_val_actions)

            B, C, H, W = y_pred.shape
            T = C // 3

            y_val_seq = y_val_frames.view(B, T, 3, H, W)
            y_pred_seq = y_pred.view(B, T, 3, H, W)

            y_val_frames = y_val_seq.reshape(B * T, 3, H, W)
            y_pred  = y_pred_seq.reshape(B * T, 3, H, W)

            adv_loss = criterion(y_val_frames, y_pred)

            loss = adv_loss

            val_loss += loss.item()

            psnr_score = psnr(y_pred, y_val_frames)
            psnr_value += psnr_score.item()

            ssim_score = ssim(y_pred, y_val_frames)
            ssim_value += ssim_score.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_psnr = psnr_value / len(val_loader)
    avg_val_ssim = ssim_value / len(val_loader)

    val_losses.append(avg_val_loss)
    val_psnr.append(avg_val_psnr)
    val_ssim.append(avg_val_ssim)

    print(f"Validation Epoch [{epoch+1}/{epochs}] | MSE Loss: ", avg_val_loss, " | PSNR: ", avg_val_psnr,  " | SSIM: ", avg_val_ssim)

    if checkpoint == True:

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "train_loss": avg_train_loss, "val_loss": avg_val_loss}, checkpoint_file)

    psnr_value = 0
    ssim_value = 0

    for (x_test_frames, x_test_actions), y_test_frames in tqdm(test_loader, desc="Testing"):

        x_test_frames = x_test_frames.to(device)
        x_test_actions = x_test_actions.to(device)
        y_test_frames = y_test_frames.to(device)

        y_pred = model(x_test_frames, x_test_actions)

        B, C, H, W = y_pred.shape
        T = C // 3

        y_test_seq = y_test_frames.view(B, T, 3, H, W)
        y_pred_seq = y_pred.view(B, T, 3, H, W)

        y_test_frames = y_test_seq.reshape(B * T, 3, H, W)
        y_pred = y_pred_seq.reshape(B * T, 3, H, W)

        adv_loss = criterion(y_test_frames, y_pred)

        loss = adv_loss

        test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)

    print(f"Test Loss: ", avg_test_loss)

    # ---------------------- Plots, Graphs, Curves ---------------------

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Loss subplot
    axes[0].plot(range(1, epochs + 1), train_losses, label="Train Loss")
    axes[0].plot(range(1, epochs + 1), val_losses,   label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].legend(loc="upper right")

    # Perplexity subplot
    axes[1].plot(range(1, epochs + 1), train_psnr, label="Train PSNR")
    axes[1].plot(range(1, epochs + 1), val_psnr,   label="Val PSNR")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR")
    axes[1].set_title("Training vs Validation PSNR")
    axes[1].legend()
    axes[1].legend(loc="lower right")

    axes[2].plot(range(1, epochs + 1), train_ssim, label="Train SSIM")
    axes[2].plot(range(1, epochs + 1), val_ssim,   label="Val SSIM")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("Training vs Validation SSIM")
    axes[2].legend()
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Plots_LR_{learning_rate}_{mode}_{game}.png"))

    # ----------------- plot full sequence grid for first sample -------
    fig_seq, axs_seq = plt.subplots(2, T, figsize=(3*T, 6), squeeze=False)

    for t in range(T):
        gt   = y_val_seq[0, t].detach().clamp(0, 1).cpu()
        pred = y_pred_seq[0, t].detach().clamp(0, 1).cpu()

        axs_seq[0, t].imshow(TF.to_pil_image(gt))
        axs_seq[0, t].set_title(f"GT t{t+1}")

        axs_seq[1, t].imshow(TF.to_pil_image(pred))
        axs_seq[1, t].set_title(f"Pred t{t+1}")

        axs_seq[0, t].axis("off")
        axs_seq[1, t].axis("off")
        
    plt.tight_layout()
    plt.savefig(f"sequence_grid_{epoch}_{learning_rate}_{mode}_{game}.png")
    plt.close(fig_seq)

