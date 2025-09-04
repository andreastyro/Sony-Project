import os
import random
from pathlib import Path
from torch.utils.data import Subset, DataLoader, random_split
from frame_dataset import Frame_Dataset
from torch.cuda.amp import GradScaler, autocast
from unet import UNet
from unet_no_actions import UNet_Free
from unet_lite import UNet_Lite
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

checkpoint_dir = Path("/scratch/uceeaty/Checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

root_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(root_dir, "Plots")
log_dir = os.path.join(root_dir, "Final Results")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

to_pil = ToPILImage()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stochastic = False
mode_type = "stochastic" if stochastic else "deterministic"
mode = "predict"
game = "astro"
epoch = 4

lr_test = False

frame_gap = 5
frames = frame_gap - 1

dataset = Frame_Dataset(frame_gap=frame_gap, mode=mode, game=game)

action_dim = dataset.action_dimensions

total_len  = len(dataset)
train_len  = int(0.70 * total_len)          # 70 %
val_len    = int(0.15 * total_len)          # 15 %
test_len   = total_len - train_len - val_len  # remaining 15 %

generator = torch.Generator().manual_seed(42)

dataset_loader = DataLoader(dataset, batch_size=2, shuffle=False)
# ----------------------------------------------------------------------------------

learning_rate = 3e-4

model = UNet_Free(frames=frames, action_dim=action_dim * (frame_gap+1), mode=mode, stochastic=stochastic, noise_sigma=0.1).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
#vgg_loss = VGGPerceptualLoss().to(device)
psnr = PeakSignalNoiseRatio().to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

checkpoint = True
checkpoint_file = checkpoint_dir / f"checkpoint_LR_{learning_rate}_{mode}_{game}_{mode_type}_action_free.pt"

if checkpoint_file.exists():
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    #print(checkpoint)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resumed from epoch {start_epoch}")


def get_batch(data_loader, batch_idx):
    dataset = data_loader.dataset
    batch_size = data_loader.batch_size

    start = batch_idx * batch_size
    end = start + batch_size

    if isinstance(dataset, torch.utils.data.Subset):
        # dataset.indices points to items in the underlying dataset
        indices = dataset.indices[start:end]
        samples = [dataset.dataset[i] for i in indices]
    else:
        # dataset is a plain Dataset (like Frame_Dataset)
        samples = [dataset[i] for i in range(start, end)]

    return data_loader.collate_fn(samples)

with torch.no_grad():
    
    model.eval()

    (x_val_frames, x_val_actions), y_val_frames = get_batch(dataset_loader, 150)

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
    plt.savefig(f"inference_{mode}_{game}_{mode_type}_action_free.png")
    plt.close(fig_seq)

