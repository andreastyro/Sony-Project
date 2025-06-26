import os
from pathlib import Path
from torch.utils.data import Subset, DataLoader, random_split
from frame_dataset import Astro
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


os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

to_pil = ToPILImage()

# Load VGG model
vgg = models.vgg19(pretrained=True).features
vgg.eval()  # Set to evaluation mode

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# Define VGG feature extractor
class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg, layers=3):  # Extract up to the first 7 layers
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = nn.Sequential(*list(vgg[:layers]))  # Extract layers

    def forward(self, x):
        return self.vgg(x)

# Initialize the VGG feature extractor
vgg_features = VGGFeatureExtractor(vgg, layers=7).to(device)

# Define VGG loss
class VGGPerceptualLoss(nn.Module): 
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device) # Source for values: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49?permalink_comment_id=4423554
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def forward(self, truth_frame, gen_frame):

        # Normalize input
        gen_frame = (gen_frame - self.mean) / self.std
        truth_frame = (truth_frame - self.mean) / self.std

        # Compute feature maps using VGG
        gen_features = vgg_features(gen_frame)
        true_features = vgg_features(truth_frame)

        # Compute loss
        loss = self.criterion(gen_features, true_features)
        return loss

# -------------------------------- Split Dataset -----------------------------------

frame_gap = 21
frames = frame_gap - 1

astro = Astro(frame_gap=frame_gap)

action_dim = astro.action_dimensions

total_len  = len(astro)
train_len  = int(0.70 * total_len)          # 70 %
val_len    = int(0.15 * total_len)          # 15 %
test_len   = total_len - train_len - val_len  # remaining 15 %

train_dataset, val_dataset, test_dataset = random_split(astro, [train_len, val_len, test_len])

print(len(train_dataset), len(val_dataset), len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# ----------------------------------------------------------------------------------

learning_rate = 1e-4

model = UNet(frames=frames, action_dim=action_dim * 2).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
perceptual_loss = VGGPerceptualLoss().to(device)

epochs = 25

train_loss = 0
val_loss = 0

train_losses = []
val_losses = []

for epoch in range(epochs):

    train_loss = 0
    val_loss = 0

    model.train()

    for (x_train_frames, x_train_actions), (y_train_frames, y_train_actions) in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        #print("y_train shape:", y_train.shape)

        #B, T, C, H, W = y_train.shape
        #y_train = y_train.view(B * T, C, H, W) # Reduce 5 dimensions to 4

        x_train_frames = x_train_frames.to(device)
        x_train_actions = x_train_actions.to(device)
        y_train_frames = y_train_frames.to(device)
        y_train_actions = y_train_actions.to(device)

        optimizer.zero_grad()

        y_pred = model(x_train_frames, x_train_actions)

        B, C, H, W = y_pred.shape
        T = C // 3

        y_train_frames = y_train_frames.view(B, T, 3, H, W).reshape(B * T, 3, H, W)
        y_pred  = y_pred.view(B, T, 3, H, W).reshape(B * T, 3, H, W)

        adv_loss = criterion(y_train_frames, y_pred)
        vgg_loss = perceptual_loss(y_train_frames, y_pred)
        
        loss = adv_loss + 0.1 * vgg_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_losses.append(train_loss)

    print(train_loss/len(train_loader))

    with torch.no_grad():
        
        model.eval()

        for (x_val_frames, x_val_actions), (y_val_frames, y_val_actions) in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):

            x_val_frames = x_val_frames.to(device)
            x_val_actions = x_val_actions.to(device)
            y_val_frames = y_val_frames.to(device)
            y_val_actions = y_val_actions.to(device)

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
            vgg_loss = perceptual_loss(y_val_frames, y_pred)

            loss = adv_loss + 0.1 * vgg_loss

            val_loss += loss.item()

        val_losses.append(val_loss)

    print(val_loss/len(val_loader))

    # ----------------- save a middle predicted frame ------------------
    mid_idx = T // 2                         # middle frame index
    middle_pred = y_pred_seq[0, mid_idx].detach().cpu().clamp(0, 1)  # (3,H,W)
    TF.to_pil_image(middle_pred).save(f"frame{epoch+1}.png")

    # ----------------- plot full sequence grid for first sample -------
    fig_seq, axs_seq = plt.subplots(2, T, figsize=(3*T, 6), squeeze=False)

    for t in range(T):
        gt   = y_val_seq[0, t].detach().cpu().clamp(0, 1)
        pred = y_pred_seq[0, t].detach().cpu().clamp(0, 1)

        axs_seq[0, t].imshow(TF.to_pil_image(gt))
        axs_seq[0, t].set_title(f"GT t{t+1}")

        axs_seq[1, t].imshow(TF.to_pil_image(pred))
        axs_seq[1, t].set_title(f"Pred t{t+1}")

        axs_seq[0, t].axis("off")
        axs_seq[1, t].axis("off")
        
    plt.tight_layout()
    plt.savefig("sequence_grid.png")
    plt.close(fig_seq)

    # ----------------- plot first / middle / last comparison ----------
    input_pair = x_val_frames[0].cpu()
    f1 = input_pair[:3]                     # first frame (3,H,W)
    f3 = input_pair[3:]                     # last  frame (3,H,W)

    gt_mid   = y_val_seq [0, mid_idx].cpu()
    pred_mid = y_pred_seq[0, mid_idx].cpu()

    fig_cmp, axs_cmp = plt.subplots(2, 3, figsize=(10, 6))

    # top row: ground truth frames
    axs_cmp[0,0].imshow(TF.to_pil_image(f1));      axs_cmp[0,0].set_title("Frame 1")
    axs_cmp[0,1].imshow(TF.to_pil_image(gt_mid));  axs_cmp[0,1].set_title("GT mid")
    axs_cmp[0,2].imshow(TF.to_pil_image(f3));      axs_cmp[0,2].set_title("Frame 3")

    # bottom row: predicted mid frame
    axs_cmp[1,0].imshow(TF.to_pil_image(f1));      axs_cmp[1,0].set_title("Frame 1")
    axs_cmp[1,1].imshow(TF.to_pil_image(pred_mid));axs_cmp[1,1].set_title("Pred mid")
    axs_cmp[1,2].imshow(TF.to_pil_image(f3));      axs_cmp[1,2].set_title("Frame 3")

    for ax in axs_cmp.flatten(): ax.axis("off")
    plt.tight_layout()
    plt.savefig("comparison_grid.png")
    plt.close(fig_cmp)