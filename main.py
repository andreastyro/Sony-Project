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

astro = Astro()

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

model = UNet().to(device)

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

    for x_train, y_train in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        optimizer.zero_grad()

        y_pred = model(x_train)

        adv_loss = criterion(y_train, y_pred)
        vgg_loss = perceptual_loss(y_train, y_pred)
        
        loss = adv_loss + 0.1 * vgg_loss

        loss.backward()
        optimizer.step()

        train_loss += loss

    train_losses.append(train_loss)

    print(train_loss/len(train_loader))

    with torch.no_grad():
        
        model.eval()

        for x_val, y_val in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):

            x_val = x_val.to(device)
            y_val = y_val.to(device)

            y_pred = model(x_val)

            adv_loss = criterion(y_val, y_pred)
            vgg_loss = perceptual_loss(y_val, y_pred)

            loss = adv_loss + 0.1 * vgg_loss

            val_loss += loss

        val_losses.append(val_loss)

    print(val_loss/len(val_loader))

    output = y_pred[0].detach().cpu().clamp(0, 1)
    img = to_pil(output)
    img.save(f"frame{epoch+1}.png")

    input_pair = x_val[0].cpu()
    f1 = input_pair[:3]
    f3 = input_pair[3:]

    f2_gt = y_val[0].cpu()
    f2_pred = y_pred[0].detach().cpu().clamp(0, 1)

    # Convert all to PIL images
    f1_img = to_pil(f1)
    f2_gt_img = to_pil(f2_gt)
    f3_img = to_pil(f3)
    f2_pred_img = to_pil(f2_pred)

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

    # Top row: ground truth
    axs[0, 0].imshow(f1_img)
    axs[0, 0].set_title("Frame 1")
    axs[0, 1].imshow(f2_gt_img)
    axs[0, 1].set_title("Ground Truth (Frame 2)")
    axs[0, 2].imshow(f3_img)
    axs[0, 2].set_title("Frame 3")

    # Bottom row: predicted
    axs[1, 0].imshow(f1_img)
    axs[1, 0].set_title("Frame 1")
    axs[1, 1].imshow(f2_pred_img)
    axs[1, 1].set_title("Predicted Frame 2")
    axs[1, 2].imshow(f3_img)
    axs[1, 2].set_title("Frame 3")

    # Clean up axes
    for ax_row in axs:
        for ax in ax_row:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("comparison_grid.png")
    plt.show()
    