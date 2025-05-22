import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from frame_dataset import Astro
from unet import UNet
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

to_pil = ToPILImage()

# -------------------------------- Split Dataset -----------------------------------

astro = Astro()

indices = list(range(len(astro)))

train_indices, temp = train_test_split(indices, test_size=0.3, random_state=42, shuffle=True)

val_indices, test_indices = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)

train_dataset = Subset(astro, train_indices)
val_dataset = Subset(astro, val_indices)
test_dataset = Subset(astro, test_indices)

print(len(train_dataset), len(val_dataset), len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# ----------------------------------------------------------------------------------

learning_rate = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

epochs = 25

train_loss = 0
val_loss = 0

train_losses = []

for epoch in range(epochs):

    model.train()

    for x_train, y_train in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        optimizer.zero_grad()

        y_pred = model(x_train)
        loss = criterion(y_train, y_pred)
        
        loss.backward()
        optimizer.step()

        train_loss += loss

    train_losses.append(train_loss)

    print(train_loss/len(train_loader))

    output = y_pred[0].detach().cpu().clamp(0, 1)
    img = to_pil(output)
    img.save(f"frame{epoch+1}.png")

    input_pair = x_train[0].cpu()
    f1 = input_pair[:3]
    f3 = input_pair[3:]

    f2_gt = y_train[0].cpu()
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