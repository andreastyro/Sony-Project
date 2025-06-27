import torch
import torch.nn as nn

from torch.utils.data import DataLoader

class Block(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

class UNet(nn.Module):

    def __init__(self, frames, action_dim):
        super(UNet, self).__init__()

        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU()
        )

        # Contracting path
        self.enc1 = Block(6, 64)
        self.enc2 = Block(64, 128)
        self.enc3 = Block(128, 256)
        self.enc4 = Block(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)  # shared 2×2 pool

        # Bottleneck
        self.bottleneck = Block(512, 1024)

        # Expansive path

        self.transpose1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
        self.dec1 = Block(1024, 512)
        
        self.transpose2 = nn.ConvTranspose2d(512,  256,  kernel_size=2, stride=2)
        self.dec2 = Block(512, 256)

        self.transpose3  = nn.ConvTranspose2d(256, 128,  kernel_size=2, stride=2)
        self.dec3 = Block(256, 128)

        self.transpose4  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = Block(128,  64)

        self.out = nn.Conv2d(64, 3*frames, kernel_size=1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, frames, actions):

        # Encoder
        s1 = self.enc1(frames)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        b  = self.bottleneck(self.pool(s4))

        action_features = self.action_mlp(actions)

        action_features = action_features.unsqueeze(-1).unsqueeze(-1)

        action_features = action_features.expand(-1, -1, b.size(2), b.size(3))

        b = torch.cat([b, action_features], dim=1)

        # Decoder + Skip Connections
        up1 = self.transpose1(b)

        d1 = self.dec1(torch.cat([up1, s4], dim=1))

        up2 = self.transpose2(d1)
        d2 = self.dec2(torch.cat([up2, s3], dim=1))

        up3 = self.transpose3(d2)
        d3 = self.dec3(torch.cat([up3, s2], dim=1))

        up4 = self.transpose4(d3)
        d4 = self.dec4(torch.cat([up4, s1], dim=1))

        x = self.out(d4)

        x = self.sigmoid(x)

        return x