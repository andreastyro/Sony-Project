import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path


class Astro(Dataset):
    def __init__(self, frame_dir=None):
        if frame_dir is None:
            root_dir = Path(__file__).resolve().parent.parent
            frame_dir = root_dir / "Data" / "extracted_frames"

            print(frame_dir)

        self.frame_paths = sorted(glob.glob(f"{frame_dir}/*.png"))
        
    def __len__(self):
        
        return max(0, len(self.frame_paths) - 2)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
            
        f1 = cv2.imread(self.frame_paths[idx])
        f2 = cv2.imread(self.frame_paths[idx + 1])
        f3 = cv2.imread(self.frame_paths[idx + 2])

        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
        f3 = cv2.cvtColor(f3, cv2.COLOR_BGR2RGB)

        # Normalize and reshape to CxHxW
        f1 = torch.tensor(f1 / 255.0, dtype=torch.float32).permute(2, 0, 1)
        f2 = torch.tensor(f2 / 255.0, dtype=torch.float32).permute(2, 0, 1)
        f3 = torch.tensor(f3 / 255.0, dtype=torch.float32).permute(2, 0, 1)

        input_pair = torch.cat([f1, f3], dim=0)  # shape (6, H, W)
        return input_pair, f2  # input, target