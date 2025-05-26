import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path

class Astro(Dataset):
    def __init__(self, frame_gap, frame_dir=None):
        self.frame_gap = frame_gap

        if frame_dir is None:
            root_dir = Path(__file__).resolve().parent.parent
            frame_dir = root_dir / "Data" / "extracted_frames"

            print(frame_dir)

        self.frame_paths = sorted(glob.glob(f"{frame_dir}/*.png"))
        
    def __len__(self):
        
        return max(0, len(self.frame_paths) - self.frame_gap)

    def __getitem__(self, idx):

        frame_list = []

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
        
        first = cv2.imread(self.frame_paths[idx])
        first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)

        for frame in range(1, self.frame_gap):
            new_frame = cv2.imread(self.frame_paths[idx + frame])
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            frame_list.append(new_frame)

        final = cv2.imread(self.frame_paths[idx + self.frame_gap])
        final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

        # Normalize and reshape to CxHxW
        first = torch.tensor(first / 255.0, dtype=torch.float32).permute(2, 0, 1)
        frame_list = [torch.tensor(f / 255.0, dtype=torch.float32).permute(2, 0, 1) for f in frame_list]
        final = torch.tensor(final / 255.0, dtype=torch.float32).permute(2, 0, 1)

        input_pair = torch.cat([first, final], dim=0)  # shape (6, H, W)
        frame_list = torch.cat(frame_list, dim=0)
        
        return input_pair, frame_list  # input, target