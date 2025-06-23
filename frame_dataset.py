import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

class Astro(Dataset):
    def __init__(self, frame_gap, frame_dir=None, action_csv=None):
        self.frame_gap = frame_gap

        if frame_dir is None:
            root_dir = Path(__file__).resolve().parent.parent
            frame_dir = root_dir / "Data" / "extracted_frames"

            print(frame_dir)

        self.frames = sorted(glob.glob(f"{frame_dir}/*.png"))

        if action_csv is None:
            root_dir = Path(__file__).resolve().parent.parent
            action_csv = root_dir / "Data" / "actions.csv"

        df = pd.read_csv(action_csv)
    
        df["FRAMES"] = self.df["TIMESTAMP"] // (1000/60)

        full = pd.DataFrame({"FRAMES": range(len(self.frames))})

        df = full.merge(df, on="FRAMES", how="left")

        self.df = df.fillna(0).astype(df.dtypes)
        
    def __len__(self):
        
        return max(0, len(self.frames) - self.frame_gap)

    def __getitem__(self, idx):

        frame_list = []

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
        
        first_frame = cv2.cvtColor(cv2.imread(self.frames[idx]), cv2.COLOR_BGR2RGB)
        first_action = self.df.iloc[idx]

        for frame in range(1, self.frame_gap):
            new_frame = cv2.cvtColor(cv2.imread(self.frames[idx + frame]), cv2.COLOR_BGR2RGB)
            new_action = self.df.iloc[idx + frame]

            frame_list.append(new_frame)

        last_frame = cv2.cvtCoslor(cv2.imread(self.frames[idx + self.frame_gap]), cv2.COLOR_BGR2RGB)
        last_action = self.df.iloc[idx + self.frame_gap]

        # Normalize and reshape to CxHxW
        first_frame = torch.tensor(first_frame / 255.0, dtype=torch.float32).permute(2, 0, 1)
        first_action = torch.tensor(first_action, dtype=torch.float32)

        frame_list = [torch.tensor(f / 255.0, dtype=torch.float32).permute(2, 0, 1) for f in frame_list]

        last_frame = torch.tensor(last_frame / 255.0, dtype=torch.float32).permute(2, 0, 1)
        last_action = torch.tensor(last_action, dtype=torch.float32)

        input_pair = torch.cat([first_frame, last_frame], dim=0)  # shape (6, H, W)

        frame_list = torch.cat(frame_list, dim=0)
        
        return input_pair, frame_list  # input, target