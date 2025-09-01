# src/datasets/myset.py
import os, glob, cv2, torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

def imread_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_tensor(img):
    # HWC uint8 [0..255] -> CHW float32 [0..1]
    return torch.from_numpy(img).permute(2,0,1).float().div_(255.0)

class EDEN(Dataset):
    """
    Returns a dict with:
      - 'start':   CxHxW (frame t)
      - 'gt':      CxHxW (middle frame)
      - 'end':     CxHxW (frame t+gap)
    EDEN resizes inside its pipeline; if you prefer, you can resize here to (height,width).
    """
    def __init__(self, data_dir, split_name="train", dur_list=[3], height=None, width=None):
        self.data_dir = Path(data_dir) / split_name
        self.height, self.width = height, width
        self.dur_list = sorted(dur_list)  # e.g., [3,5,7]
        # gather per-video frame lists
        self.videos = []
        for vid in sorted([d for d in self.data_dir.iterdir() if d.is_dir()]):
            frames = sorted(glob.glob(str(vid / "*.jpg")))
            if len(frames) >= max(self.dur_list):
                self.videos.append(frames)
        # build (video_idx, start_idx, dur) tuples for uniform sampling
        self.samples = []
        for v_idx, frames in enumerate(self.videos):
            n = len(frames)
            for dur in self.dur_list:
                gap = dur - 1
                max_start = n - dur
                if max_start < 0: continue
                for s in range(0, max_start + 1, gap):  # non-overlapping windows by default
                    self.samples.append((v_idx, s, dur))

    def __len__(self):
        return len(self.samples)

    def _resize(self, img):
        if self.height is None or self.width is None:
            return img
        return cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def __getitem__(self, idx):
        v_idx, s, dur = self.samples[idx]
        frames = self.videos[v_idx]
        gap = dur - 1
        mid = s + gap // 2

        I0 = self._resize(imread_rgb(frames[s]))
        I1 = self._resize(imread_rgb(frames[s + gap]))
        GT = self._resize(imread_rgb(frames[mid]))

        out = {
            "start": to_tensor(I0),
            "gt":    to_tensor(GT),
            "end":   to_tensor(I1),
        }
        return out
