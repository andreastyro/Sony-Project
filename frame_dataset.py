import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

class Astro_Multi(Dataset):
    def __init__(self, frame_gap, mode, root_dir=None):

        if root_dir==None:
            self.root_dir = Path(__file__).resolve().parent.parent / "Data" / "extracted_frames"

        else:
            self.root_dir = Path(root_dir)

        self.frame_gap = frame_gap
        self.mode = mode
        
        self.trajectory_dir = [d for d in self.root_dir.iterdir() if d.is_dir()]
        self.trajectory_ids = [d.name for d in self.trajectory_dir]

        self.lookup_table = {}
        self.build_lookup_table()

    def build_lookup_table(self):

        self.lookup_table = {}

        starting_index = 0
        self.cumulative_windows = 0

        for trajectory_id in self.trajectory_ids:

            frames_dir = self.root_dir / trajectory_id

            frames = sorted(glob.glob(f"{frames_dir}/*.jpg")) # Check file type

            self.actions_csv = frames_dir / "actions.csv"

            df = pd.read_csv(self.actions_csv, engine='python', on_bad_lines='skip')

            num_frames = len(frames)

            self.df = self.process_actions(df, num_frames)

            self.action_dimensions = len(self.df.columns)

            window_len = self.frame_gap + 1
            max_start = num_frames - window_len
            num_windows = max_start // self.frame_gap + 1
            end_index = starting_index + num_windows - 1   # inclusive upper bound

            self.cumulative_windows += num_windows

            self.lookup_table[end_index] = {
                'trajectory_id': trajectory_id,
                'frames': frames.copy(),
                'actions_df': self.df,
                'num_frames': num_frames,
                'num_windows': num_windows,
                'starting_index': starting_index
            }

            starting_index = end_index + 1                 # next traj starts here
            self.cumulative_windows = end_index + 1        # dataset length so far
            
    def process_actions(self, df, num_frames):

        df["FRAME"] = (df["TIMESTAMP"] // (1000/60))

        neutral_cols = ["LSTICKX", "LSTICKY", "RSTICKX", "RSTICKY"]

        df = (
            df.sort_values("FRAME")             # order by frame
            .drop_duplicates("FRAME", keep="first")
            .set_index("FRAME")                 # FRAME becomes the index
            .reindex(range(1, num_frames+1))    # create every row 0 … num_frames-1
            .ffill()                            # forward-fill missing rows
            .reset_index()                      # move index back to column "FRAME"
        )

        df[neutral_cols] = df[neutral_cols].fillna(127)
        df = df.fillna(0)
        cols_to_remove = ["FRAME", "VFRAME_ID", "TIMESTAMP", "TOUCHPOINT0_ID", "TOUCHPOINT0_X", "TOUCHPOINT0_y","TOUCHPOINT1_ID","TOUCHPOINT1_X", "TOUCHPOINT1_y"]
        df = df.drop(columns=cols_to_remove)

        return df
        
    def __len__(self):

        return self.cumulative_windows

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self)}")

        lookup_key = None
        for i in self.lookup_table.keys():
            if i >= idx:
                lookup_key = i
                
                break
        
        lookup_values = self.lookup_table[lookup_key]

        trajectory_id = lookup_values['trajectory_id']
        frames = lookup_values['frames']
        actions_df = lookup_values['actions_df']
        starting_index = lookup_values['starting_index']
    
        local_idx = idx - starting_index

        start_frame_idx = local_idx * self.frame_gap

        #print("Start Frame Index: ", start_frame_idx, "Local Index: ", local_idx, "Starting Index: ", starting_index, "Frames: ", len(frames), "Actions: \n", actions_df)
        
        frame_list = []
        action_list = []

        def _imread_rgb(path: str):
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        first_frame = _imread_rgb(frames[start_frame_idx])
        first_frame = torch.tensor(first_frame / 255.0, dtype=torch.float32).permute(2, 0, 1)

        frame_list = [
            torch.tensor(
                _imread_rgb(frames[start_frame_idx + i]) / 255.0, 
                dtype=torch.float32
                ).permute(2, 0, 1) 
                for i in range(1, self.frame_gap)
        ]

        if self.mode == "interpolate":

            last_frame = _imread_rgb(frames[start_frame_idx + self.frame_gap])
            last_frame = torch.tensor(last_frame / 255.0, dtype=torch.float32).permute(2, 0, 1)

            frame_pair = torch.cat([first_frame, last_frame], dim=0)

        frame_list = torch.stack(frame_list, dim=0)

        action_list = [
            torch.tensor(actions_df.iloc[i].values, dtype=torch.float32) 
                for i in range(start_frame_idx, start_frame_idx + self.frame_gap + 1)
        ]
        
        action_list = torch.cat(action_list, dim=0)

        if self.mode == "interpolate":

            return (frame_pair, action_list), frame_list
        
        else:
            
            return (first_frame, action_list), frame_list
