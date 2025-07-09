import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd

class Astro_Multi(Dataset):
    def __init__(self, frame_gap, root_dir=None):

        if root_dir==None:
            self.root_dir = Path(__file__).resolve().parent.parent / "Data" / "extracted_frames"

        else:
            self.root_dir = Path(root_dir)

        self.frame_gap = frame_gap
        
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

            self.frames = sorted(glob.glob(f"{frames_dir}/*.png")) # fix this sort, based on number

            self.actions_csv = frames_dir / "actions.csv"

            df = pd.read_csv(self.actions_csv)
            self.df = self.process_actions(df)

            self.action_dimensions = len(self.df.columns)

            num_frames = len(self.frames)

            num_windows = ((num_frames - self.frame_gap) // self.frame_gap) + 1

            self.cumulative_windows += num_windows

            self.lookup_table[self.cumulative_windows] = {
                'trajectory_id': trajectory_id,
                'frames': self.frames,
                'actions_df': self.df,
                'num_frames': num_frames,
                'num_windows': num_windows,
                'starting_index': starting_index
            }

            starting_index = self.cumulative_windows

    def process_actions(self, df):

        df["FRAME"] = (df["TIMESTAMP"] // (1000/60))

        # Insert rows before first action

        first_row = df.iloc[[0]]
        first_action = int(df["FRAME"].iloc[0])

        first_row.loc[:, ["LSTICKX", "LSTICKY", "RSTICKX", "RSTICKY"]] = 127

        rows_inserted = []

        for i in range(1, first_action):

            custom_row = first_row.copy()

            custom_row["FRAME"] = i
            rows_inserted.append(custom_row)

        # Insert Rows in between action gaps
        
        for i in range(len(df) - 1):

            current_row = df.iloc[[i]].copy()

            current_action = int(current_row["FRAME"].values[0])
            next_action = int(df["FRAME"].iloc[i+1])

            if current_action == next_action:
                df.iloc[i+1, df.columns.get_loc("FRAME")] = next_action + 1 # Eliminates Rounding duplication of frame numbers

            difference = next_action - current_action

            if difference > 1:

                for j in range(1, difference):

                    new_row = current_row.copy()

                    new_row["FRAME"] = new_row["FRAME"] + j

                    rows_inserted.append(new_row)

        last_row = df.iloc[[-1]]
        last_action = int(df["FRAME"].iloc[-1])

        difference = len(self.frames) - last_action

        for i in range(1, difference+1):
            
            new_row = last_row.copy()
            new_row["FRAME"] = new_row["FRAME"] + i

            rows_inserted.append(new_row)

        df = pd.concat([df] + rows_inserted, ignore_index=True)

        df = df.set_index("FRAME")
        df = df.sort_index()
        df = df.reset_index()

        return df
    
    def __len__(self):

        return self.cumulative_windows

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {len(self)}")

        lookup_key = None
        for i in self.lookup_table.keys():
            if i > idx:
                lookup_key = i
                
                break
        
        lookup_values = self.lookup_table[lookup_key]

        trajectory_id = lookup_values['trajectory_id']
        frames = lookup_values['frames']
        actions_df = lookup_values['actions_df']
        starting_index = lookup_values['starting_index']
    
        local_idx = idx - starting_index

        start_frame_idx = local_idx * self.frame_gap

        frame_list = []
        action_list = []
        
        first_frame = cv2.cvtColor(cv2.imread(frames[start_frame_idx]), cv2.COLOR_BGR2RGB)
        first_action = self.df.iloc[start_frame_idx]

        first_frame = torch.tensor(first_frame / 255.0, dtype=torch.float32).permute(2, 0, 1)
        first_action = torch.tensor(first_action.values, dtype=torch.float32)

        for frame in range(1, self.frame_gap):

            new_frame = cv2.cvtColor(cv2.imread(frames[start_frame_idx + frame]), cv2.COLOR_BGR2RGB)
            new_action = actions_df.iloc[start_frame_idx + frame]

            new_frame = torch.tensor(new_frame / 255.0, dtype=torch.float32).permute(2, 0, 1)
            new_action = torch.tensor(new_action.values, dtype=torch.float32)

            frame_list.append(new_frame)
            action_list.append(new_action)

        last_frame = cv2.cvtColor(cv2.imread(frames[start_frame_idx + self.frame_gap]), cv2.COLOR_BGR2RGB)
        last_action = self.df.iloc[start_frame_idx + self.frame_gap]

        last_frame = torch.tensor(last_frame / 255.0, dtype=torch.float32).permute(2, 0, 1)
        last_action = torch.tensor(last_action.values, dtype=torch.float32)

        frame_pair = torch.cat([first_frame, last_frame], dim=0)
        action_pair = torch.cat([first_action, last_action])

        frame_list = torch.cat(frame_list, dim=0)
        action_list = torch.cat(action_list, dim=0)

        return (frame_pair, action_pair), (frame_list, action_list)
    
class Astro(Dataset):
    def __init__(self, frame_gap, frame_dir=None, action_csv=None):
        self.frame_gap = frame_gap

        if frame_dir is None:
            root_dir = Path(__file__).resolve().parent.parent
            frame_dir = root_dir / "Data" / "extracted_frames"

        self.frames = sorted(glob.glob(f"{frame_dir}/*.png"))

        if action_csv is None:
            root_dir = Path(__file__).resolve().parent.parent
            action_csv = root_dir / "Data" / "actions.csv"

        df = pd.read_csv(action_csv)
    
        df["FRAME"] = (df["TIMESTAMP"] // (1000/60))

        # Insert rows before first action

        first_row = df.iloc[[0]]
        first_action = int(df["FRAME"].iloc[0])

        first_row.loc[:, ["LSTICKX", "LSTICKY", "RSTICKX", "RSTICKY"]] = 127

        rows_inserted = []

        for i in range(1, first_action):

            custom_row = first_row.copy()

            custom_row["FRAME"] = i
            rows_inserted.append(custom_row)

        # Insert Rows in between action gaps
        
        for i in range(len(df) - 1):

            current_row = df.iloc[[i]].copy()

            current_action = int(current_row["FRAME"].values[0])
            next_action = int(df["FRAME"].iloc[i+1])

            if current_action == next_action:
                df.iloc[i+1, df.columns.get_loc("FRAME")] = next_action + 1 # Eliminates Rounding duplication of frame numbers

            difference = next_action - current_action

            if difference > 1:

                for j in range(1, difference):

                    new_row = current_row.copy()

                    new_row["FRAME"] = new_row["FRAME"] + j

                    rows_inserted.append(new_row)

        last_row = df.iloc[[-1]]
        last_action = int(df["FRAME"].iloc[-1])

        difference = len(self.frames) - last_action

        for i in range(1, difference+1):

            new_row = last_row.copy()
            new_row["FRAME"] = new_row["FRAME"] + j

            rows_inserted.append(new_row)

        df = pd.concat([df] + rows_inserted, ignore_index=True)
        
        df = df.set_index("FRAME")
        df = df.sort_index()
        df = df.reset_index()

        self.df = df

        self.action_dimensions  = len(self.df.columns) # feature count
        
    def __len__(self):
        
        return max(0, len(self.frames) - self.frame_gap)

    def __getitem__(self, idx):

        frame_list = []
        action_list = []

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}")
        
        first_frame = cv2.cvtColor(cv2.imread(self.frames[idx]), cv2.COLOR_BGR2RGB)
        first_action = self.df.iloc[idx]

        # Normalize and reshape to CxHxW
        first_frame = torch.tensor(first_frame / 255.0, dtype=torch.float32).permute(2, 0, 1)
        first_action = torch.tensor(first_action, dtype=torch.float32)

        for frame in range(1, self.frame_gap):
            new_frame = cv2.cvtColor(cv2.imread(self.frames[idx + frame]), cv2.COLOR_BGR2RGB)
            new_action = self.df.iloc[idx + frame]
            
            new_frame = torch.tensor(new_frame / 255.0, dtype=torch.float32).permute(2, 0, 1)
            new_action = torch.tensor(new_action, dtype=torch.float32)

            frame_list.append(new_frame)
            action_list.append(new_action)

        last_frame = cv2.cvtColor(cv2.imread(self.frames[idx + self.frame_gap]), cv2.COLOR_BGR2RGB)
        last_action = self.df.iloc[idx + self.frame_gap]

        last_frame = torch.tensor(last_frame / 255.0, dtype=torch.float32).permute(2, 0, 1)
        last_action = torch.tensor(last_action, dtype=torch.float32)

        frame_pair = torch.cat([first_frame, last_frame], dim=0)  # shape (6, H, W)
        action_pair = torch.cat([first_action, last_action])

        frame_list = torch.cat(frame_list, dim=0)
        action_list = torch.cat(action_list, dim=0)

        return (frame_pair, action_pair), (frame_list, action_list)  # input, target