import xarray as xr
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Literal, Callable, Optional

class SequenceDataset(Dataset):
    def __init__(
        self,
        data: xr.Dataset,
        var: str | list = "sla",
        seq_length: int = 3,
        seq_target: int = 1,
        seq_stride: int = 1,
        stack_vars: bool = True, # merge variables into a single tensor (default: True)
    ):
        self.data = data
        self.var = var if isinstance(var, list) else [var]
        
        assert 0 < seq_stride <= seq_length + seq_target, "seq_stride must be positive and <= seq_length + seq_target"
        assert all(v in self.data for v in self.var), f"One or more variable in {self.var} has not been found in the dataset"

        self.seq_length = seq_length
        self.seq_target = seq_target
        self.seq_stride = seq_stride
        self.stack_vars = stack_vars

        self.length = self.data[self.var[0]].shape[0]  # Assuming all variables have the same time dimension
        self.length = (self.length - seq_length - seq_target) // seq_stride + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert 0 <= idx < self.length, f"Index {idx} out of bounds for dataset length {self.length}"

        seq_start = idx * self.seq_stride
        seq_end = seq_start + self.seq_length

        x, y = {}, {}

        for v in self.var:
            x[v] = self.data[v][seq_start:seq_end]  # (S, H, W)
            y[v] = self.data[v][seq_end:seq_end + self.seq_target]  # (1, H, W)
            
            # Remove nan values
            x[v] = np.nan_to_num(x[v], nan=0.0)
            y[v] = np.nan_to_num(y[v], nan=0.0)

            x[v] = torch.from_numpy(x[v]).to(torch.float32)
            y[v] = torch.from_numpy(y[v]).to(torch.float32)

        if self.stack_vars:
            if len(self.var) > 1:
                x = torch.stack([x[v] for v in self.var], dim=1) # [(S, H, W), ...] -> (S, C, H, W)
                y = torch.stack([y[v] for v in self.var], dim=1) # [(1, H, W), ...] -> (1, C, H, W)
            else:
                x = x[self.var[0]]
                y = y[self.var[0]]

        # # Metadata might be useful to embed when more zone are integrated
        # latitude = self.data.isel(time=slice(seq_start, seq_end)).latitude
        # longitude = self.data.isel(time=slice(seq_start, seq_end)).longitude
        # days = self.data.isel(time=slice(seq_start, seq_end)).time.dt.dayofyear.to_numpy()
        # # latitude, longitude = np.meshgrid(latitude, longitude, indexing="ij")
        # days = torch.from_numpy(days).long()
        # longitude = (longitude + 180) / 360
        # latitude = (latitude + 90) / 180
        # latitude = torch.from_numpy(latitude).float()
        # longitude = torch.from_numpy(longitude).float()
        # metadata = {
        #     "lat": latitude,
        #     "lon": longitude,
        #     "days": days,
        # }
        return x, y


class CustomLoader(Dataset):
    """
    Reference: https://www.mdpi.com/2072-4292/16/8/1466#B32-remotesensing-16-01466
    """
    def __init__(
        self,
        data: xr.Dataset,
        var: str | list = "sla",
        mode: Literal["3d", "7d"] = "3d",
        stack_vars: bool = True, # merge variables into a single tensor (default: True)
    ):
        self.data = data
        self.var = var if isinstance(var, list) else [var]
        
        assert all(v in self.data for v in self.var), f"One or more variable in {self.var} has not been found in the dataset"
        self.seq_target = 1
        self.seq_stride = 1
        self.stack_vars = stack_vars

        if mode == "3d":
            self.seq_length = 28
            self.pick_days = list(range(0, 28, 3))
        else:
            self.seq_length = 36
            self.pick_days = [0, 7, 14, 17, 20, 23, 26, 27, 28, 29]

        self.length = self.data[self.var[0]].shape[0]  # Assuming all variables have the same time dimension
        self.length = (self.length - self.seq_length - 1) // self.seq_stride + 1


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert 0 <= idx < self.length, f"Index {idx} out of bounds for dataset length {self.length}"

        seq_start = idx * self.seq_stride
        seq_end = seq_start + self.seq_length

        x = {}
        y = {}

        for v in self.var:
            x[v] = [self.data[v][seq_start + i].to_numpy() for i in self.pick_days]
            y[v] = np.array([self.data[v][seq_end].to_numpy()])
            x[v] = np.stack(x[v], axis=0)  # (S, H, W)

            x[v] = torch.from_numpy(x[v]).float()  # (S, H, W)
            y[v] = torch.from_numpy(y[v]).float() # (1, H, W)

            # Remove nan values
            x[v] = torch.nan_to_num(x[v], nan=0.0)
            y[v] = torch.nan_to_num(y[v], nan=0.0)


        if self.stack_vars and len(self.var) > 1:
            x = torch.stack([x[v] for v in self.var], dim=1) # [(S, H, W), ...] -> (S, C, H, W)
            y = torch.stack([y[v] for v in self.var], dim=1) # [(1, H, W), ...] -> (C, H, W)
        else:
            x = x[self.var[0]]
            y = y[self.var[0]]
    
        return x, y