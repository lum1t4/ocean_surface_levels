import xarray as xr
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Literal, Callable, Optional


class SequenceDataset(Dataset):
    def __init__(self, data: xr.Dataset,
        var: str | list = ["sla", "ugos", "vgos"],
        seq_length: int = 3,
        seq_stride: int | None = None
    ):
        self.data = data
        self.var = var if isinstance(var, list) else [var]

        assert seq_length > 0, "seq_length must be positive and greater than zero"
        assert seq_stride is None or seq_stride > 0, "seq_stride must be positive and greater than zero"
        assert all(v in self.data for v in self.var), f"One or more variable in {self.var} has not been found in the dataset"


        self.seq_length = seq_length
        self.seq_stride = seq_length if seq_stride is None else seq_stride

        # Assuming all variables have the same time dimension
        length = self.data[self.var[0]].shape[0]
        self.length = (length - self.seq_length - 1) // self.seq_stride + 1

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> dict:
        seq_s = index * self.seq_stride
        seq_e = seq_s + self.seq_length

        values = {}
        for v in self.var:
            values[v] = self.data[v][seq_s:seq_e + 1]
            values[v] = torch.from_numpy(np.nan_to_num(values[v], nan=0.0)).to(dtype=torch.float32)

        days = self.data.isel(time=slice(seq_s, seq_e)).time.dt.dayofyear.to_numpy()
        values = torch.stack([values[v] for v in self.var], dim=1)        
        return {"inputs": values[:-1], "targets": values[1:], "days": days}



class TimeInterval(SequenceDataset):
    def __init__(self, data, interval: list[int], var = ["sla", "ugos", "vgos"]):
        super().__init__(data, var, interval[-1] + 1, interval[-1] + 1)
        self.interval = interval

    def __getitem__(self, index: int) -> dict:
        seq_s = index * self.seq_stride
        
        values = {}
        for v in self.var:
            values[v] = np.stack([self.data[v][seq_s + i] for i in self.interval])
            values[v] = torch.from_numpy(np.nan_to_num(values[v], nan=0.0)).to(dtype=torch.float32)

        values = torch.stack([values[v] for v in self.var], dim=1)
        days = self.data.isel(time=[seq_s + i for i in self.interval]).time.dt.dayofyear.to_numpy()
        return {"inputs": values[:-1], "targets": values[1:], "days": days}


# register_dataset('', S)
# register_dataset('time-interval-28d', TimeInterval, dict(interval=[0, 3,  6,  9, 12, 15, 18, 21, 24, 27, 30]))
# register_dataset('time-interval-30d', TimeInterval, dict(interval=[0, 7, 14, 17, 20, 23, 26, 27, 28, 29, 36]))
