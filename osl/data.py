import xarray as xr
import torch
import numpy as np

import os
from pathlib import Path


def dataset_stats_compute(dataset: xr.Dataset, variables: list) -> dict:
    return {
        v: {
            'mean': np.nanmean(dataset[v]),
            'std': np.nanstd(dataset[v]),
            'min': np.nanmin(dataset[v]),
            'max': np.nanmax(dataset[v]),
        } for v in variables
    }


def dataset_stats_save(stats: dict, stats_path: str | Path):
    import json
    with open(stats_path, 'w') as fd:
        json.dump(stats, fd)


def dataset_stats_read(stats_path: str | Path):
    import json
    with open(stats_path, 'r') as fd:
        content = json.load(fd)
    return content


def dataset_stats_load(stats_path: str | Path, dataset: xr.Dataset, variables: list, overwrite: bool = False):
    if os.path.exists(stats_path):
        stats = dataset_stats_read(stats_path)
        return stats
    
    stats = dataset_stats_compute(dataset, ['sla', 'ugos', 'vgos'])
    if overwrite:
        dataset_stats_save(stats, stats_path)
    return stats



# TODO: implement test style dataset where valid chuncks seek + month (year?)
# need to compute rollout metrics

# ------------------------
# Dataset
# ------------------------
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: xr.Dataset,
        vars: str | list = ["sla", "ugos", "vgos"],
        seq_length: int = 3,
        seq_stride: int | None = None,
        seq_seek: int = 0,
        normalize: bool = False,
        stats: dict = {},
        bound: list | tuple | None = None
    ):


        self.data = data
        self.variables = vars if isinstance(vars, list) else [vars]
        self.normalize = normalize
        self.stats = stats

        self.seq_seek = seq_seek
        self.seq_length = seq_length
        self.seq_stride = seq_length if seq_stride is None else seq_stride

        assert seq_length > 0, "Sequence length must be positive and greater than zero"
        assert seq_stride is None or seq_stride > 0, "Sequence stride must be positive and greater than zero"
        assert all(v in self.data for v in self.variables), f"One or more variable in {self.variables} has not been found in the dataset"

        self.mean = np.stack([self.stats[v]['mean'] for v in self.variables])
        self.std = np.stack([self.stats[v]['std'] for v in self.variables])

        # Assuming all variables have the same time dimension
        self.bound = bound or [0, self.data[self.variables[0]].shape[0]]

    def __len__(self):
        size = self.bound[-1] - self.bound[0]
        return (size - self.seq_length) // self.seq_stride + 1
    
    def get_land_mask(self) -> torch.Tensor:
        """Returns a boolean tensor (H, W) indicating land points in the dataset."""
        valid = ~np.isnan(self.data[self.variables[0]][0].values)
        return torch.from_numpy(valid.astype(np.bool_))

    def __getitem__(self, index: int) -> dict:
        seq_m = self.bound[0] + index * self.seq_stride
        seq_s = seq_m - self.seq_seek
        seq_e = seq_m + self.seq_length

        values = np.stack([self.data[v][seq_s:seq_e] for v in self.variables], axis=1)
        values = np.nan_to_num(values, nan=0.0)
        if self.normalize:
            values = (values - self.mean) / self.std
        values = torch.from_numpy(values)
        days = self.data.isel(time=slice(seq_s, seq_e)).time.dt.dayofyear.to_numpy()

        if self.seq_seek > 0:
            return {"inputs": values[:self.seq_seek], "targets": values[self.seq_seek:], "days": days}
        return {"inputs": values, "days": days}


class FrameDataset(torch.utils.data.Dataset):
    """Returns individual frames (C, H, W), not sequences."""

    def __init__(
        self,
        data: xr.Dataset,
        vars: str | list = ["sla", "ugos", "vgos"],
        normalize: bool = False,
        stats: dict = {},
        bound: list | tuple | None = None
    ):
        self.data = data
        self.variables = vars if isinstance(vars, list) else [vars]
        self.normalize = normalize
        self.stats = stats
        # Assuming all variables have the same time dimension
        self.bound = bound or [0, self.data[self.variables[0]].shape[0]]

    def __len__(self):
        return self.bound[-1] - self.bound[0]

    def get_land_mask(self) -> torch.Tensor:
        """Returns a boolean tensor (H, W) indicating land points in the dataset."""
        valid = ~np.isnan(self.data[self.variables[0]][0].values)
        return torch.from_numpy(valid.astype(np.bool_))

    def __getitem__(self, index: int) -> dict:
        channels = []
        for var in self.variables:
            frame = self.data[var][index].values
            frame = np.nan_to_num(frame, nan=0.0)
            if self.normalize:
                frame = (frame - self.stats[var]['mean']) / self.stats[var]['std']
            channels.append(torch.from_numpy(frame).to(dtype=torch.float32))

        x = torch.stack(channels, dim=0)  # (C, H, W)
        return {"inputs": x}



class TimeInterval(SequenceDataset):
    def __init__(
        self,
        data: xr.Dataset,
        interval: list[int],
        var = ["sla", "ugos", "vgos"],
        normalize: bool = False,
        mean: list | np.ndarray | None = None,
        std: list | np.ndarray | None = None,
    ):
        super().__init__(data, var, interval[-1] + 1, None, normalize, mean, std)
        self.interval = interval

    def __getitem__(self, index: int) -> dict:
        seq_s = index * self.seq_stride
        
        values = {}
        for v in self.variables:
            values[v] = np.stack([self.data[v][seq_s + i] for i in self.interval])
            values[v] = torch.from_numpy(np.nan_to_num(values[v], nan=0.0)).to(dtype=torch.float32)

        values = torch.stack([values[v] for v in self.variables], dim=1)
        days = self.data.isel(time=[seq_s + i for i in self.interval]).time.dt.dayofyear.to_numpy()
        return {"inputs": values[:-1], "targets": values[1:], "days": days}


# register_dataset('', S)
# register_dataset('time-interval-28d', TimeInterval, dict(interval=[0, 3,  6,  9, 12, 15, 18, 21, 24, 27, 30]))
# register_dataset('time-interval-30d', TimeInterval, dict(interval=[0, 7, 14, 17, 20, 23, 26, 27, 28, 29, 36]))
