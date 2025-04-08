from pathlib import Path
import xarray as xr
import numpy as np
from torch.utils.data import Dataset


class SLADataser(Dataset):
    """
    Dataset for ocean satellite data.
    
    PHYSICS/GEOL: 
    - 'sla' (Sea Level Anomaly) indicates deviations in sea surface height
      and is used here as the input signal.
    - 'ugosa' and 'vgosa' are the eastward and northward geostrophic velocity 
      components, respectively. These are the target variables that represent 
      ocean surface currents derived from altimetry data.
    """
    
    def __init__(self, path: Path, seq_len: int = 3, stride: int = 3, pred_horizon: int = 1):
        self.path = path
        self.seq_len = seq_len
        self.stride = stride
        self.pred_horizon = pred_horizon

        ds = xr.open_dataset(path)
