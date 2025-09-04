"""
Field computation module for AMEDA (Angular Momentum Eddy Detection Algorithm).

This module handles:
- Loading velocity fields from AVISO NetCDF data
- Computing detection fields (KE, divergence, vorticity, OW, LOW, LNAM)
- Grid interpolation and mask management
"""

from typing import Optional, Tuple

import numpy as np
import xarray as xr
from scipy.interpolate import interp2d

from osl.ameda.params import AMEDAParams
from osl.ameda.utils import get_missing_val_2d


def load_fields(
    dataset: xr.Dataset,
    step: int,
    params: AMEDAParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load velocity fields from AVISO NetCDF data.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Input NetCDF dataset
    step : int
        Time step to load
    params : AMEDAParams
        Algorithm parameters
    resol : int
        Interpolation factor (1 = no interpolation)
    deg : int
        Degradation factor (1 = no degradation)
    
    Returns
    -------
    x, y : ndarray
        Grid coordinates (lon, lat)
    mask : ndarray
        Ocean mask (1 = ocean, 0 = land)
    u, v : ndarray
        Velocity components (m/s)
    ssh : ndarray or None
        Sea surface height (m) if type_detection >= 2
    """
    print(f"Loading fields at step {step}...")
    
    # Extract data at given time step
    data_step = dataset.isel(time=step)
    # Get coordinates and fields
    lon0 = data_step[params.x_name].values
    lat0 = data_step[params.y_name].values
    
    # Create 2D meshgrids if needed
    if lon0.ndim == 1:
        lon0, lat0 = np.meshgrid(lon0, lat0)
    
    # Get velocity fields
    u0 = data_step[params.u_name].values
    v0 = data_step[params.v_name].values
    
    # Create mask from velocity fields
    mask0 = ~(np.isnan(u0) | np.isnan(v0))
    mask0 = mask0.astype(float)
    
    # Get SSH if needed
    ssh0 = None
    if params.type_detection >= 2:
        if params.s_name in data_step:
            ssh0 = data_step[params.s_name].values
    
    # Apply degradation if requested
    if params.deg != 1:
        print(f"  Degrading fields by factor {params.deg}")
        x = lon0[::params.deg, ::params.deg]
        y = lat0[::params.deg, ::params.deg]
        mask = mask0[::params.deg, ::params.deg]
        u = u0[::params.deg, ::params.deg]
        v = v0[::params.deg, ::params.deg]
        if ssh0 is not None:
            ssh = ssh0[::params.deg, ::params.deg]
        else:
            ssh = None
    else:
        x = lon0
        y = lat0
        mask = mask0
        u = u0.copy()
        v = v0.copy()
        ssh = ssh0.copy() if ssh0 is not None else None
    
    N, M = x.shape
    
    # Interpolation
    if params.resol == 1:
        print("NO INTERPOLATION")
        
        # Set NaN in land
        u[mask == 0] = np.nan
        v[mask == 0] = np.nan
        
        # Enlarge mask by 1 pixel into land
        print("Enlarging coastal mask by 1 pixel...")
        for i in range(N):
            for j in range(M):
                if mask[i, j] == 0:
                    # Check if any neighbor is ocean
                    i_min, i_max = max(i-1, 0), min(i+1, N-1)
                    j_min, j_max = max(j-1, 0), min(j+1, M-1)
                    
                    if np.sum(mask[i_min:i_max+1, j_min:j_max+1]) > 0:
                        u[i, j] = 0
                        v[i, j] = 0
                        if ssh is not None and np.isnan(ssh[i, j]):
                            ssh_neighbors = ssh[i_min:i_max+1, j_min:j_max+1]
                            ssh[i, j] = np.nanmean(ssh_neighbors)
    
    else:
        print(f"Interpolating grid by factor {params.resol}")
        
        # New grid size
        Ni = params.resol * (N - 1) + 1
        Mi = params.resol * (M - 1) + 1
        
        # Grid spacing
        dy = (y[1, 0] - y[0, 0]) / params.resol if N > 1 else 0
        dx = (x[0, 1] - x[0, 0]) / params.resol if M > 1 else 0
        
        # Create interpolated grid
        xi, yi = np.meshgrid(
            np.arange(Mi) * dx + x.min(),
            np.arange(Ni) * dy + y.min()
        )
        
        # Interpolate mask
        mask_interp = interp2d(x[0, :], y[:, 0], mask, kind='linear', bounds_error=False, fill_value=0)
        maski = mask_interp(xi[0, :], yi[:, 0])
        maski = np.where(maski >= 0.5, 1.0, 0.0)

        # Enlarge mask by 1 pixel
        maski1 = maski.copy()
        print("Enlarging coastal mask by 1 pixel...")
        for i in range(Ni):
            for j in range(Mi):
                if maski[i, j] == 0:
                    i_min, i_max = max(i-1, 0), min(i+1, Ni-1)
                    j_min, j_max = max(j-1, 0), min(j+1, Mi-1)
                    
                    if np.sum(maski[i_min:i_max+1, j_min:j_max+1]) > 0:
                        maski1[i, j] = 1
        
        # Set land velocities to 0 for interpolation
        u[mask == 0 | np.isnan(u)] = 0
        v[mask == 0 | np.isnan(v)] = 0
        
        # Interpolate velocity fields
        fu_interp = interp2d(x[0, :], y[:, 0], u, kind='cubic', bounds_error=False, fill_value=0)
        fv_interp = interp2d(x[0, :], y[:, 0], v, kind='cubic', bounds_error=False, fill_value=0)
        ui = fu_interp(xi[0, :], yi[:, 0])
        vi = fv_interp(xi[0, :], yi[:, 0])
        
        # Interpolate SSH if needed
        if ssh is not None:
            ssh1 = get_missing_val_2d(x, y, ssh)
            ssh_interp = interp2d(x[0, :], y[:, 0], ssh1, kind='cubic', bounds_error=False, fill_value=np.nan)
            sshi = ssh_interp(xi[0, :], yi[:, 0])
        else:
            sshi = None
        
        # Apply enlarged mask
        ui[maski1 == 0] = np.nan
        vi[maski1 == 0] = np.nan
        if sshi is not None:
            sshi[maski1 == 0] = np.nan
        
        # Export interpolated fields
        x = xi
        y = yi
        mask = maski
        u = ui
        v = vi
        ssh = sshi
    
    return x, y, mask, u, v, ssh



