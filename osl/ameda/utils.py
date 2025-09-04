import numpy as np
from pathlib import Path
from scipy.interpolate import griddata


ROOT = Path(__file__).resolve().parent

# Constants
DEG2RAD = np.pi / 180
DEG2NM = 60.0
NM2KM = 1.8520
EARTH_RADIUS = 6378.137  # km


def is_regular_grid(lon, lat):
    """Check if grid is regular"""
    dy_var = np.var(np.diff(lat, axis=0))
    dx_var = np.var(np.diff(lon, axis=1))
    return dy_var < 1e-6 and dx_var < 1e-6



def get_missing_val_2d(x: np.ndarray, y: np.ndarray, 
                        field: np.ndarray, missvalue: float = np.nan,
                        default: float = 0) -> np.ndarray:
    """
    Fill missing values in a 2D field using nearest neighbor interpolation.
    
    Parameters
    ----------
    x, y : ndarray
        Grid coordinates
    field : ndarray
        2D field with missing values
    missvalue : float
        Value indicating missing data
    default : float
        Default value if no valid data
        
    Returns
    -------
    field : ndarray
        Field with missing values filled
    """
    field = field.copy()
    
    if np.isnan(missvalue):
        ismask = np.isnan(field)
    else:
        ismask = (field == missvalue)
    
    isdata = ~ismask
    
    # Check if there's any valid data
    if not np.any(isdata):
        return np.full_like(field, default)
    
    if np.sum(isdata) < 6:
        default = np.min(field[isdata])
        return np.full_like(field, default)
    
    if not np.any(ismask):
        return field
    
    # Fill missing values using nearest neighbor
    if x.ndim == 1:
        x, y = np.meshgrid(x, y)
    
    valid_points = np.column_stack([x[isdata].ravel(), y[isdata].ravel()])
    valid_values = field[isdata].ravel()
    invalid_points = np.column_stack([x[ismask].ravel(), y[ismask].ravel()])
    
    filled_values = griddata(valid_points, valid_values, invalid_points,
                            method='nearest', fill_value=default)
    
    field[ismask] = filled_values
    
    return field
