"""
Field computation module for AMEDA (Angular Momentum Eddy Detection Algorithm).

This module handles:
- Loading velocity fields from AVISO NetCDF data
- Computing detection fields (KE, divergence, vorticity, OW, LOW, LNAM)
- Grid interpolation and mask management
"""

from dataclasses import dataclass
from typing import Tuple

import numba as nb
import numpy as np
import xarray as xr

from osl.ameda.params import AMEDAParams, R


@dataclass
class DetectionFields:
    """Container for computed detection fields"""
    ke: np.ndarray      # Kinetic energy
    div: np.ndarray     # Divergence
    vort: np.ndarray    # Vorticity
    OW: np.ndarray      # Okubo-Weiss
    LOW: np.ndarray     # Local Okubo-Weiss
    LNAM: np.ndarray    # Local Normalized Angular Momentum



@nb.jit(nopython=True, parallel=False)
def compute_localized_fields(
    uu: np.ndarray,
    vv: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    okubo: np.ndarray,
    b: np.ndarray,
    f: np.ndarray,
    grid_ll: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LNAM and LOW fields
    
    Parameters
    ----------
    uu, vv : ndarray
        Velocity components
    x, y : ndarray
        Grid coordinates
    okubo : ndarray
        Okubo-Weiss field
    b : ndarray
        Half box size for LNAM calculation
    f : ndarray
        Coriolis parameter
    grid_ll : bool
        True if coordinates are lon/lat
    R : float
        Earth radius factor for lon/lat grids
    
    Returns
    -------
    L : ndarray
        LNAM field
    LOW : ndarray
        Local Okubo-Weiss field
    """
    N, M = uu.shape
    LNAM = np.zeros_like(uu)
    LOW = np.full_like(uu, np.nan)
    
    borders = int(np.max(b)) + 1
    
    for i in range(borders, N - borders + 1):
        for j in range(borders, M - borders + 1):
            
            if not np.isnan(vv[i, j]):
                bi = int(b[i, j])
                
                # Calculate LOW
                OW = okubo[i-bi:i+bi+1, j-bi:j+bi+1]
                LOW[i, j] = np.nanmean(OW)
                
                # Calculate LNAM
                xlocal = x[i-bi:i+bi+1, j-bi:j+bi+1]
                ylocal = y[i-bi:i+bi+1, j-bi:j+bi+1]
                ulocal = uu[i-bi:i+bi+1, j-bi:j+bi+1]
                vlocal = vv[i-bi:i+bi+1, j-bi:j+bi+1]
                
                # Center coordinates
                coordcentre = bi
                xc = xlocal[coordcentre, coordcentre]
                yc = ylocal[coordcentre, coordcentre]
                
                if grid_ll:
                    # Convert to km distances
                    d_xcentre = (xlocal - xc) * R * np.cos(ylocal * np.pi / 180)
                    d_ycentre = (ylocal - yc) * R
                else:
                    d_xcentre = xlocal - xc
                    d_ycentre = ylocal - yc
                
                # Angular momentum calculation
                cross = d_xcentre * vlocal - d_ycentre * ulocal
                dot = ulocal * d_xcentre + vlocal * d_ycentre
                produit = np.sqrt(ulocal**2 + vlocal**2) * np.sqrt(d_xcentre**2 + d_ycentre**2)
                
                sum_cross = np.nansum(cross)
                sum_dp = np.nansum(dot) + np.nansum(produit)
                
                if sum_dp != 0:
                    LNAM[i, j] = sum_cross / sum_dp * np.sign(f[i, j])
                else:
                    LNAM[i, j] = 0
    
    return LNAM, LOW


def compute_fields(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    step: int,
    params: AMEDAParams,
) -> DetectionFields:
    """
    Compute detection fields from velocity data.
    
    Parameters
    ----------
    x, y : ndarray
        Grid coordinates
    mask : ndarray
        Ocean mask
    u, v : ndarray
        Velocity components (m/s)
    params : AMEDAParams
        Algorithm parameters
    derived_params : AMEDADerivedParams
        Derived parameters (b, f, etc.)
    step : int
        Time step
    resol : int
        Interpolation factor
    
    Returns
    -------
    fields : DetectionFields
        Computed detection fields
    """
    print(f"Computing fields for step {step}")
    
    # Use interpolated parameters if resol > 1
    if params.resol > 1:
        b = params.bi
        f = params.fi
    else:
        b = params.b
        f = params.f
    
    # Ensure integer b values and handle NaN
    b = np.nan_to_num(b, nan=1.0)
    b = np.round(b).astype(int)
    b[b < 1] = 1
    
    # Calculate kinetic energy
    ke = (u**2 + v**2) / 2
    
    # Initialize derivative arrays
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dux = np.zeros_like(u)
    duy = np.zeros_like(u)
    dvx = np.zeros_like(v)
    dvy = np.zeros_like(v)
    
    # Compute spatial derivatives
    dx[1:-1, 1:-1] = x[1:-1, 2:] - x[1:-1, :-2]
    dy[1:-1, 1:-1] = y[2:, 1:-1] - y[:-2, 1:-1]
    
    if params.grid_ll:
        # Convert to meters
        dx = dx * R * np.cos(np.deg2rad(y))
        dy = dy * R
    
    # Convert to meters
    dx = dx * 1000
    dy = dy * 1000
    
    # Compute velocity derivatives
    dux[1:-1, 1:-1] = u[1:-1, 2:] - u[1:-1, :-2]
    duy[1:-1, 1:-1] = u[2:, 1:-1] - u[:-2, 1:-1]
    dvx[1:-1, 1:-1] = v[1:-1, 2:] - v[1:-1, :-2]
    dvy[1:-1, 1:-1] = v[2:, 1:-1] - v[:-2, 1:-1]
    
    # Avoid division by zero
    dx[dx == 0] = np.nan
    dy[dy == 0] = np.nan
    
    # Calculate Okubo-Weiss components
    sn = dux / dx - dvy / dy  # Shear
    ss = dvx / dx + duy / dy  # Strain
    om = dvx / dx - duy / dy  # Vorticity
    
    okubo = sn**2 + ss**2 - om**2
    
    # Calculate divergence
    div = dux / dx + dvy / dy
    
    # Calculate vorticity (sign-adjusted)
    vorticity = om * np.sign(f)
    
    # Calculate LNAM and LOW
    print("Computing LNAM...")
    
    # Ensure consistent float64 types for Numba
    u = u.astype(np.float64)
    v = v.astype(np.float64)
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    okubo = okubo.astype(np.float64)
    f = f.astype(np.float64)
    
    # Use Numba-accelerated function
    L, LOW = compute_localized_fields(u, v, x, y, okubo, b, f, params.grid_ll)
    
    # Handle NaN values
    L[np.isnan(L)] = 0
    
    # Apply mask to all fields
    fields = DetectionFields(
        ke=ke * mask,
        div=div * mask,
        vort=vorticity * mask,
        OW=okubo * mask,
        LOW=LOW * mask,
        LNAM=L * mask
    )
    
    return fields

