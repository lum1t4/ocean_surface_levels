import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator, griddata

from osl.ameda.rossby import read_rossby_region

G = 9.8  # Gravity (m/s²)
T = 24 * 3600  # Rotation period (seconds) in a sidereal day
DEG2RAD = np.pi / 180
OMEGA =  2 * np.pi / T  # rad s^-1
EARTH_RADIUS = 6378.137 # Earth radius in kilometers
R = EARTH_RADIUS * np.pi / 180.0  # Radius in radians for degree conversion 111.32m
RAD2DEG = 180.0 / np.pi
DEG2NM = 60.0
NM2KM = 1.8520

def sw_dist(lat, lon, units="nm"):
    """
    Distance and bearing between successive (lat, lon) positions
    using the 'Plane Sailing' approximation.

    Parameters
    ----------
    lat : array-like
        Latitudes in decimal degrees (+N, -S).
    lon : array-like
        Longitudes in decimal degrees (+E, -W).
    units : str, optional
        'nm' (nautical miles, default) or 'km'.

    Returns
    -------
    dist : ndarray
        Distance(s) between successive positions in given units.
    phaseangle : ndarray
        Bearing angle relative to x-axis (east).
        Range: -180..+180 (E=0, N=90, S=-90).
    """

    lat = np.asarray(lat, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()

    if lat.shape != lon.shape:
        raise ValueError("lat and lon must have the same shape")


    dlon = np.diff(lon)
    # wrap longitude difference to [-180, 180]
    dlon = np.where(np.abs(dlon) > 180,
                    -np.sign(dlon) * (360 - np.abs(dlon)),
                    dlon)

    latrad = np.abs(lat * DEG2RAD)
    dep = np.cos((latrad[1:] + latrad[:-1]) / 2.0) * dlon
    dlat = np.diff(lat)

    # distance in nautical miles
    dist = DEG2NM * np.sqrt(dlat**2 + dep**2)

    if units == "km":
        dist = dist * NM2KM
    elif units != "nm":
        raise ValueError("units must be 'nm' or 'km'")

    # bearing (phase angle relative to east)
    phaseangle = np.angle(dep + 1j * dlat) * RAD2DEG

    return dist, phaseangle


def distance_matrix_from_ll(lon, lat):
    """
    lon, lat: 2D meshgrid in degrees
    returns: 2D average spacing (km) cell-centered
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    ny, nx = lon.shape

    # central differences along E-W (km per degree lon = 111.32*cos(lat))
    dlon = np.empty_like(lon)
    dlon[:, 1:-1] = (lon[:, 2:] - lon[:, :-2]) / 2.0
    dlon[:, 0]    = lon[:, 1] - lon[:, 0]
    dlon[:, -1]   = lon[:, -1] - lon[:, -2]
    east = 111.32 * np.cos(np.deg2rad(lat)) * np.abs(dlon)
    # central differences along N-S (km per degree lat ≈ 111.32)
    dlat = np.empty_like(lat)
    dlat[1:-1, :] = (lat[2:, :] - lat[:-2, :]) / 2.0
    dlat[0, :]    = lat[1, :] - lat[0, :]
    dlat[-1, :]   = lat[-1, :] - lat[-2, :]
    north = 111.32 * np.abs(dlat)
    return 0.5 * (east + north)


def interp(x_orig, y_orig, values, x_new, y_new) -> np.ndarray:
    """
    Interpolates values from an original regular grid (x_orig, y_orig)
    onto a new regular grid (x_new, y_new).
    """
    y_coords = y_orig[:, 0]
    x_coords = x_orig[0, :]
    intrp = RegularGridInterpolator(
        (y_coords, x_coords),
        values,
        bounds_error=False,
        fill_value=np.nan
    )
    pts = np.column_stack([y_new.ravel(), x_new.ravel()])
    return intrp(pts).reshape(x_new.shape)


sshtype = Literal["adt", "sla", "ssh"]

@dataclass
class AMEDAParams:
    grid_ll: int = 1  # when true longitude/latitude coordinates are given in degrees else cartesian coordinates in km (x,y)
    grid_reg: int = 1
    deg: int = 1
    y_name: str = "lat"
    x_name: str = "lon"
    m_name: str = "mask"
    u_name: str = "ugos"
    v_name: str = "vgos"
    s_name: sshtype = "sla"
    Rd_typ: float = 12  # Typical Rossby radius (km) for Med Sea
    nRmin: float = 0.5  # Minimal eddy size in terms of Dx
    K: float = 0.7  # LNAM threshold for center detection
    resol: int = 1  # Grid interpolation coefficient
    type_detection: int = 3  # 1: velocity, 2: SSH, 3: both (and keep max velocity along the eddy contour)
    DH: float = 0.002  # SSH spacing for streamline scanning (m)
    nH_lim: int = 200  # Maximum number of streamlines
    n_min: int = 6  # Minimum points for valid contour
    epsil: float = 0.01  # Velocity increase threshold (1%)
    k_vel_decay: float = 0.97  # Velocity decay coefficient
    nR_lim: int = 100  # Maximum eddy size in Rd units
    Np: int = 3
    nrho_lim: float = 0.2  # Density threshold for eddy detection
    lat_min: float = 5  # Minimum latitude for detection
    dc_max: float = 3.5
    V_eddy: float = 6.5  # Typical eddy speed (km/day)
    Dt: int = 10  # Maximum tracking delay (days) for AVISO
    cut_off: int = 0  # Minimum eddy lifetime (0 = 2*turnover time)
    D_stp: int = 4  # Steps for averaging during tracking
    Lb: int = 10
    N_can: int = 30
    extended_diags: int = 1
    streamlines: int = 1
    daystreamfunction: int = 1
    periodic: int = 0
    level: int = 1
    dps: int = 1
    nrt: int = 1
    b: np.ndarray = None # shape: (LAT, LON) Half box size for LNAM calculation
    bx: np.ndarray = None # shape: (LAT, LON) Half box size of streamline scan
    Dx: np.ndarray = None # shape: (LAT, LON) grid horizontal spacing (km)
    Rd: np.ndarray = None # shape: (LAT, LON) Rossby radius
    gama: np.ndarray = None # shape: (LAT, LON) Rd resolution
    Rb: np.ndarray = None # shape: (LAT, LON) Half box size for LNAM check
    f: np.ndarray = None # shape: (LAT, LON) # Coriolis parameter
    bi: np.ndarray = None # shape: (LAT * resol, LON * resol) interpolated grid of b
    bxi: np.ndarray = None # shape: (LAT * resol, LON * resol) interpolated grid of bx
    Dxi: np.ndarray = None # shape: (LAT * resol, LON * resol) interpolated grid of Dx
    Rdi: np.ndarray = None # shape: (LAT * resol, LON * resol) interpolated grid of Rd
    gamai: np.ndarray = None # shape: (LAT * resol, LON * resol) interpolated grid of gama
    fi: np.ndarray = None # shape: (LAT * resol, LON * resol) interpolated grid of f


def derive_params(dataset: xr.Dataset, step: int, params: AMEDAParams) -> AMEDAParams:
    data_step = dataset.isel(time=step)
    lon = data_step[params.x_name].values
    lat = data_step[params.y_name].values
    ugos = data_step[params.u_name].values


    Rd = read_rossby_region(lon.min(), lon.max(), lat.min(), lat.max(), lon[1] - lon[0])
    lon0 = np.array(lon)
    lat0 = np.array(lat)
    lon0, lat0 = np.meshgrid(lon0, lat0)
    mask0 = ~np.isnan(ugos)

    x = lon0[::params.deg, ::params.deg]
    y = lat0[::params.deg, ::params.deg]
    mask = mask0[::params.deg, ::params.deg]
    # Grid horizontal spacing I x J km - Grid
    if params.grid_ll:
        if params.grid_reg:
            Dx = distance_matrix_from_ll(x, y)
        else:
            print("Irregular grid: regridding")
            dy = np.mean(np.diff(y, axis=0))
            dx = np.mean(np.diff(x, axis=1))
            # Create regular grid
            xr, yr = np.meshgrid(
                np.arange(x.min(), x.max() + dx, dx),
                np.arange(y.min(), y.max() + dy, dy)
            )
            N, M = xr.shape
            Dx = distance_matrix_from_ll(xr, yr)
            # Regrid mask
            maskr = griddata(
                (x.ravel(), y.ravel()), mask.ravel(), (xr, yr),
                method='linear', fill_value=0
            )
            maskr[maskr < 0.5] = 0
            maskr[maskr >= 0.5] = 1
            maskr[(xr < x.min()) | (xr > x.max()) | (yr < y.min()) | (yr > y.max())] = 0
    else:
        Dx = 0.5 * (np.abs(np.diff(x, axis=1, append=x[:, -1:])) + np.abs(np.diff(y, axis=0, append=y[-1:, :])))


    if params.grid_ll:
        if params.grid_reg:
            f = 4.0 * math.pi / T * np.sin(np.deg2rad(y))
        else:
            f = 4.0 * math.pi / T * np.sin(np.deg2rad(yr))
    else:
        f = 4.0 * math.pi / T * np.ones_like(y)

    if Rd is None:
        Rd = np.full_like(x, params.Rd_typ)  # Rossby radius in km

    idx = (Rd < 200.0) & np.isfinite(Rd) & (mask > 0)
    gama = np.full_like(Rd, np.nan)
    gama[idx] = Rd[idx] / Dx[idx]
    if params.resol is None or params.resol < 1:
        # resol towards ~3 pixels per Rd; clamp to [1,3]
        params.resol = int(np.clip(round(3.0 / np.nanmean(gama)), 1, 3))
    # LNAM box half length in pixels b s.t. Lb/Rd ≈ 1.2
    b = np.maximum(1, np.rint(1.2 * gama / 2.0))
    Rb = 2.0 * b / gama
    bx = np.maximum(1, np.rint(2.0 * gama * 5.0))

    # Interpolated grid (resol > 1)
    if params.resol == 1:
        if params.grid_reg:
            xi, yi, maski = x, y, mask
        else:
            xi, yi, maski = xr, yr, maskr
        f_i = f
        bi, bxi = b, bx
        Dxi, Rdi, gamai = Dx, Rd, gama
    else:
        Ni = params.resol * (x.shape[0] - 1) + 1
        Mi = params.resol * (x.shape[1] - 1) + 1
        if params.grid_reg:
            dy = (y[1, 0] - y[0, 0]) / params.resol
            dx = (x[0, 1] - x[0, 0]) / params.resol
        else:
            dy = (yr[1, 0] - yr[0, 0]) / params.resol
            dx = (xr[0, 1] - xr[0, 0]) / params.resol

        xi, yi = np.meshgrid(
            np.arange(Mi) * dx + x.min(),
            np.arange(Ni) * dy + y.min()
        )
        # Interpolate fields
        if params.grid_reg:
            maski = interp(x, y, mask, xi, yi)
        else:
            maski = interp(xr, yr, maskr, xi, yi)
            maski[(xi < x.min()) | (xi > x.max()) | (yi < y.min()) | (yi > y.max())] = 0
        maski = np.where(np.nan_to_num(maski, nan=0.0) >= 0.5, 1.0, 0.0)

        if params.grid_reg:
            f_i = interp(x, y, f, xi, yi)
            bi = np.rint(interp(x, y, b, xi, yi)) * params.resol
            bxi = np.rint(interp(x, y, bx, xi, yi)) * params.resol
            Rdi = interp(x, y, Rd, xi, yi)
            Dxi = interp(x, y, Dx, xi, yi)
        else:
            f_i = interp(xr, yr, f, xi, yi)
            bi = np.rint(interp(xr, yr, b, xi, yi)) * params.resol
            bxi = np.rint(interp(xr, yr, bx, xi, yi)) * params.resol
            Rdi = interp(xr, yr, Rd, xi, yi)
            Dxi = interp(xr, yr, Dx, xi, yi)

        gamai = Rdi / Dxi

    params.b = b
    params.bx = bx
    params.Dx = Dx
    params.Rd = Rd
    params.gama = gama
    params.Rb = Rb
    params.bi = bi
    params.bxi = bxi
    params.Dxi = Dxi
    params.Rdi = Rdi
    params.gamai = gamai
    params.f = f
    params.fi = f_i

    return params
