import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from pathlib import Path
from osl.ameda.utils import ROOT


ROSSBY_DEFAULT_FILE = ROOT / "assets" / "rossrad.dat"

def read_rossby(resolution: float = 0.25, file: Path = ROSSBY_DEFAULT_FILE):
    data = np.loadtxt(file.as_posix())
    y = data[:, 0]  # latitude
    x = data[:, 1]  # longitude (0-360)
    phase_speed = data[:, 2]  # phase speed (m/s)
    rossby_radius = data[:, 3]  # Rossby radius (km)
    # Convert longitude from 0-360 to -180-180
    x[x > 180] = x[x > 180] - 360
    lon = np.arange(-180, 180, resolution)
    lat = np.arange(-80, 80, resolution)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    # Grid the data using scipy's griddata
    points = np.column_stack([y, x])
    phase_speed = phase_speed * 3600 * 24 / 1000  # m/s to km/day
    # Rd: Rossby radius in km (already in correct units)
    rossby_radius = griddata(points, rossby_radius, (lat_grid, lon_grid), method="linear", fill_value=np.nan)
    # Vd: phase speed in km/day
    # Convert from m/s to km/day: m/s * 3600*24/1000
    phase_speed = griddata(points, phase_speed, (lat_grid, lon_grid), method="linear", fill_value=np.nan)
    return lon, lat, phase_speed, rossby_radius


def extract_region(
        rossby: np.ndarray, # (X, Y)
        lat: np.ndarray, # (X,)
        lon: np.ndarray, # (Y,)
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        resolution: float | None = None
) -> np.ndarray:
    ilat = (lat >= min_lat) & (lat < max_lat)
    ilon = (lon >= min_lon) & (lon < max_lon)
    lat_box = lat[ilat]
    lon_box = lon[ilon]
    rossby_box = rossby[np.ix_(ilat, ilon)]

    if resolution is None:
        return lat_box, lon_box, rossby_box

    new_lat = np.arange(min_lat, max_lat, resolution)
    new_lon = np.arange(min_lon, max_lon, resolution)

    interpolator = RegularGridInterpolator(
        (lat_box, lon_box),
        rossby_box,
        method="linear",  # "nearest" or "linear" or "slinear" (SciPy>=1.14)
        bounds_error=False,
        fill_value=np.nan
    )
    pts = np.column_stack([new_lat.ravel(), new_lon.ravel()])
    new_rossby = interpolator(pts).reshape(new_lat.size, new_lon.size)
    return lat_box, lon_box, new_rossby



def read_rossby_region(
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    resolution: float = 0.0625,
    file: Path = ROSSBY_DEFAULT_FILE,
):
    """
    Read Rossby radius file and extract directly the region of interest
    at desired resolution (default 0.0625°).
    """
    # Load data
    data = np.loadtxt(file.as_posix())
    lat_raw = data[:, 0]
    lon_raw = data[:, 1]
    phase_speed = data[:, 2]  # m/s
    rossby_radius = data[:, 3]  # km

    # Convert longitudes from 0-360 to -180-180
    lon_raw[lon_raw > 180] -= 360

    # Target ROI grid
    lat = np.arange(min_lat, max_lat + resolution, resolution)
    lon = np.arange(min_lon, max_lon + resolution, resolution)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Interpolate directly in ROI
    points = np.column_stack([lat_raw, lon_raw])
    # Convert phase speed to km/day
    phase_speed = phase_speed * 3600 * 24 / 1000.0
    # vd = griddata(points, phase_speed, (lat_grid, lon_grid), method="linear", fill_value=np.nan)
    rd = griddata(points, rossby_radius, (lat_grid, lon_grid), method="linear", fill_value=np.nan)
    return rd
