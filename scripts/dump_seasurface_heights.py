import xarray as xr
from pathlib import Path
data_dir = Path('data/raw/SEALEVEL_EUR_PHY_L4_NRT_008_060/cmems_obs-sl_eur_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D_202311/')
files = list(data_dir.glob('**/*.nc'))


# Open multiple files as a single dataset
dataset = xr.open_mfdataset(
    files,
    combine='by_coords',  # Combines along matching dimensions (e.g., time, lat, lon)
    parallel=False        # Disables parallel reading to avoid segmentation fault
)

# Load the dataset into memory (optional, but useful for faster access)
dataset = dataset.load()

# Save the dataset to a single NetCDF file
dataset.to_netcdf('data/processed/sea_level.nc')
