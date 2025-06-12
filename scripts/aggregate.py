import xarray as xr
from pathlib import Path
import argparse
import dotenv

dotenv.load_dotenv()

def aggregate(src: str, dst: str):
    src = Path(src)
    dst = Path(dst)

    assert src.is_dir(), f"Source path {src} is not a directory or does not exist"
    files = list(src.glob('**/*.nc'))
    # Open multiple files as a single dataset
    dataset = xr.open_mfdataset(
        files,
        combine='by_coords',  # Combines along matching dimensions (e.g., time, lat, lon)
        parallel=False        # Disables parallel reading to avoid segmentation fault
    )
    # Load the dataset into memory (optional, but useful for faster access)
    dataset = dataset.load()
    # Save the dataset to a single NetCDF file
    dataset.to_netcdf(dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate data by coords")
    parser.add_argument("--src", type=str, help="Dataset to download")
    parser.add_argument("--dst", type=str, help="filename", default="data/processed")
    arguments = parser.parse_args()
    aggregate(arguments.src, arguments.dst)