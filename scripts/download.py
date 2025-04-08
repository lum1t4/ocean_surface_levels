import copernicusmarine
import os
import argparse
from pathlib import Path



def download(src: str, dst: str):
    USERNAME = os.getenv("CMEMS_USERNAME")
    PASSWORD = os.getenv("CMEMS_PASSWORD")

    assert USERNAME is not None, "CMEMS_USERNAME environment variable is not set"
    assert PASSWORD is not None, "CMEMS_PASSWORD environment variable is not set"
    assert Path(dst).exists(), f"Output directory {dst} does not exist"
    copernicusmarine.get(src, username=USERNAME, password=PASSWORD, output_directory=dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sea surface heights data")
    parser.add_argument("--src", type=str, help="Dataset to download", default="cmems_obs-sl_eur_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D")
    parser.add_argument("--dst", type=str, help="Destination directory", default="data/raw")
    args = parser.parse_args()
    download(args.src, args.dst)
