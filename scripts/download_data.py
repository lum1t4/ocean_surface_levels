import copernicusmarine
import os
import argparse
from pathlib import Path

USERNAME = os.getenv("CMEMS_USERNAME")
PASSWORD = os.getenv("CMEMS_PASSWORD")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sea surface heights data")
    # cmems_obs-sl_eur_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D
    parser.add_argument("--dataset", type=str, help="Dataset to download", default="cmems_obs-sl_eur_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D")
    parser.add_argument("--output_directory", type=str, help="Output directory", default="data/raw")
    args = parser.parse_args()

    dataset = args.dataset
    output_directory = args.output_directory
    assert Path(output_directory).exists(), f"Output directory {output_directory} does not exist"
    copernicusmarine.get(dataset, username=USERNAME, password=PASSWORD, output_directory=output_directory)
