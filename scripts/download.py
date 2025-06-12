"""
Download a spatial-temporal subset from the Copernicus Marine Data Store
using the official `copernicusmarine` Python API.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
import dotenv
import copernicusmarine



dotenv.load_dotenv()

# -----------------------------------------------------------------------------
# Default for dataset & variables
DEFAULT_DATASET_ID = "cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.0625deg_P1D"
DEFAULT_VARIABLES = ["sla", "ugos", "vgos"]
# -----------------------------------------------------------------------------


def setup_logging() -> None:
    """Configure root logger for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.  Any of the spatial/temporal flags can be
    omitted; their values will remain `None` and thus not constrain the API call.
    """
    parser = argparse.ArgumentParser(
        prog="subset_cmems",
        description="Subset Copernicus Marine data via the Python API",
    )

    # Dataset & variables
    parser.add_argument(
        "-d", "--dataset-id", default=DEFAULT_DATASET_ID, help="Dataset identifier"
    )
    parser.add_argument(
        "-v",
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        help="Variables to download (e.g. sla ugos vgos)",
    )

    # Temporal bounds (all optional)
    parser.add_argument(
        "--time-start",
        help="Start datetime (ISO format, e.g. 1993-01-01T00:00:00)",
    )
    parser.add_argument(
        "--time-end",
        help="End   datetime (ISO format, e.g. 2024-06-14T00:00:00)",
    )

    # Spatial bounds (all optional)
    parser.add_argument(
        "--min-lon",
        type=float,
        help="Minimum longitude",
    )
    parser.add_argument(
        "--max-lon",
        type=float,
        help="Maximum longitude",
    )
    parser.add_argument(
        "--min-lat",
        type=float,
        help="Minimum latitude",
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        help="Maximum latitude",
    )

    # Output options
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        required=True,
        help="Name of the output file (NetCDF or Zarr)",
    )
    parser.add_argument(
        "-O",
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to save the output file",
    )
    parser.add_argument(
        "-c",
        "--compression",
        type=int,
        choices=range(0, 10),
        default=0,
        metavar="[0-9]",
        help="NetCDF compression level",
    )
    parser.add_argument(
        "--file-format",
        choices=["netcdf", "zarr"],
        default="netcdf",
        help="Output file format",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing file",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    try:
        # Parse ISO datetimes if provided
        time_start = datetime.fromisoformat(args.time_start) if args.time_start else None
        time_end = datetime.fromisoformat(args.time_end) if args.time_end else None

    except ValueError as e:
        logger.error("Invalid datetime format: %s", e)
        sys.exit(1)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Call the subset API
    try:
        response = copernicusmarine.subset(
            dataset_id=args.dataset_id,
            variables=args.variables,
            start_datetime=time_start,
            end_datetime=time_end,
            minimum_longitude=args.min_lon,
            maximum_longitude=args.max_lon,
            minimum_latitude=args.min_lat,
            maximum_latitude=args.max_lat,
            output_directory=str(args.output_dir),
            output_filename=str(args.output_file.name),
            netcdf_compression_level=args.compression,
            file_format=args.file_format,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
        )
        logger.info("Download complete: %s", response)
    except Exception as e:
        logger.error("Data subset failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
