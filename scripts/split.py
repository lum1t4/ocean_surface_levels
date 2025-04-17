#!/usr/bin/env python
import xarray as xr
import argparse
from sklearn.model_selection import train_test_split

def split(src, train_dst, test_dst, test_size):
    ds = xr.open_dataset(src)
    # Flatten time into samples
    sla = ds['sla']
    times = sla.time.values
    # Split time indices
    train_times, test_times = train_test_split(
        times, test_size=test_size, shuffle=False
    )
    # Select along time dimension
    ds.isel(time=[int(i) for i in range(len(times)) if times[i] in train_times]) \
      .to_netcdf(train_dst)
    ds.isel(time=[int(i) for i in range(len(times)) if times[i] in test_times]) \
      .to_netcdf(test_dst)
    print(f"Wrote {len(train_times)} train / {len(test_times)} test samples.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',       required=True)
    parser.add_argument('--train_dst', required=True)
    parser.add_argument('--test_dst',  required=True)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    split(args.src, args.train_dst, args.test_dst, args.test_size)
