"""
This script is used to generate a video to view model predictions
and compare them with real data.

Since training takes about 80 of the data to better evaluate model
predictions it is preferable to plot starting from date after '2018-03-01'

"""

import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from matplotlib.animation import FFMpegWriter
from tqdm.auto import tqdm
from datetime import datetime
from osl.core.utils import IterableSimpleNamespace, yaml_load
from osl.model.registry import load_model
import pandas as pd
from torch.nn.functional import interpolate
from copy import deepcopy

def main(config: IterableSimpleNamespace):
    dataset = xr.open_dataset(config.dataset)
    
    device = torch.device(config.device)
    dataset = dataset.sel(time=slice(
        datetime.fromisoformat(config.start_date),
        datetime.fromisoformat(config.end_date)
    ))

    model = load_model(config.model, config={'num_labels': 3}, weights=config.weights).to(device)
    model.eval()

    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    vmin = np.nanmin(dataset[config.variables[0]])
    vmax = np.nanmax(dataset[config.variables[0]])
    lon = dataset.longitude.values
    lat = dataset.latitude.values

    T = config.seq_length
    window_size = len(dataset.time)


    assert T > 0, "Sequence length should be positive and greater than zero."
    assert window_size > T, "Selected time range must be longer than sequence length."

    # -----------------------
    # Generate predictions
    # -----------------------
    inputs = [dataset[v][0:T] for v in config.variables]
    inputs = np.nan_to_num(np.stack(inputs, 1), nan=0.0) # -> (T, C, H, W)
    inputs = torch.from_numpy(inputs).to(device, dtype=torch.float32)
    inputs = inputs.unsqueeze(0) if T > 1 else inputs

    output = []
    with torch.inference_mode():
        for i in range(window_size - T):
            preds = model(inputs)

            B = preds.shape[0]
            C, H, W = preds.shape[-3:]
            preds = preds.view(B * T, C, H, W)
            preds = interpolate(preds, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            preds = preds.view(B, T, C, H, W) if T > 1 else preds

            inputs = preds
            prediction = preds[0, -1, 0] if T > 1 else preds[0, 0]
            prediction = deepcopy(prediction).cpu().numpy()
            output.append(prediction)

    # -----------------------
    # Plot schema
    # -----------------------
    ground = np.ascontiguousarray(dataset.sla[T:])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), dpi=150, subplot_kw={"projection": ccrs.PlateCarree()})
    for ax in axes:
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.axis("off")

    axes[0].set_title("Original (SLA)")
    axes[1].set_title("Generated (SLA)")

    mesh0 = axes[0].pcolormesh(lon, lat, np.zeros_like(ground[0]), vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance, transform=ccrs.PlateCarree())
    mesh1 = axes[1].pcolormesh(lon, lat, np.zeros_like(ground[0]), vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance, transform=ccrs.PlateCarree())

    title = fig.suptitle("")

    writer = FFMpegWriter(fps=10, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "fast"])
    with writer.saving(fig, destination.as_posix(), dpi=150):
        for i in tqdm(range(window_size - T), desc="Rendering frames", unit="frame"):
            mesh0.set_array(ground[i].ravel())
            mesh1.set_array(output[i].ravel())
            title.set_text(np.datetime_as_string(dataset.time[i]))
            writer.grab_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a comparison video from a trained model")

    parser.add_argument("--config", type=str, help="Configuration file path")

    # Data configuration
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--variables", type=str, nargs="+", help="Variables to use from the dataset")
    parser.add_argument("--seq_length", type=int, help="Number of past steps for model input")

    # Model configuration
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--weights", type=str, help="Path to model weights")
    parser.add_argument("--batch_size", type=int, help="Batch size for inference")

    # Time range
    parser.add_argument("--start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date (YYYY-MM-DD)")

    # Output configuration
    parser.add_argument("--output", "--dst", dest="output", type=str, help="Output video path")

    # System configuration
    parser.add_argument("--device", type=str, help="Device identifier (e.g., 'cuda:0', 'cpu', 'mps:0')")

    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}

    config = args.pop("config", None)
    if config:
        base = yaml_load(config)
        args = {**base, **args}

    main(IterableSimpleNamespace(**args))
