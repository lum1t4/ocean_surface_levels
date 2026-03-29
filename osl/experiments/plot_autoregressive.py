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
from datetime import datetime, timedelta
from osl.core.utils import IterableSimpleNamespace, yaml_load
from osl.model.registry import load_model
from torch.nn.functional import interpolate
from copy import deepcopy
from osl.data import dataset_stats_read
import os



def main(config: IterableSimpleNamespace):
    dataset = xr.open_dataset(config.dataset)
    stats = dataset_stats_read(config.stats_path)
    T = config.seq_length
    NUM_VARS = len(config.variables)
    date_start = datetime.fromisoformat(config.start_date)
    date_end = datetime.fromisoformat(config.end_date)
    window_size = (date_end - date_start).days

    device = torch.device(config.device)
    dataset = dataset.sel(time=slice(date_start - timedelta(days=T), date_end))
    model = load_model(config.model, config={'num_channels': NUM_VARS, 'num_labels': NUM_VARS}, weights=config.weights).to(device)
    model.eval()

    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    vmin = np.nanmin(dataset[config.variables[0]])
    vmax = np.nanmax(dataset[config.variables[0]])

    lon = dataset.longitude.values
    lat = dataset.latitude.values
    land_mask = np.isnan(dataset[config.variables[0]][0].values)

    assert T > 0, "Sequence length should be positive and greater than zero."
    assert window_size > 0, "Selected time range must be greather than zero."

    # -----------------------
    # Generate predictions
    # -----------------------
    inputs = {v: dataset[v][0:T].values for v in config.variables}
    for var in config.variables:
        inputs[var] = np.nan_to_num(inputs[var], nan=0.0)
        inputs[var] = (inputs[var] - stats[var]["mean"]) / stats[var]["std"]
        inputs[var] = torch.from_numpy(inputs[var]).to(device, dtype=torch.float32)

    inputs = torch.stack([inputs[v] for v in config.variables], dim=1)  # -> (T, C, H, W)
    inputs = inputs.unsqueeze(0)
    
    # -----------------------
    # AR rollout
    # -----------------------
    output = []
    with torch.inference_mode():
        for i in range(window_size):
            preds = model(inputs)
            B = preds.shape[0]
            C, H, W = preds.shape[-3:]
            preds = preds.view(B * T, C, H, W)
            preds = interpolate(preds, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            preds = preds.view(B, T, C, H, W) if T > 1 else preds
            inputs = preds
            # Take SLA (var 0) of the last prediction
            pred = preds[0, -1, 0] if T > 1 else preds[0, 0]
            pred = deepcopy(pred).cpu().numpy()

            if config.normalize:
                # Denormalize
                pred = pred * stats["sla"]["std"] + stats["sla"]["mean"]

            pred[land_mask] = np.nan
            output.append(pred)

    # -----------------------
    # Plot schema
    # -----------------------
    ground = np.ascontiguousarray(dataset[config.variables[0]][T:])

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
            title.set_text(np.datetime_as_string(dataset.time[T + i]))
            writer.grab_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a comparison video from a trained model")

    parser.add_argument("--config", action="append", default=[], help="Configuration file path")

    # Data configuration
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--variables", type=str, nargs="+", help="Variables to use from the dataset")
    parser.add_argument("--seq_length", type=int, help="Number of past steps for model input")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="Enable normalization",)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, help="Enable augmentation")

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
