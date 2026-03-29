"""
Generate a comparison video for the Dyffusion model.

AR rollout using cold sampling:
  1. Start with first frame x0
  2. Cold-sample to get all H intermediate frames [x1, ..., xH]
  3. Use xH as the new x0, repeat
Each cold-sampling call produces H frames (one per day), giving per-day resolution.

Same visual format as AR/plot.py for direct comparison.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
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
from osl.data import dataset_stats_read




def interpolate(model: nn.Module, x0: torch.Tensor, xT: torch.Tensor, step: torch.Tensor):
    return model(torch.cat([x0, xT], dim=1), step)


def forecast(model: nn.Module, x0: torch.Tensor, xi: torch.Tensor, step: torch.Tensor):
    return model(torch.cat([x0, xi], dim=1), step)


@torch.inference_mode()
def cold_sampling(interpolator: nn.Module, forecaster: nn.Module, start: torch.Tensor, horizon: int) -> torch.Tensor:
    """Dyffusion cold sampling: given x0, produce xH."""
    B = start.shape[0]

    C, H, W = start.shape[-3:]
    xi = start.clone()
    preds = np.zeros((horizon, C, H, W), dtype=np.float32)
    for i in range(horizon):
        curr = torch.full((B,), i, device=start.device, dtype=torch.long)
        next = torch.full((B,), i + 1, device=start.device, dtype=torch.long)
        x_pred = forecast(forecaster, start, xi, curr)
        interp_curr = interpolate(interpolator, start, x_pred, curr) if i > 0 else xi
        interp_next = interpolate(interpolator, start, x_pred, next) if i + 1 < horizon else x_pred
        xi = xi - interp_curr + interp_next

    preds[-1] = x_pred[0].cpu().numpy()

    # Refinement
    for i in range(horizon - 1):
        curr = torch.full((B,), i + 1, device=start.device, dtype=torch.long)
        preds[i] = interpolate(interpolator, start, x_pred, curr)[0].cpu().numpy()

    return preds


def main(config: IterableSimpleNamespace):
    dataset = xr.open_dataset(config.dataset)
    stats = dataset_stats_read(config.stats_path)
    H = config.seq_length - 1  # horizon = seq_length - 1 (matches training)
    date_start = datetime.fromisoformat(config.start_date)
    date_end = datetime.fromisoformat(config.end_date)
    window_size = (date_end - date_start).days

    device = torch.device(config.device)
    # Load enough data: we need 1 seed frame before date_start, then window_days of ground truth
    dataset = dataset.sel(time=slice(date_start - timedelta(days=1), date_end))

    # Load models
    interpolator = load_model(config.interpolator, config={'horizon': H}, weights=config.interpolator_weights).to(device)
    interpolator.eval()

    forecaster = load_model(config.forecaster, config={'horizon': H}, weights=config.forecaster_weights).to(device)
    forecaster.eval()

    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    var = config.variables[0]  # dyffusion operates on single variable (sla)
    vmin = np.nanmin(dataset[var])
    vmax = np.nanmax(dataset[var])

    lon = dataset.longitude.values
    lat = dataset.latitude.values
    land_mask = np.isnan(dataset[var][0].values)

    assert H > 0, "Horizon must be positive."
    assert window_size > 0, "Selected time range must be greater than zero."

    # -----------------------
    # Prepare initial frame
    # -----------------------
    x_start = dataset[var][0].values
    x_start = np.nan_to_num(x_start, nan=0.0)
    if config.normalize:
        x_start = (x_start - stats[var]["mean"]) / stats[var]["std"]
    x_start = torch.from_numpy(x_start).to(device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # -----------------------
    # AR rollout
    # Each cold_sampling call produces H frames (one per day)
    # -----------------------
    n_jumps = (window_size + H - 1) // H  # ceil division
    output = []

    with torch.inference_mode():
        for _ in tqdm(range(n_jumps), desc="Cold sampling", unit="step"):
            preds = cold_sampling(interpolator, forecaster, x_start, horizon=H)  # (H, C, Hs, Ws)

            for frame in preds:
                pred = frame[0].copy()  # first channel (SLA), (Hs, Ws)
                if config.normalize:
                    pred = pred * stats[var]["std"] + stats[var]["mean"]
                pred[land_mask] = np.nan
                output.append(pred)

            # Slide window: last predicted frame becomes the new x0
            x_start = torch.from_numpy(preds[-1]).to(device, dtype=torch.float32).unsqueeze(0)  # (1, C, Hs, Ws)

    # -----------------------
    # Plot schema
    # -----------------------
    # Ground truth: skip the seed frame (index 0), take the rest
    # output[i] corresponds to day (i+1) after the seed, 1:1 with ground truth
    ground = dataset[config.variables[0]][1:].values

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 9),
        dpi=150,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for ax in axes:
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.axis("off")

    axes[0].set_title("Original (SLA)")
    axes[1].set_title("Generated (SLA) — Dyffusion")

    mesh0 = axes[0].pcolormesh(lon, lat, np.zeros_like(ground[0]), vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance, transform=ccrs.PlateCarree())
    mesh1 = axes[1].pcolormesh(lon, lat, np.zeros_like(ground[0]), vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance, transform=ccrs.PlateCarree())

    title = fig.suptitle("")

    writer = FFMpegWriter(
        fps=10, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "fast"]
    )

    n_frames = min(len(output), len(ground))

    with writer.saving(fig, destination.as_posix(), dpi=150):
        for i in tqdm(range(n_frames), desc="Rendering frames", unit="frame"):
            mesh0.set_array(ground[i].ravel())
            mesh1.set_array(output[i].ravel())
            title.set_text(np.datetime_as_string(dataset.time[1 + i]))
            writer.grab_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comparison video from a trained Dyffusion model"
    )

    parser.add_argument("--config", action="append", default=[], help="Configuration file path")

    # Data configuration
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--variables", type=str, nargs="+", help="Variables to use")
    parser.add_argument("--seq_length", type=int, help="Sequence length (horizon = seq_length - 1)")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="Enable normalization")

    # Model configuration
    parser.add_argument("--interpolator", type=str, help="Interpolator model name")
    parser.add_argument("--forecaster", type=str, help="Forecaster model name")
    parser.add_argument("--interpolator_weights", type=str, required=True, help="Path to interpolator weights")
    parser.add_argument("--forecaster_weights", type=str, required=True, help="Path to forecaster weights")

    # Time range
    parser.add_argument("--start_date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, help="End date (YYYY-MM-DD)")

    # Output configuration
    parser.add_argument("--output", "--dst", dest="output", type=str, help="Output video path")

    # System configuration
    parser.add_argument("--device", type=str, help="Device identifier")

    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}

    config = args.pop("config", None)
    if config:
        base = yaml_load(config)
        args = {**base, **args}

    main(IterableSimpleNamespace(**args))
