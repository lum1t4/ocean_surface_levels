import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch
import xarray as xr
from tqdm.auto import tqdm

from osl.core.utils import IterableSimpleNamespace, yaml_load
from osl.data import dataset_stats_read
from osl.model.registry import load_model


def predict_sample(
    model: torch.nn.Module,
    x0: torch.Tensor,
    step: torch.Tensor,
    target_mode: str,
) -> torch.Tensor:
    pred = model(x0, step)
    return pred + x0 if target_mode == "delta" else pred


@torch.inference_mode()
def rollout_predictor(
    model: torch.nn.Module,
    start: torch.Tensor,
    horizon: int,
    target_mode: str,
) -> np.ndarray:
    batch_size = start.shape[0]
    channels, height, width = start.shape[-3:]
    preds = np.zeros((horizon, channels, height, width), dtype=np.float32)
    for i in range(1, horizon + 1):
        step = torch.full((batch_size,), i, device=start.device, dtype=torch.long)
        x_pred = predict_sample(model, start, step, target_mode=target_mode)
        preds[i - 1] = x_pred[0].cpu().numpy()
    return preds


def main(config: IterableSimpleNamespace):
    dataset = xr.open_dataset(config.dataset)
    stats = dataset_stats_read(config.stats_path)
    horizon = config.seq_length - 1
    date_start = datetime.fromisoformat(config.start_date)
    date_end = datetime.fromisoformat(config.end_date)
    window_size = (date_end - date_start).days

    if horizon <= 0:
        raise ValueError("seq_length must be >= 2")
    if window_size <= 0:
        raise ValueError("end_date must be after start_date")

    device = torch.device(config.device)
    dataset = dataset.sel(time=slice(date_start - timedelta(days=1), date_end))

    num_vars = len(config.variables)
    model = load_model(
        config.model,
        config={"in_channels": num_vars, "out_channels": num_vars},
        weights=config.weights,
    ).to(device)
    model.eval()

    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Visualization currently targets first variable
    var = config.variables[0]
    vmin = np.nanmin(dataset[var])
    vmax = np.nanmax(dataset[var])
    lon = dataset.longitude.values
    lat = dataset.latitude.values
    land_mask = np.isnan(dataset[var][0].values)

    x_start = dataset[var][0].values
    x_start = np.nan_to_num(x_start, nan=0.0)
    if config.normalize:
        x_start = (x_start - stats[var]["mean"]) / stats[var]["std"]
    x_start = (
        torch.from_numpy(x_start)
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    n_jumps = (window_size + horizon - 1) // horizon
    output = []
    with torch.inference_mode():
        for _ in tqdm(range(n_jumps), desc="Predictor rollout", unit="step"):
            preds = rollout_predictor(
                model,
                start=x_start,
                horizon=horizon,
                target_mode=config.target_mode,
            )

            for frame in preds:
                pred = frame[0].copy()
                if config.normalize:
                    pred = pred * stats[var]["std"] + stats[var]["mean"]
                pred[land_mask] = np.nan
                output.append(pred)

            x_start = (
                torch.from_numpy(preds[-1])
                .to(device=device, dtype=torch.float32)
                .unsqueeze(0)
            )

    ground = dataset[var][1:].values

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 9),
        dpi=150,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for ax in axes:
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
        ax.axis("off")

    axes[0].set_title("Original (SLA)")
    axes[1].set_title(f"Generated (SLA) — Predictor [{config.target_mode}]")

    mesh0 = axes[0].pcolormesh(
        lon,
        lat,
        np.zeros_like(ground[0]),
        vmin=vmin,
        vmax=vmax,
        cmap=cmocean.cm.balance,
        transform=ccrs.PlateCarree(),
    )
    mesh1 = axes[1].pcolormesh(
        lon,
        lat,
        np.zeros_like(ground[0]),
        vmin=vmin,
        vmax=vmax,
        cmap=cmocean.cm.balance,
        transform=ccrs.PlateCarree(),
    )

    title = fig.suptitle("")
    writer = FFMpegWriter(
        fps=10,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-preset", "fast"],
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
        description="Generate comparison video from a trained predictor model"
    )
    parser.add_argument("--config", action="append", default=[], help="Configuration file path")

    # Data configuration
    parser.add_argument("--dataset", type=str, help="Path to dataset")
    parser.add_argument("--variables", type=str, nargs="+", help="Variables to use")
    parser.add_argument("--seq_length", type=int, help="Sequence length")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="Enable normalization")
    parser.add_argument("--stats_path", type=str, help="Path to stats.json")

    # Model configuration
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--target_mode", type=str, choices=["sample", "delta"], help="Predict x_t or delta")

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
