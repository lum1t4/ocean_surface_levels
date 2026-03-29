"""
# A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting [NeurIPS 2023] 
References:
 - https://github.com/Rose-STL-Lab/dyffusion
 - https://arxiv.org/pdf/2306.01984

Dyffusion is **inspired** from diffusion models but it really differs from them.

1. The first phase is to train an interpolator which is a neural network that models
a function $I(x_{t}, x_{t+h}, i) = x_{t+i}$ where:
    - h is horizon (sequence length - 1).
    - i is time step an integer $[1, h - 1]$ used to pick one of the element of
    the sequence between the start and the end.
Note: The interpolator might use, in inference, a real valued time step i to estimate an
intermidiate representation which might not be present in the original discrete
sequence emulating a continuos sampling from the sequence.

2. The second phase is to train a forecaster which is a neural network that models
a function $F(x_{t}, x_{t+j}, j)$ that should approximate $x_{t+h}$, j is an integer $[0, h - 1]$
The forecaster uses the interpolator to generate intermidiate representations
between $x_{t}$ and $x_{t+h}$ to condition its forecast. $F(I(x_{t}, x_{t + h}, i), i) = x_{t+h}$


3. Cold Sampling / Inference
xi = x0.clone()
for i in range(0, H):
    xH = F(x0, xi, i)
    xi = I(x0, xH, i + 1) - I(x0, xH, i) + xi

where I(x0, xH, 0) = x0 and I(x0, xH, H) = xH

# TODO: In the original code base they support sampling steps which are more than
# sequence length and are generally added before predicting x1 from x1 hence learning
# some intermidiate rappresentetion between x0 and x1 and that seems to be working
"""


from torch.utils.data import DataLoader
import xarray as xr
from osl.core.pytorch import RANK, device_memory_used
from osl.core.train import TrainContext
from osl.core.utils import LOGGER, IterableSimpleNamespace, yaml_load
from osl.data import SequenceDataset, dataset_stats_load
from osl.core.metrics import ssim_metric
import math
import numpy as np
import tqdm
from torch import nn, optim
import torch
from osl.model import load_model
import matplotlib
import matplotlib.pyplot as plt
import cmocean


matplotlib.use("Agg")


def interpolate(model, x0: torch.Tensor, xT: torch.Tensor, step: torch.Tensor):
    return model(torch.cat([x0, xT], dim=1), step)


def forecast(model, x0: torch.Tensor, xi: torch.Tensor, step: torch.Tensor):
    return model(torch.cat([x0, xi], dim=1), step)


def inference_dropout_enable(model: nn.Module):
    """Set all dropout layers to training mode"""
    # find all dropout layers
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d)]
    for layer in dropout_layers:
        layer.train()


def inference_dropout_disable(model: nn.Module):
    """Set all dropout layers to eval mode"""
    # find all dropout layers
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d)]
    for layer in dropout_layers:
        layer.eval()


def model_train_step(ctx: TrainContext, batch_idx: int, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch['inputs']
    B, T, _, _, _ = inputs.shape
    # Take a random time step for each sequence
    ts = torch.randint(0, T - 1, (B,))
    # First and last item for each sequence
    x0, xT = inputs[:, 0], inputs[:, -1]

    x0 = x0.to(ctx.device, non_blocking=True, dtype=torch.float32)
    ts = ts.to(ctx.device, non_blocking=True, dtype=torch.long)
    xT = xT.to(ctx.device, non_blocking=True, dtype=torch.float32)

    # xi is the interpolated item at a specific timestep
    # xi is initialized to x0 (initially assuming steps 0)
    xi = x0.clone()
    mask = ts > 0

    with torch.inference_mode():
        inference_dropout_enable(ctx.interpolator)
        xi[mask] = interpolate(ctx.interpolator, x0[mask], xT[mask], ts[mask])
        inference_dropout_disable(ctx.interpolator)

    outputs = forecast(ctx.model, x0, xi, ts)
    loss = torch.nn.functional.mse_loss(outputs, xT)
    return outputs, loss


@torch.inference_mode()
def model_generate(interpolator: nn.Module, forecaster: nn.Module, start: torch.Tensor, horizon: int) -> torch.Tensor:
    """Return all intermediate forecasts [x1, ..., xH] from x0=start."""
    B = start.shape[0]
    xi = start.clone()
    x_pred = xi

    for i in range(horizon):
        curr = torch.full((B,), i, device=start.device, dtype=torch.long)
        next = torch.full((B,), i + 1, device=start.device, dtype=torch.long)
        x_pred = forecast(forecaster, start, xi, curr)
        interp_curr = interpolate(interpolator, start, x_pred, curr) if i > 0 else xi
        interp_next = interpolate(interpolator, start, x_pred, next) if i + 1 < horizon else x_pred
        xi = xi - interp_curr + interp_next

    # Refinement
    outputs = []
    for i in range(horizon - 1):
        curr = torch.full((B,), i + 1, device=start.device, dtype=torch.long)
        outputs.append(interpolate(interpolator, start, x_pred, curr))

    outputs.append(x_pred)
    return torch.stack(outputs, dim=1)


def model_valid_step(ctx: TrainContext, batch_idx: int, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["inputs"].to(ctx.device, non_blocking=True, dtype=torch.float32)  # (B, 1, C, H, W)
    targets = batch["targets"].to(ctx.device, non_blocking=True, dtype=torch.float32)  # (B, G, C, H, W)
    G = targets.shape[1]
    H = ctx.config.seq_length - 1
    outputs = []
    chuncks = math.ceil(G / H)

    inputs = inputs.squeeze(1)
    for chunck_idx in range(chuncks):
        chunck_length = min(H, G - chunck_idx * H)
        chunck = model_generate(ctx.interpolator, ctx.model, inputs, H)
        chunck = chunck[:, :chunck_length]
        outputs.append(chunck)
        inputs = chunck[:, -1]
    
    outputs = torch.concat(outputs, dim=1)
    return outputs, targets


def plot_valid_batch(ctx: TrainContext, outputs: torch.Tensor, targets: torch.Tensor):
    """Plot a single validation sample: target vs prediction at selected lead days.

    Args:
        ctx: Training context (used for plot_dir, curr_iter, config).
        outputs: (B, T, C, H, W) predictions in normalized space.
        targets: (B, T, C, H, W) ground truth in normalized space.
    """
    lead_days = [1, 7, 14, 30]
    T = targets.shape[1]
    lead_days = [d for d in lead_days if d <= T]
    ncols = len(lead_days)

    # Take first sample, first variable (SLA)
    pred = outputs[0, :, 0].cpu().numpy()   # (T, H, W)
    true = targets[0, :, 0].cpu().numpy()   # (T, H, W)

    land_mask = ctx.valid_set.get_land_mask().numpy()
    vmin = np.nanmin(true)
    vmax = np.nanmax(true)

    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(4 * ncols, 7))
    if ncols == 1:
        axes = axes[:, None]

    for j, day in enumerate(lead_days):
        idx = day - 1
        t = true[idx].copy()
        p = pred[idx].copy()
        t[~land_mask] = np.nan
        p[~land_mask] = np.nan

        mse = np.nanmean((pred[idx] - true[idx]) ** 2)

        axes[0, j].imshow(t, vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance, origin="upper")
        axes[0, j].set_title(f"Target day {day}")
        axes[0, j].axis("off")

        axes[1, j].imshow(p, vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance, origin="upper")
        axes[1, j].set_title(f"Pred day {day}\nMSE={mse:.6f}")
        axes[1, j].axis("off")

    fig.tight_layout()
    dst = ctx.plot_dir / f"valid_epoch_{ctx.curr_iter}.png"
    fig.savefig(dst, dpi=120)
    plt.close(fig)
    return dst


def schedule_setup_dataset(ctx: TrainContext, config: IterableSimpleNamespace):
    # Data, dataset and dataloader setup
    LOGGER.info(f"Loading dataset from: {config.dataset}")
    data = xr.open_dataset(config.dataset)
    stats = dataset_stats_load(config.stats_path, data, config.variables, overwrite=True)

    train_fold = config.train
    valid_fold = config.valid

    train_bound = [int(train_fold[0]), int(train_fold[1])]
    valid_bound = [int(valid_fold[0]), int(valid_fold[1])]

    LOGGER.info(
        f"Train set size: {train_bound[1] - train_bound[0]} "
        f"Valid set size: {valid_bound[1] - valid_bound[0]} "
    )

    reader_args = dict(vars=config.variables, seq_length=config.seq_length, seq_stride=config.seq_length, normalize=config.normalize, stats=stats)
    roller_args = {**reader_args, **dict(seq_seek=1, seq_length=30, seq_stride=30)}
    loader_args = dict(batch_size=config.batch_size, num_workers=config.workers, pin_memory=False)

    if ctx.device.type not in {'cpu', 'mps'}:
        loader_args["pin_memory"] = True

    ctx.train_set = SequenceDataset(data, bound=train_bound, **reader_args)
    ctx.valid_set = SequenceDataset(data, bound=valid_bound, **roller_args)

    ctx.train_loader = DataLoader(ctx.train_set, shuffle=True, **loader_args)
    ctx.valid_loader = DataLoader(ctx.valid_set, shuffle=False, drop_last=False, **loader_args)

    return ctx


def schedule_train_epoch(ctx: TrainContext):
    ctx.model.train()
    running_loss = 0.0
    running_norm = 0.0
    progress = enumerate(ctx.train_loader)
    if RANK in {-1, 0}:
        LOGGER.info(("\n" + "%11s" * 4) % ("Epoch", "GPU_mem", "Loss", "Norm"))
        progress = tqdm.tqdm(progress, total=len(ctx.train_loader))

    ctx.optimizer.zero_grad()
    for batch_idx, batch in progress:
        _, loss = model_train_step(ctx, batch_idx, batch)
        loss /= ctx.config.grad_acc
        loss.backward()
        if (batch_idx + 1) % ctx.config.grad_acc == 0:
            norm = torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), max_norm=1.0)
            ctx.optimizer.step()
            ctx.optimizer.zero_grad()
            step_idx = (batch_idx + 1) // ctx.config.grad_acc
            running_norm = (running_norm * step_idx + norm.item()) / (step_idx + 1)
            running_loss = (running_loss * step_idx + loss.item()) / (step_idx + 1)

        if RANK in {-1, 0}:
            epoch_desc = f"{ctx.curr_iter + 1}/{ctx.config.epochs}"
            memory_used = device_memory_used(ctx.device)
            progress.set_description("%11s%11.4g%11.4g%11.4g" % (epoch_desc, memory_used, running_loss, running_norm))

    ctx.metrics["train/loss"] = running_loss
    ctx.metrics["train/norm"] = running_norm


@torch.inference_mode()
def schedule_valid_epoch(ctx: TrainContext):
    ctx.model.eval()
    if hasattr(ctx, "interpolator"):
        ctx.interpolator.eval()

    progress = enumerate(ctx.valid_loader)
    if RANK in {-1, 0}:
        LOGGER.info("%11s" % "Val. Loss")
        progress = tqdm.tqdm(progress, total=len(ctx.valid_loader))

    lead_days = [1, 7, 30]
    running_loss = 0.0
    running_leads = torch.zeros((len(lead_days)), device=ctx.device)
    running_leads_ssim = torch.zeros((len(lead_days)), device=ctx.device)
    leads_m2 = torch.zeros((len(lead_days)), device=ctx.device)

    plotted = None
    for batch_idx, batch in progress:
        outputs, targets = model_valid_step(ctx, batch_idx, batch)
        loss = nn.functional.mse_loss(outputs, targets)
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)

        if RANK in {-1, 0}:
            progress.set_description("%11.4g" % running_loss)
            if batch_idx == 0:
                plotted = plot_valid_batch(ctx, outputs, targets)

        for j, lead_day in enumerate(lead_days):
            pred_frame = outputs[:, lead_day - 1]
            tgt_frame = targets[:, lead_day - 1]
            lead_loss = nn.functional.mse_loss(pred_frame, tgt_frame)
            lead_ssim = ssim_metric(pred_frame, tgt_frame)
            old_mean = running_leads[j]
            running_leads[j] = (old_mean * batch_idx + lead_loss) / (batch_idx + 1)
            leads_m2[j] += (lead_loss - old_mean) * (lead_loss - running_leads[j])
            running_leads_ssim[j] = (running_leads_ssim[j] * batch_idx + lead_ssim) / (batch_idx + 1)

    n_batches = len(ctx.valid_loader)
    for j, lead_day in enumerate(lead_days):
        mse_mean = running_leads[j].item()
        mse_std = torch.sqrt(leads_m2[j] / max(n_batches - 1, 1)).item()
        rmse_mean = math.sqrt(mse_mean)
        rmse_std = mse_std / (2 * rmse_mean) if rmse_mean > 0 else 0.0
        ctx.metrics[f"valid/MSE@{lead_day}"] = mse_mean
        ctx.metrics[f"valid/RMSE@{lead_day}"] = rmse_mean
        ctx.metrics[f"valid/RMSE_std@{lead_day}"] = rmse_std
        ctx.metrics[f"valid/SSIM@{lead_day}"] = running_leads_ssim[j].item()

    if plotted is not None and RANK in {-1, 0}:
        LOGGER.info(f"Saved validation plot to {plotted}")
    ctx.metrics["valid/loss"] = running_loss
    ctx.tracker.log(ctx.metrics, step=ctx.curr_iter)


def main(config: IterableSimpleNamespace):
    ctx = TrainContext(config)
    # Load dataset
    schedule_setup_dataset(ctx, config)

    # Load model
    model_ovverides = {"horizon": ctx.config.seq_length - 1}
    ctx.model = load_model(ctx.config.model, config=model_ovverides).to(ctx.device)
    ctx.metrics["model/params"] = sum(p.numel() for p in ctx.model.parameters() if p.requires_grad)

    ctx.interpolator = load_model(ctx.config.interpolator, config={"horizon": ctx.config.seq_length - 1}, weights=ctx.config.interpolator_weights).to(ctx.device)
    # Freeze interpolator (it must be pretrained)
    ctx.interpolator.eval()
    for p in ctx.interpolator.parameters():
        p.requires_grad = False

    # Optimization comp
    ctx.criterion = nn.MSELoss()
    ctx.optimizer = optim.Adam(ctx.model.parameters(), lr=ctx.config.lr, weight_decay=ctx.config.weight_decay)
    ctx.scheduler = optim.lr_scheduler.ReduceLROnPlateau(ctx.optimizer, mode="min", factor=0.5, patience=5)

    LOGGER.info(f"Model: {config.model}, Params: {ctx.metrics['model/params']:,}")
    LOGGER.info(f"Criterion: {ctx.config.criterion}")
    LOGGER.info(f"Device: {ctx.device}")
    LOGGER.info(f"Training for {ctx.config.epochs} epochs...")

    # Train loop
    for epoch in range(ctx.start_iter, ctx.config.epochs):
        ctx.curr_iter = epoch
        ctx.metrics = {}
        schedule_train_epoch(ctx)
        schedule_valid_epoch(ctx)
        ctx.checkpointing()
        ctx.iteration_end()
        ctx.early_stopping()
        if ctx.stop:
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ocean currents predictor")
    parser.add_argument("--config", action="append", default=[], help="Configuration file path")
    parser.add_argument("--name", type=str, help="Run name")
    parser.add_argument("--tracker", type=str, help="Tracker")

    # Data configuration
    parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    parser.add_argument("--seq_length", type=int, help="Number of past steps")
    parser.add_argument("--seq_stride", type=int, help="Stride for sequence sampling")
    parser.add_argument("--variables", type=str, nargs="+", help="Variables to use from the dataset")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="Enable normalization")
    parser.add_argument("--modality", type=str, choices=["regressive", "progressive", "dyffusion"], default="regressive")

    # Training configuration
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for optimizer")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--patience", type=int, help="Patience for early stopping")
    parser.add_argument("--grad_acc", "--gradient_accumulation", type=int, help="Gradient accumulation steps")
    parser.add_argument("--criterion", type=str, help="Loss criterion name")

    # System configuration
    parser.add_argument("--workers", type=int, help="Number of dataloader worker processes")
    parser.add_argument("--device", type=str, help="Device identifier (e.g., 'cuda:0', 'cpu', 'mps:0')")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, help="Enable deterministic behavior")
    parser.add_argument("--save_dir", type=str, help="Directory for saving checkpoints")

    # Modality specifics
    # Regressive specific
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction)
    parser.add_argument("--augment_noise_start", type=float)
    parser.add_argument("--augment_noise_end", type=float)
    parser.add_argument("--augment_noise_schedule", type=str)

    # Progressive specific
    parser.add_argument("--prediction_type", type=str, choices=["sample", "epsilon"], help="Predict x_t or epsilon (xt - x0)")
    # Dyffusion specific
    parser.add_argument("--stage")  # interpolator | forecaster
    parser.add_argument("--interpolator", type=str, help="Interpolator model name")
    parser.add_argument("--interpolator_weights", type=str, help="Interpolator model name")

    # Plot video
    parser.add_argument("--start_date", "--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", "--end", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    args = {k: v for k, v in vars(args).items() if v is not None}
    configs = args.pop("config")
    base = {}
    for config in configs:
        base.update(yaml_load(config))

    args = {**base, **args}  # Command-line args override config file
    config = IterableSimpleNamespace(**args)
    main(config)
