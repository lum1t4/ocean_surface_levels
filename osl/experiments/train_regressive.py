from torch.utils.data import DataLoader
import xarray as xr
from osl.core.pytorch import RANK, device_memory_used
from osl.core.train import TrainContext
from osl.core.utils import LOGGER, IterableSimpleNamespace, yaml_load
from osl.data import SequenceDataset, dataset_stats_load
from osl.core.metrics import ssim_metric

import numpy as np
import tqdm
from torch import nn, optim
import torch
from osl.model import load_model
import matplotlib
import matplotlib.pyplot as plt
import cmocean

import math


matplotlib.use("Agg")


def apply_augmentation(context: TrainContext, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Apply augmentation using noise injection based on scheduled sampling
    # Rate is computed on epoch and applied to batches (not to single samples)
    def noise_rate_fn(epoch: int, epochs: int, start: float = 1.0, end: float = 0.0, schedule: str = 'linear'):
        progress = epoch / max(epochs - 1, 1)
        ratio = 0.0
        if schedule == "linear":
            ratio = start + progress * (end - start)
        if schedule == "exponential":
            ratio = start * math.exp(-math.log(max(end, 1e-6) / start) * progress)
        if schedule == "cosine":
            ratio = end + (start - end) * (1 + math.cos(math.pi * progress)) / 2
        return max(ratio, end)

    if not context.config.augment:
        return inputs, targets

    ratio = noise_rate_fn(
        context.curr_iter, context.config.epochs,
        context.config.augment_noise_start,
        context.config.augment_noise_end,
        context.config.augment_noise_schedule,
    )

    if ratio >= 1.0:
        return inputs, targets

    # Inject noise to simulate prediction errors
    # Noise std is proportional to input std and (1 - ratio)
    noise_scale = (1.0 - ratio) * 0.1  # Max 10% noise at tf=0
    noise = torch.randn_like(inputs, device=inputs.device) * inputs.std() * noise_scale

    # Apply noise more heavily to later frames (which would accumulate more error)
    B, T, C, H, W = inputs.shape
    temporal_weight = torch.linspace(0.2, 1.0, T, device=inputs.device).view(1, T, 1, 1, 1)
    noise = noise * temporal_weight
    return inputs + noise, targets


def model_train_step(ctx: TrainContext, batch_idx: int, batch: dict):
    sequence: torch.Tensor = batch["inputs"].to(ctx.device, non_blocking=True, dtype=torch.float32)  # (B, T + 1, C, H, W)

    inputs = sequence[:, :-1].contiguous()  # (B, T, C, H, W)
    targets = sequence[:, 1:].contiguous()  # (B, T, C, H, W)

    if ctx.model.training:
        inputs, targets = apply_augmentation(ctx, inputs, targets)

    outputs = ctx.model(inputs)
    loss = nn.functional.mse_loss(outputs, targets)
    return outputs, loss


def model_valid_step(ctx: TrainContext, batch_idx: int, batch: dict):
    inputs = batch["inputs"].to(ctx.device, non_blocking=True, dtype=torch.float32)  # (B, T, C, H, W)
    targets = batch["targets"].to(ctx.device, non_blocking=True, dtype=torch.float32)  # (B, T', C, H, W)
    window_size = targets.shape[1] # T'
    outputs = torch.zeros_like(targets)

    for i in range(window_size):
        predictions = ctx.model(inputs)

        # Interpolate
        B, T, C, H, W = predictions.shape
        predictions = predictions.view(B * T, C, H, W)
        predictions = nn.functional.interpolate(predictions, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        predictions = predictions.view(B, T, C, H, W)
        outputs[:, i] = predictions[:, -1]
        inputs = predictions

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

    reader_args = dict(vars=config.variables, seq_length=config.seq_length + 1, seq_stride=config.seq_length + 1, normalize=config.normalize, stats=stats)
    roller_args = {**reader_args, **dict(seq_seek=config.seq_length, seq_length=30, seq_stride=30)}
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

    progress = enumerate(ctx.valid_loader)
    if RANK in {-1, 0}:
        LOGGER.info("%11s" % "Val. Loss")
        progress = tqdm.tqdm(progress, total=len(ctx.valid_loader))

    lead_days = [1, 7, 30]
    running_loss = 0.0
    running_leads = torch.zeros((len(lead_days)), device=ctx.device)
    running_leads_ssim = torch.zeros((len(lead_days)), device=ctx.device)
    # Welford's online variance for per-lead RMSE std
    leads_m2 = torch.zeros((len(lead_days)), device=ctx.device)

    plotted = None
    for batch_idx, batch in progress:
        outputs, targets = model_valid_step(ctx, batch_idx, batch)# B, T, C, H, W
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
        # Delta method: std(sqrt(X)) ≈ std(X) / (2 * sqrt(mean(X)))
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
    num_vars = len(ctx.config.variables)  # e.g. SLA, UGOS, VGOS
    model_ovverides = {
        "num_channels": num_vars,
        "num_labels": num_vars,
        "in_channels": num_vars,
        "out_channels": num_vars,
    }
    ctx.model = load_model(ctx.config.model, config=model_ovverides).to(ctx.device)
    ctx.metrics["model/params"] = sum(p.numel() for p in ctx.model.parameters() if p.requires_grad)

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
