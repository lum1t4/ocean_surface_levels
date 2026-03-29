"""
Direction 1a: Progressive training with Beta-weighted timestep sampling.

Instead of sampling timesteps uniformly, we use Beta(alpha, beta) distribution
to bias toward later (harder) timesteps, and weight the loss by sqrt(t/T) so
that long-horizon errors get more gradient signal.

# Example run (test)
uv run osl/experiments/pixel/train_progressive_beta.py \
    --config config/pixel/base.yml config/pixel/progressive.yml \
    --epochs 1 --device cuda:0 --batch_size 4 --name "beta_test" \
    --seq_length 8 --variables sla --normalize --prediction_type sample \
    --beta_a 2.0 --beta_b 1.0

# Example run (tracked)
uv run osl/experiments/pixel/train_progressive_beta.py \
    --config config/pixel/base.yml config/pixel/progressive.yml \
    --epochs 100 --device cuda:0 --batch_size 4 --name "progressive_beta_unet_008" \
    --model unet-time --seq_length 8 --variables sla --normalize \
    --prediction_type sample --beta_a 2.0 --beta_b 1.0 --tracker wandb
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


def model_train_step(ctx: TrainContext, batch_idx: int, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = batch["inputs"]
    B, T = inputs.shape[:2]

    # Beta distribution sampling: biases toward later timesteps when beta_a > beta_b
    beta_a = getattr(ctx.config, 'beta_a', 2.0)
    beta_b = getattr(ctx.config, 'beta_b', 1.0)
    beta_dist = torch.distributions.Beta(beta_a, beta_b)
    ts = beta_dist.sample((B,))  # in [0, 1], biased toward 1.0 when a > b
    ts = (ts * (T - 1)).long().clamp(1, T - 1)  # map to [1, T-1]

    x0 = inputs[:, 0]
    xt = inputs[torch.arange(B), ts]

    ts = ts.to(ctx.device, non_blocking=True, dtype=torch.long)
    x0 = x0.to(ctx.device, non_blocking=True, dtype=torch.float32)
    xt = xt.to(ctx.device, non_blocking=True, dtype=torch.float32)

    outputs = ctx.model(x0, ts)
    if ctx.config.prediction_type == "epsilon":
        per_sample_loss = nn.functional.mse_loss(outputs, xt - x0, reduction="none").mean(dim=(1, 2, 3))
        outputs = x0 + outputs
    else:
        per_sample_loss = ctx.criterion(outputs, xt).mean(dim=(1, 2, 3))  # (B,)

    # Per-sample loss weighted by sqrt(t/T) — later days get more gradient
    weights = (ts.float() / T).sqrt()
    loss = (per_sample_loss * weights).mean()

    return outputs, loss


@torch.inference_mode()
def model_generate(model: nn.Module, inputs: torch.Tensor, horizion: int, prediction_type: str = "sample") -> torch.Tensor:
    batch_size = inputs.shape[0]
    outputs = []
    for i in range(1, horizion + 1):
        step = torch.full((batch_size,), i, device=inputs.device, dtype=torch.long)
        prediction = model(inputs, step)
        if prediction_type == "epsilon":
            prediction = prediction + inputs
        outputs.append(prediction)
    return torch.stack(outputs, dim=1)


def model_valid_step(ctx: TrainContext, batch_idx: int, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["inputs"].to(ctx.device, non_blocking=True, dtype=torch.float32)
    targets = batch["targets"].to(ctx.device, non_blocking=True, dtype=torch.float32)
    G = targets.shape[1]
    H = ctx.config.seq_length - 1
    outputs = []
    chuncks = math.ceil(G / H)

    inputs = inputs.squeeze(1)
    for chunck_idx in range(chuncks):
        chunck_length = min(H, G - chunck_idx * H)
        chunck = model_generate(ctx.model, inputs, chunck_length, prediction_type=ctx.config.prediction_type)
        outputs.append(chunck)
        inputs = chunck[:, -1]

    outputs = torch.concat(outputs, dim=1)
    return outputs, targets


def plot_valid_batch(ctx: TrainContext, outputs: torch.Tensor, targets: torch.Tensor):
    lead_days = [1, 7, 14, 30]
    T = targets.shape[1]
    lead_days = [d for d in lead_days if d <= T]
    ncols = len(lead_days)

    pred = outputs[0, :, 0].cpu().numpy()
    true = targets[0, :, 0].cpu().numpy()

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

    reader_args = dict(
        vars=config.variables,
        seq_length=config.seq_length,
        seq_stride=config.seq_stride,
        normalize=config.normalize,
        stats=stats,
    )
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

    progress = enumerate(ctx.valid_loader)
    if RANK in {-1, 0}:
        LOGGER.info("%11s" % "Val. Loss")
        progress = tqdm.tqdm(progress, total=len(ctx.valid_loader))

    lead_days = [1, 7, 30]
    running_loss = 0.0
    running_leads = torch.zeros((len(lead_days)), device=ctx.device)
    running_leads_ssim = torch.zeros((len(lead_days)), device=ctx.device)
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
            running_leads[j] = (running_leads[j] * batch_idx + lead_loss) / (batch_idx + 1)
            running_leads_ssim[j] = (running_leads_ssim[j] * batch_idx + lead_ssim) / (batch_idx + 1)

    for j, lead_day in enumerate(lead_days):
        ctx.metrics[f"valid/MSE@{lead_day}"] = running_leads[j].item()
        ctx.metrics[f"valid/RMSE@{lead_day}"] = torch.sqrt(running_leads[j]).item()
        ctx.metrics[f"valid/SSIM@{lead_day}"] = running_leads_ssim[j].item()

    if plotted is not None and RANK in {-1, 0}:
        LOGGER.info(f"Saved validation plot to {plotted}")
    ctx.metrics["valid/loss"] = running_loss
    ctx.tracker.log(ctx.metrics, step=ctx.curr_iter)


def main(config: IterableSimpleNamespace):
    ctx = TrainContext(config)
    schedule_setup_dataset(ctx, config)

    num_vars = len(ctx.config.variables)
    model_ovverides = {
        "num_channels": num_vars,
        "num_labels": num_vars,
        "in_channels": num_vars,
        "out_channels": num_vars,
    }
    ctx.model = load_model(ctx.config.model, config=model_ovverides).to(ctx.device)
    ctx.metrics["model/params"] = sum(p.numel() for p in ctx.model.parameters() if p.requires_grad)

    # reduction='none' so we can weight per-sample by lead time
    ctx.criterion = nn.MSELoss(reduction='none')
    ctx.optimizer = optim.Adam(ctx.model.parameters(), lr=ctx.config.lr, weight_decay=ctx.config.weight_decay)
    ctx.scheduler = optim.lr_scheduler.ReduceLROnPlateau(ctx.optimizer, mode="min", factor=0.5, patience=5)

    LOGGER.info(f"Model: {config.model}, Params: {ctx.metrics['model/params']:,}")
    LOGGER.info(f"Beta sampling: a={getattr(config, 'beta_a', 2.0)}, b={getattr(config, 'beta_b', 1.0)}")
    LOGGER.info(f"Device: {ctx.device}")
    LOGGER.info(f"Training for {ctx.config.epochs} epochs...")

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
    parser = argparse.ArgumentParser(description="Progressive training with Beta-weighted timestep sampling")
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
    parser.add_argument("--device", type=str, help="Device identifier")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_dir", type=str, help="Directory for saving checkpoints")

    # Progressive specific
    parser.add_argument("--prediction_type", type=str, choices=["sample", "epsilon"], help="Predict x_t or epsilon")

    # Beta sampling specific
    parser.add_argument("--beta_a", type=float, help="Beta distribution alpha parameter (default: 2.0)")
    parser.add_argument("--beta_b", type=float, help="Beta distribution beta parameter (default: 1.0)")

    # Plot video
    parser.add_argument("--start_date", "--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", "--end", type=str, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    configs = args.pop("config")
    base = {}
    for config in configs:
        base.update(yaml_load(config))

    args = {**base, **args}
    config = IterableSimpleNamespace(**args)
    main(config)
