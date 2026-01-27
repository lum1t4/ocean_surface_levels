"""
Next-Frame Prediction Training Script

This script trains models that predict the next frame given a single input frame.
Unlike train_ar.py which uses sequences (B, T, C, H, W), this script works with
single frames (B, C, H, W) for both input and output.

Suitable for models like SegFormer that operate on 2D images.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import tqdm
import xarray as xr

from osl.core.pytorch import RANK, device_memory_used
from osl.core.train import TrainContext
from osl.core.utils import LOGGER, IterableSimpleNamespace, yaml_load
from osl.model.registry import load_model


class NextFrameDataset(Dataset):
    """Dataset for next-frame prediction.

    Each sample consists of:
    - input: frame at time t
    - target: frame at time t+1
    """

    def __init__(self, data: xr.Dataset, variables: list[str] = ['sla', 'ugos', 'vgos']):
        self.data = data
        self.variables = variables
        self.length = len(data.time) - 1  # -1 because we need t and t+1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get frame at t and t+1
        frame_t = self.data.isel(time=idx)
        frame_t1 = self.data.isel(time=idx + 1)

        # Stack variables into channels
        input_frame = torch.stack([
            torch.from_numpy(frame_t[var].values).float()
            for var in self.variables
        ], dim=0)  # (C, H, W)

        target_frame = torch.stack([
            torch.from_numpy(frame_t1[var].values).float()
            for var in self.variables
        ], dim=0)  # (C, H, W)

        # Handle NaN values
        input_frame = torch.nan_to_num(input_frame, nan=0.0)
        target_frame = torch.nan_to_num(target_frame, nan=0.0)

        return {'inputs': input_frame, 'targets': target_frame}


def schedule_train_epoch(ctx: TrainContext):
    ctx.model.train()
    running_loss = 0.0
    running_norm = 0.0
    progress = enumerate(ctx.train_loader)
    if RANK in {-1, 0}:
        LOGGER.info(("\n" + "%11s" * 4) % ("Epoch", "GPU_mem", "Loss", "Norm"))
        progress = tqdm.tqdm(progress, total=len(ctx.train_loader))

    for batch_idx, batch in progress:
        x = batch['inputs'].to(ctx.device)   # (B, C, H, W)
        y = batch['targets'].to(ctx.device)  # (B, C, H, W)

        ctx.optimizer.zero_grad()
        preds = ctx.model(x)  # (B, C, H', W')

        # Interpolate predictions to match target size if needed
        if preds.shape[-2:] != y.shape[-2:]:
            preds = torch.nn.functional.interpolate(
                preds, size=y.shape[-2:], mode='bilinear', align_corners=False
            )

        loss = ctx.criterion(preds, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), max_norm=1.0)
        ctx.optimizer.step()
        running_norm = (running_norm * batch_idx + norm.item()) / (batch_idx + 1)
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)

        if RANK in {-1, 0}:
            progress.set_description("%11s%11.4g%11.4g%11.4g" % (
                f"{ctx.curr_iter + 1}/{ctx.config.epochs}",
                device_memory_used(ctx.device),
                running_loss,
                running_norm
            ))

    ctx.metrics["train/loss"] = running_loss
    ctx.metrics["train/norm"] = running_norm


@torch.inference_mode()
def schedule_valid_epoch(ctx: TrainContext):
    ctx.model.eval()
    running_loss = 0.0
    progress = enumerate(ctx.valid_loader)
    if RANK in {-1, 0}:
        LOGGER.info("%11s" % "Val. Loss")
        progress = tqdm.tqdm(progress, total=len(ctx.valid_loader))

    for batch_idx, batch in progress:
        x = batch['inputs'].to(ctx.device)
        y = batch['targets'].to(ctx.device)
        preds = ctx.model(x)

        # Interpolate predictions to match target size if needed
        if preds.shape[-2:] != y.shape[-2:]:
            preds = torch.nn.functional.interpolate(
                preds, size=y.shape[-2:], mode='bilinear', align_corners=False
            )

        loss = ctx.criterion(preds, y)
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
        progress.set_description("%11.4g" % running_loss)

    ctx.metrics["valid/loss"] = running_loss
    ctx.scheduler.step(ctx.metrics["valid/loss"])
    ctx.tracker.log(ctx.metrics, step=ctx.curr_iter)


def schedule_setup_dataset(ctx: TrainContext):
    print(f"Loading dataset from: {ctx.config.dataset}")
    data = xr.open_dataset(ctx.config.dataset)

    # 80 / 20 split for training and testing
    data_len = len(data.time)
    train_size = int(0.8 * data_len)
    train_data = data.isel(time=slice(0, train_size))
    valid_data = data.isel(time=slice(train_size, data_len))

    print(f"Train set size: {len(train_data.time)}, Validation set size: {len(valid_data.time)}")
    ctx.train_set = NextFrameDataset(train_data)
    ctx.valid_set = NextFrameDataset(valid_data)
    ctx.train_loader = DataLoader(
        ctx.train_set, batch_size=ctx.config.batch_size, shuffle=True,
        num_workers=ctx.config.workers, pin_memory=True
    )
    ctx.valid_loader = DataLoader(
        ctx.valid_set, batch_size=ctx.config.batch_size, shuffle=False,
        num_workers=ctx.config.workers, pin_memory=True
    )
    return ctx


def main(config: IterableSimpleNamespace):
    ctx = TrainContext(config)
    schedule_setup_dataset(ctx)

    # Load model with num_labels=3 for 3-channel output
    ctx.model = load_model(config.model, config={'num_labels': 3}).to(ctx.device)
    ctx.metrics["model/params"] = sum(p.numel() for p in ctx.model.parameters() if p.requires_grad)

    ctx.criterion = nn.MSELoss()
    ctx.optimizer = optim.Adam(ctx.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ctx.scheduler = optim.lr_scheduler.ReduceLROnPlateau(ctx.optimizer, mode='min', factor=0.5, patience=5)

    print(f"Model parameters: {ctx.metrics['model/params']:,}")
    print(f"Device: {ctx.device}")
    print(f"Training for {config.epochs} epochs...")

    for epoch in range(ctx.start_iter, config.epochs):
        ctx.curr_iter = epoch
        ctx.metrics = {}
        schedule_train_epoch(ctx)
        schedule_valid_epoch(ctx)
        ctx.checkpointing()
        ctx.iteration_end()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train next-frame prediction model")

    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--name", type=str, help="Run name")
    parser.add_argument("--tracker", type=str, help="Tracker")

    # Data configuration
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--variables', type=str, nargs='+', help='Variables to use from the dataset')

    # Training configuration
    parser.add_argument('--model', type=str, help="Model name")
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')

    # System configuration
    parser.add_argument("--workers", type=int, help="Number of dataloader worker processes")
    parser.add_argument("--device", type=str, help="Device identifier (e.g., 'cuda:0', 'cpu', 'mps:0')")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic behavior")
    parser.add_argument("--save_dir", type=str, help="Directory for saving checkpoints")

    args = parser.parse_args()

    args = {k: v for k, v in vars(args).items() if v is not None}

    config = args.pop('config', None)
    if config:
        base = yaml_load(config)
        args = {**base, **args}

    # Remove seq_length and seq_stride if present (not used in next-frame prediction)
    args.pop('seq_length', None)
    args.pop('seq_stride', None)

    config = IterableSimpleNamespace(**args)
    main(config)
