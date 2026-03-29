from torch.utils.data import DataLoader
import xarray as xr
from osl.core.pytorch import RANK, device_memory_used
from osl.core.train import TrainContext
from osl.core.utils import LOGGER, IterableSimpleNamespace, yaml_load
from osl.data import SequenceDataset, dataset_stats_load
from osl.core.metrics import ssim_metric

import tqdm
from torch import nn, optim
import torch
from osl.model import load_model
import matplotlib


matplotlib.use("Agg")


def interpolate(model, x0: torch.Tensor, xT: torch.Tensor, step: torch.Tensor):
    return model(torch.cat([x0, xT], dim=1), step)


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
    sequence: torch.Tensor = batch["inputs"]
    B, T, _, _, _ = sequence.shape
    # Take a random time step for each sequence
    timesteps = torch.randint(1, T - 1, (B,))
    # First and last item for each sequence
    x0, xT = sequence[:, 0], sequence[:, -1]
    # Sample indices [0, 1, ..., B] are need to extract the sample at time ts in the batch
    xI = sequence[torch.arange(B), timesteps]

    x0 = x0.to(ctx.device, non_blocking=True, dtype=torch.float32)
    xT = xT.to(ctx.device, non_blocking=True, dtype=torch.float32)
    xI = xI.to(ctx.device, non_blocking=True, dtype=torch.float32)

    timesteps = timesteps.to(ctx.device, non_blocking=True, dtype=torch.long)
    preds = interpolate(ctx.model, x0, xT, timesteps)
    loss = ctx.criterion(preds, xI)
    return preds, loss


def model_valid_step(ctx: TrainContext, batch_idx: int, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence: torch.Tensor = batch["inputs"]
    B, T, _, _, _ = sequence.shape
    # Take a random time step for each sequence
    timesteps = torch.randint(1, T - 1, (B,))
    # First and last item for each sequence
    x0, xT = sequence[:, 0], sequence[:, -1]
    # Sample indices [0, 1, ..., B] are need to extract the sample at time ts in the batch
    xI = sequence[torch.arange(B), timesteps]

    x0 = x0.to(ctx.device, non_blocking=True, dtype=torch.float32)
    xT = xT.to(ctx.device, non_blocking=True, dtype=torch.float32)
    xI = xI.to(ctx.device, non_blocking=True, dtype=torch.float32)

    timesteps = timesteps.to(ctx.device, non_blocking=True, dtype=torch.long)
    preds = interpolate(ctx.model, x0, xT, timesteps)
    loss = ctx.criterion(preds, xI)
    return preds, xI, loss


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
    loader_args = dict(batch_size=config.batch_size, num_workers=config.workers, pin_memory=False)

    if ctx.device.type not in {'cpu', 'mps'}:
        loader_args["pin_memory"] = True

    ctx.train_set = SequenceDataset(data, bound=train_bound, **reader_args)
    ctx.valid_set = SequenceDataset(data, bound=valid_bound, **reader_args)

    ctx.train_loader = DataLoader(ctx.train_set, shuffle=True, **loader_args)
    ctx.valid_loader = DataLoader(ctx.valid_set, shuffle=False, **loader_args)

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
    progress = enumerate(ctx.valid_loader)
    
    if RANK in {-1, 0}:
        LOGGER.info("%11s" % "Val. Loss")
        progress = tqdm.tqdm(progress, total=len(ctx.valid_loader))

    running_loss = 0.0
    running_ssim = 0.0
    for batch_idx, batch in progress:
        preds, targets, loss = model_valid_step(ctx, batch_idx, batch)# B, T, C, H, W
        batch_ssim = ssim_metric(preds, targets)
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
        running_ssim = (running_ssim * batch_idx + batch_ssim.item()) / (batch_idx + 1)
        if RANK in {-1, 0}:
            progress.set_description("%11.4g" % running_loss)

    ctx.metrics["valid/loss"] = running_loss
    ctx.metrics["valid/ssim"] = running_ssim
    ctx.tracker.log(ctx.metrics, step=ctx.curr_iter)


def main(config: IterableSimpleNamespace):
    ctx = TrainContext(config)
    # Load dataset
    schedule_setup_dataset(ctx, config)

    # Load model
    model_ovverides = {"horizon": ctx.config.seq_length - 1}
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
