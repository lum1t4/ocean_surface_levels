
"""
1. The first phase is to train an interpolator which is a neural network that models
a function I(x_{t}, x_{t+i}, i) that should approximate x_{t+i} where:
    - h is horizon or seq_length.
    - i is time step an integer ]0, h] used to pick one of the element of
    the sequence between the start and the end.
The interpolator can use, in inference, a real valued time step i to estimate an
intermidiate representation which might not be present in the original discrete
sequence emulating a continuos sampling from the sequence.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import tqdm
import xarray as xr

from osl.core.pytorch import RANK, device_memory_used
from osl.core.train import TrainContext
from osl.core.utils import LOGGER, IterableSimpleNamespace, yaml_load
from osl.data import SequenceDataset
from osl.model.registry import load_model

"""

TODO:
- [ ] Diffusion and Interpolation steps
- [ ] Forecast train and sampling

"""

# ------------------------------
# Intepolator
# ------------------------------

def interpolate(interpolator: nn.Module, x0: torch.Tensor, xT: torch.Tensor, time: torch.Tensor):
    """Interpolator forward pass"""
    x0T = torch.cat([x0, xT], dim=1)
    return interpolator(x0T, time)


def interpolator_step(ctx: TrainContext, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    sequence: torch.Tensor = batch['inputs']
    B, T, _, _, _ = sequence.shape
    # Take a random time step for each sequence
    ts = torch.randint(1, T, (B,))
    # First and last item for each sequence
    x0, xT = sequence[:,  0], sequence[:, -1]
    # Sample indices [0, 1, ..., B] are need to extract the sample at time ts in the batch
    xi = sequence[torch.arange(B), ts]
    
    x0 = x0.to(ctx.device, non_blocking=True, dtype=torch.float32)
    xT = xT.to(ctx.device, non_blocking=True, dtype=torch.float32)
    xi = xi.to(ctx.device, non_blocking=True, dtype=torch.float32)
    ts = ts.to(ctx.device, non_blocking=True, dtype=torch.float32)

    preds = interpolate(ctx.interpolator, x0, xT, ts)
    loss = ctx.criterion(preds, xi)
    return preds, loss


def interpolator_train_epoch(ctx: TrainContext):
    ctx.interpolator.train()
    running_loss = 0.0
    running_norm = 0.0
    progress = enumerate(ctx.train_loader)
    if RANK in {-1, 0}:
        LOGGER.info(("\n" + "%11s" * 4) % ("Epoch", "GPU_mem", "Loss", "Norm"))
        progress = tqdm.tqdm(progress, total=len(ctx.train_loader))

    for batch_idx, batch in progress:
        ctx.optimizer.zero_grad()
        _, loss = interpolator_step(ctx, batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ctx.interpolator.parameters(), max_norm=1.0)
        ctx.optimizer.step()

        running_norm = (running_norm * batch_idx + norm.item()) / (batch_idx + 1)
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
        
        if RANK in {-1, 0}:
            epoch = f"{ctx.curr_iter + 1}/{ctx.config.epochs}"
            mem = device_memory_used(ctx.device)
            progress.set_description("%11s%11.4g%11.4g%11.4g" % (epoch, mem, running_loss, running_norm))

    ctx.metrics["interpolator/train/loss"] = running_loss
    ctx.metrics["interpolator/train/norm"] = running_norm


@torch.inference_mode()
def interpolator_valid_epoch(ctx: TrainContext):
    ctx.interpolator.eval()
    running_loss = 0.0
    progress = enumerate(ctx.valid_loader)
    if RANK in {-1, 0}:
        LOGGER.info("%11s" % "Val. Loss")
        progress = tqdm.tqdm(progress, total=len(ctx.valid_loader))

    for batch_idx, batch in progress:
        _, loss = interpolator_step(ctx, batch)
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
        progress.set_description("%11.4g" % running_loss)

    ctx.metrics["interpolator/valid/loss"] = running_loss
    ctx.scheduler.step(running_loss) # update learning rate
    ctx.tracker.log(ctx.metrics, step=ctx.curr_iter) # log metrics


def schedule_interpolator(ctx: TrainContext):
    ctx.metrics = {}
    # avoids conflicts when saving a model checkpoint
    ctx.save_dir = ctx.save_dir / 'interpolator'
    ctx.interpolator = load_model(config.interpolator).to(ctx.device)
    ctx.metrics["interpolator/params"] = sum(p.numel() for p in ctx.interpolator.parameters() if p.requires_grad)
    ctx.criterion = nn.MSELoss()
    ctx.optimizer = optim.Adam(ctx.interpolator.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ctx.scheduler = optim.lr_scheduler.ReduceLROnPlateau(ctx.optimizer, mode='min', factor=0.5, patience=5)
    print(f"Model parameters: {ctx.metrics['model/params']:,}")
    print(f"Device: {ctx.device}")
    print(f"Training for {config.epochs} epochs...")
    
    for epoch in range(ctx.start_iter, config.epochs):
        ctx.curr_iter = epoch
        ctx.metrics = {}
        interpolator_train_epoch(ctx)
        interpolator_valid_epoch(ctx)
        ctx.checkpointing()
        ctx.iteration_end()

    ctx.save_dir = ctx.save_dir.parent
    return ctx


# ------------------------------
# Forecaster
# ------------------------------


def forecast(forecaster: nn.Module, x0: torch.Tensor, xi: torch.Tensor, time: torch.Tensor):
    """Interpolator forward pass"""
    x0i = torch.cat([x0, xi], dim=1)
    return forecaster(x0i, time)

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



def forecaster_step(ctx: TrainContext, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    sequence: torch.Tensor = batch['inputs']
    B, T, _, _, _ = sequence.shape
    # Take a random time step for each sequence
    ts = torch.randint(1, T, (B,))
    # First and last item for each sequence
    x0, xT = sequence[:,  0], sequence[:, -1]
    

    x0 = x0.to(ctx.device, non_blocking=True, dtype=torch.float32)
    ts = ts.to(ctx.device, non_blocking=True, dtype=torch.float32)
    xT = xT.to(ctx.device, non_blocking=True, dtype=torch.float32)

    # xi is the interpolated item at a specific timestep
    # xi is initialized to x0 (like assuming steps = 0)
    xi = x0.clone()
    mask = ts > 0

    with torch.inference_mode():
        inference_dropout_enable(ctx.interpolator)
        xi[mask] = interpolate(ctx.interpolator, x0[mask], xT[mask], ts[mask])
        inference_dropout_disable(ctx.intepolator)
    
    
    xT_pred = forecast(x0, xi, ts)
    loss = torch.nn.functional.mse_loss(xT_pred, xT)
    return xT_pred, loss


def schedule_setup_dataset(ctx: TrainContext):
    # Data, dataset and dataloader setup
    print(f"Loading dataset from: {ctx.config.dataset}")
    data = xr.open_dataset(ctx.config.dataset)
    
    # 80 / 20 split for training and testing
    # NOTE: the split is not random, the samples are ordered by time
    total = len(data.time)
    train_size = int(0.8 * total)
    train_data = data.isel(time=slice(0, train_size))
    valid_data = data.isel(time=slice(train_size, total))

    print(f"Train set size: {len(train_data.time)}, Validation set size: {len(valid_data.time)}")
    ctx.train_set = SequenceDataset(train_data, seq_length=ctx.config.seq_length, seq_stride=ctx.config.seq_stride)
    ctx.valid_set = SequenceDataset(valid_data, seq_length=ctx.config.seq_length, seq_stride=ctx.config.seq_stride)
    ctx.train_loader = DataLoader(ctx.train_set, batch_size=ctx.config.batch_size, shuffle=True,  num_workers=ctx.config.workers, pin_memory=True)
    ctx.valid_loader = DataLoader(ctx.valid_set, batch_size=ctx.config.batch_size, shuffle=False, num_workers=ctx.config.workers, pin_memory=True)
    return ctx



def forecaster_train_epoch(ctx: TrainContext):
    ctx.forecaster.train()
    running_loss = 0.0
    running_norm = 0.0
    progress = enumerate(ctx.train_loader)
    if RANK in {-1, 0}:
        LOGGER.info(("\n" + "%11s" * 4) % ("Epoch", "GPU_mem", "Loss", "Norm"))
        progress = tqdm.tqdm(progress, total=len(ctx.train_loader))

    for batch_idx, batch in progress:
        ctx.optimizer.zero_grad()
        _, loss = forecaster_step(ctx, batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ctx.forecaster.parameters(), max_norm=1.0)
        ctx.optimizer.step()

        running_norm = (running_norm * batch_idx + norm.item()) / (batch_idx + 1)
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)
        
        if RANK in {-1, 0}:
            epoch = f"{ctx.curr_iter + 1}/{ctx.config.epochs}"
            mem = device_memory_used(ctx.device)
            progress.set_description("%11s%11.4g%11.4g%11.4g" % (epoch, mem, running_loss, running_norm))

    ctx.metrics["forecaster/train/loss"] = running_loss
    ctx.metrics["forecaster/train/norm"] = running_norm


def forecaster_valid_epoch(ctx):
    pass


def schedule_forecaster(ctx: TrainContext):
    assert ctx.interpolator is not None, "Interpolator must exist to train the forecaster"
    
    # Freeze interpolator (it must be pretrained)
    ctx.interpolator.eval()
    for p in ctx.interpolator.parameters():
        p.requires_grad = False


    forward_conditon: str = "data"
    schedule: str = "before_t1_only",
    lambda_0: float = 0.5
    lambda_1: float = 0.5

    additional_interpolation_steps: int = 0   # k, how many additional diffusion steps to add. Only used if schedule='before_t1_only'
    additional_interpolation_steps_factor: int = 0  # only use if schedule='linear'
    interpolate_before_t1: bool = True   # Whether to interpolate before t1 too. Must be true if schedule='before_t1_only'
    time_encoding: str = "dynamics"
    assert forward_conditon in {"data", "none", "data+noise"}, "forward_conditon has not a valid value"
    assert time_encoding in {"dynamics", "discrete"}


    # interpolator steps i in [1, h - 1]
    # diffusion steps
    T = ctx.config.seq_length + additional_interpolation_steps

    # h = 5
    # add = 2
    # T = [0, 1, 2, 3, 4, 5, 6]

    """
    d_to_i_step = 
    1 -> 1 / 3
    2 -> 2 / 3
    3 -> 1
    4 -> 2
    5 -> 3
    """


    ctx.metrics = {}
    # avoids conflicts when saving a model checkpoint
    ctx.save_dir = ctx.save_dir / 'forecaster'
    ctx.forecaster = load_model(config.forecaster).to(ctx.device)
    ctx.metrics["forecaster/params"] = sum(p.numel() for p in ctx.forecaster.parameters() if p.requires_grad)
    ctx.criterion = nn.MSELoss()
    ctx.optimizer = optim.Adam(ctx.forecaster.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ctx.scheduler = optim.lr_scheduler.ReduceLROnPlateau(ctx.optimizer, mode='min', factor=0.5, patience=5)
    print(f"Model parameters: {ctx.metrics['model/params']:,}")
    print(f"Device: {ctx.device}")
    print(f"Training for {config.epochs} epochs...")
    
    for epoch in range(ctx.start_iter, config.epochs):
        ctx.curr_iter = epoch
        ctx.metrics = {}
        forecaster_train_epoch(ctx)
        forecaster_valid_epoch(ctx)
        ctx.checkpointing()
        ctx.iteration_end()
    
    ctx.save_dir = ctx.save_dir.parent
    return ctx



def main(config: IterableSimpleNamespace):
    ctx = TrainContext(config)
    schedule_setup_dataset(ctx)
    schedule_interpolator(ctx)
    schedule_forecaster(ctx)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train ocean surface predictor with ConvLSTM")

    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--name", type=str, help="Run name")
    parser.add_argument("--tracker", type=str, help="Tracker")

    # 1. Data configuration
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--seq_length', type=int, help='Number of past steps')
    parser.add_argument('--seq_stride', type=int, help='Stride for sequence sampling')
    parser.add_argument('--variables', type=str, nargs='+', help='Variables to use from the dataset')

    # 2. Training configuration
    parser.add_argument('--interpolator', type=str, help="Interpolator model name")
    parser.add_argument('--forcaster', type=str, help="Forcaster model name")
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')

    # 3. System configuration
    parser.add_argument("--workers", type=int, help="Number of dataloader worker processes")
    parser.add_argument("--device", type=str, help="Device identifier (e.g., 'cuda:0', 'cpu', 'mps:0')")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic behavior")
    parser.add_argument("--save_dir", type=str, help="Directory for saving checkpoints")

    args = parser.parse_args()

    args = {k: v for k, v in vars(args).items() if v is not None}
    
    config = args.pop('config')
    if config:
        base = yaml_load(config)
        args = {**base, **args}  # Command-line args override config file
    
    config = IterableSimpleNamespace(**args)
    main(config)
