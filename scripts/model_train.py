from typing import Callable
import xarray as xr
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Any
from osl.data import SequenceDataset
from osl.dtypes import IterableSimpleNamespace
from osl.torch_utils import device_memory_clear, device_memory_used, init_seeds
from osl.model.convlstm import OceanSurfacePredictorConvLSTM

from dataclasses import dataclass, field

    
@dataclass
class TrainState:
    model: nn.Module
    optimizer: optim.Optimizer
    criterion: nn.Module | Callable
    train_set: Dataset
    valid_set: Dataset
    train_loader: DataLoader
    valid_loader: DataLoader
    device: torch.device
    config: IterableSimpleNamespace
    best: Path
    last: Path
    metrics: dict[str, Any] = field(default_factory=dict)
    epoch: int = 0


def model_train_epoch(state: TrainState):
    state.model.train()
    running_loss = 0.0

    for batch_idx, (x, y) in enumerate(state.train_loader):
       
        x, y = x.to(state.device), y.to(state.device)
        
        # Reshape based on data structure
        # or (batch, seq, h, w) if single var
        if x.dim() == 4:  # Single channel case
            x = x.unsqueeze(2)  # Add channel dimension -> (batch, seq, 1, h, w)
        
        if y.dim() == 4:  # y shape: (batch, seq_target, h, w)
            y = y.unsqueeze(2)  # Add channel dimension -> (batch, seq_target, 1, h, w)
        
        state.optimizer.zero_grad()
        pred = state.model(x)  # pred shape: (B, S, C, H, W)
        
        assert pred.shape == y.shape, f"Shape mismatch - Pred: {pred.shape}, Target: {y.shape}"
        
        loss = state.criterion(pred, y)
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=1.0)
        state.optimizer.step()
        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"  Batch [{batch_idx}/{len(state.train_loader)}] Loss: {loss.item():.6f}")

    state.metrics["train/loss"] = running_loss / len(state.train_loader)


@torch.inference_mode()
def model_valid_epoch(state: TrainState):
    state.model.eval()
    running_loss = 0.0

    for x, y in state.valid_loader:
        x, y = x.to(state.device), y.to(state.device)
        
        # Reshape based on data structure
        if x.dim() == 4:  # Single channel case
            x = x.unsqueeze(2)  # Add channel dimension
        
        if y.dim() == 4:
            y = y.unsqueeze(2)

        pred = state.model(x)
        loss = state.criterion(pred, y)
        running_loss += loss.item()

    state.metrics["valid/loss"] = running_loss / len(state.valid_loader)


def model_checkpoint(state: TrainState):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': state.epoch,
        'model_state_dict': state.model.state_dict(),
        'optimizer_state_dict': state.optimizer.state_dict(),
        'train_loss': state.metrics.get("train/loss", 0),
        'valid_loss': state.metrics.get("valid/loss", 0),
    }
    
    # Save last checkpoint
    torch.save(checkpoint, state.last)
    
    # Save best checkpoint if validation loss improved
    if "best_valid_loss" not in state.metrics or state.metrics["valid/loss"] < state.metrics["best_valid_loss"]:
        state.metrics["best_valid_loss"] = state.metrics["valid/loss"]
        torch.save(checkpoint, state.best)
        print(f"  New best model saved with validation loss: {state.metrics['valid/loss']:.6f}")


def train(hyp: IterableSimpleNamespace):
    init_seeds(hyp.seed, deterministic=hyp.deterministic)
    device = torch.device(hyp.device)
    workers = 0 if device.type in {"cpu", "mps"} else hyp.workers
    
    # Create save directory
    save_dir = Path(hyp.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Data, dataset and dataloader setup
    print(f"Loading dataset from: {hyp.dataset}")
    data = xr.open_dataset(hyp.dataset)
    
    # 80 / 20 split for training and testing
    data_len = len(data.time)
    train_size = int(0.8 * data_len)
    train_data = data.isel(time=slice(0, train_size))
    valid_data = data.isel(time=slice(train_size, data_len))

    print(f"Train set size: {len(train_data.time)}, Validation set size: {len(valid_data.time)}")
    
    # Determine number of channels based on variables
    variables = ["sla", "ugos", "vgos"]
    num_channels = len(variables) if hyp.stack_vars else 1
    
    train_set = SequenceDataset(
        train_data, 
        variables, 
        hyp.seq_length, 
        hyp.seq_target, 
        hyp.seq_stride, 
        hyp.stack_vars
    )
    valid_set = SequenceDataset(
        valid_data, 
        variables, 
        hyp.seq_length, 
        hyp.seq_target, 
        hyp.seq_stride, 
        hyp.stack_vars
    )
    
    train_loader = DataLoader(train_set, batch_size=hyp.batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=hyp.batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Get a sample to check dimensions
    sample_x, sample_y = next(iter(train_loader))
    print(f"Sample input shape: {sample_x.shape}, Sample target shape: {sample_y.shape}")
    
    # Initialize model
    model = OceanSurfacePredictorConvLSTM(
        input_channels=num_channels,
        hidden_dims=hyp.hidden_dims,
        kernel_sizes=hyp.kernel_sizes,
        num_layers=hyp.num_layers,
        seq_length=hyp.seq_length,
        seq_target=hyp.seq_target
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyp.lr, weight_decay=hyp.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    model.to(device)

    state = TrainState(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_set=train_set,
        valid_set=valid_set,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        config=hyp,
        best=save_dir / "best.pth",
        last=save_dir / "last.pth"
    )

    state.metrics["model/params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {state.metrics['model/params']:,}")
    print(f"Device: {device}")
    print(f"Training for {hyp.epochs} epochs...")
    
    for epoch in range(hyp.epochs):
        print(f"\nEpoch {epoch + 1}/{hyp.epochs}")
        print("-" * 50)
        
        model_train_epoch(state)
        model_valid_epoch(state)
        
        # Update learning rate
        scheduler.step(state.metrics["valid/loss"])
        
        print(f"  Train Loss: {state.metrics['train/loss']:.6f}")
        print(f"  Valid Loss: {state.metrics['valid/loss']:.6f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        model_checkpoint(state)
        state.epoch += 1
        
        # Clear GPU memory
        if device.type == "cuda":
            device_memory_clear()
            print(f"  GPU Memory: {device_memory_used():.2f} MB")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {state.metrics['best_valid_loss']:.6f}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train ocean surface predictor with ConvLSTM")

    # Model architecture
    parser.add_argument("--hidden_dims", type=int, nargs='+', default=[64, 128, 128, 64], help="Hidden dimensions for ConvLSTM layers")
    parser.add_argument("--kernel_sizes", type=int, nargs='+', default=[3, 3, 3, 3], help="Kernel sizes for ConvLSTM layers")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of ConvLSTM layers")
    
    # Data configuration
    parser.add_argument('--dataset', type=str, default='data/raw/cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.0625deg_P1D-224x224.nc', help='Path to dataset directory')
    parser.add_argument('--seq_length', type=int, default=3, help='Number of past steps')
    parser.add_argument('--seq_target', type=int, default=1, help='Steps ahead to predict')
    parser.add_argument('--seq_stride', type=int, default=4, help='Stride for sequence sampling')
    parser.add_argument('--stack_vars', action='store_true', default=True, help='Stack variables into channels')
    parser.add_argument('--variables', type=str, nargs='+', default=['sla', 'ugos', 'vgos'], help='Variables to use from the dataset')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    
    # System configuration
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader worker processes")
    parser.add_argument("--device", type=str, default="cpu", help="Device identifier (e.g., 'cuda:0', 'cpu', 'mps:0')")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic behavior")
    parser.add_argument("--save_dir", type=str, default="runs/convlstm", help="Directory for saving checkpoints")
    
    args = parser.parse_args()
    
    # Override with test configuration for 1 epoch test
    hyp = IterableSimpleNamespace(
        # Model architecture - smaller for testing
        hidden_dims = [32, 64, 32],
        kernel_sizes = [3, 3, 3],
        num_layers = 3,
        
        # Data
        dataset = "data/raw/cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.0625deg_P1D-224x224.nc",
        seq_length = 3,
        seq_target = 1,
        seq_stride = 4,
        stack_vars = True,
        
        # Training
        batch_size = 4,  # Reduced for testing
        lr = 1e-3,
        weight_decay = 1e-5,
        epochs = 1,  # Just 1 epoch for testing
        
        # System
        workers = 0,  # Use 0 for debugging
        device = "mps:0" if torch.backends.mps.is_available() else "cpu",
        seed = 1337,
        deterministic = False,
        save_dir = "runs/convlstm_test",
    )

    train(hyp)