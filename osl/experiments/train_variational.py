import torch
import torch.nn as nn
from collections import defaultdict
from torch import optim
from torch.utils.data import DataLoader
import tqdm
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

from osl.core.pytorch import RANK, device_memory_used, model_get_num_params
from osl.core.train import TrainContext
from osl.core.utils import LOGGER, IterableSimpleNamespace, yaml_load
from osl.model import load_model
from osl.data import FrameDataset, dataset_stats_load
from osl.core.metrics import ssim_metric

matplotlib.use("Agg")


# ----------------------------------------------------------------
# Discriminator
# ----------------------------------------------------------------
class Discriminator(nn.Module):
    """PatchGAN discriminator. Outputs a spatial grid of real/fake logits."""

    def __init__(
        self,
        in_channels: int = 3,
        conv_channels: list = [64, 128, 256],
        kernels: list = [4, 4, 4, 4],
        strides: list = [2, 2, 2, 1],
        paddings: list = [1, 1, 1, 1],
    ):
        super().__init__()
        self.im_channels = in_channels
        activation = nn.LeakyReLU(0.2)
        layers_dim = [self.im_channels] + conv_channels + [1]
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        layers_dim[i],
                        layers_dim[i + 1],
                        kernel_size=kernels[i],
                        stride=strides[i],
                        padding=paddings[i],
                        bias=False if i != 0 else True,
                    ),
                    nn.BatchNorm2d(layers_dim[i + 1])
                    if i != len(layers_dim) - 2 and i != 0
                    else nn.Identity(),
                    activation if i != len(layers_dim) - 2 else nn.Identity(),
                )
                for i in range(len(layers_dim) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ----------------------------------------------------------------
# Losses
# ----------------------------------------------------------------
def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())


def _broadcast_mask(mask: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    while mask.ndim < ref_tensor.ndim:
        mask = mask.unsqueeze(0)
    return mask


def gradient_loss(
    preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    pred_dx = preds[..., :, 1:] - preds[..., :, :-1]
    target_dx = targets[..., :, 1:] - targets[..., :, :-1]
    pred_dy = preds[..., 1:, :] - preds[..., :-1, :]
    target_dy = targets[..., 1:, :] - targets[..., :-1, :]

    diff_dx = (pred_dx - target_dx).abs()
    diff_dy = (pred_dy - target_dy).abs()

    if mask is None:
        return 0.5 * (diff_dx.mean() + diff_dy.mean())

    mask = mask.to(device=preds.device, dtype=preds.dtype)
    mask_dx_raw = mask[..., :, 1:] * mask[..., :, :-1]
    mask_dy_raw = mask[..., 1:, :] * mask[..., :-1, :]

    mask_dx = _broadcast_mask(mask_dx_raw, diff_dx).expand_as(diff_dx)
    mask_dy = _broadcast_mask(mask_dy_raw, diff_dy).expand_as(diff_dy)

    loss_dx = (diff_dx * mask_dx).sum() / mask_dx.sum().clamp_min(1.0)
    loss_dy = (diff_dy * mask_dy).sum() / mask_dy.sum().clamp_min(1.0)
    return 0.5 * (loss_dx + loss_dy)


def mse_loss(
    inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None
):
    if mask is None:
        return nn.functional.mse_loss(inputs, targets)
    diff = (inputs - targets) * mask
    denom = (mask.sum() * inputs.shape[0] * inputs.shape[1] * inputs.shape[2]).clamp_min(1.0)
    return (diff**2).sum() / denom



# ----------------------------------------------------------------
# Train step
# ----------------------------------------------------------------
def model_train_step(ctx: TrainContext, batch_idx: int, batch: dict) -> tuple:
    x = batch["inputs"].to(ctx.device, non_blocking=True, dtype=torch.float32)

    w_grad = ctx.config.w_grad
    w_recon = ctx.config.w_recon
    w_adv = ctx.config.w_adv
    w_kl = ctx.config.w_kl

    # Encode source frame
    mu, sigma = ctx.model.encode(x)
    z = ctx.model.sample(mu, sigma)
    y = ctx.model.decode(z)

    losses = {}

    recon = mse_loss(y, x)
    grad = gradient_loss(y, x)
    kl = kl_divergence(mu, sigma)

    losses["train/recon"] = recon.item()
    losses["train/grad"] = grad.item()
    losses["train/kl"] = kl.item()

    loss = w_recon * recon + w_grad * grad + w_kl * kl


    if ctx.enable_discriminator:
        if ctx.curr_iter >= ctx.config.disc_start:
            ctx.discriminator.eval()
            logits_fake = ctx.discriminator(y)
            adversarial = nn.functional.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
            losses["train/adv"] = adversarial.item()
            loss += w_adv * adversarial

        ctx.discriminator.train()
        # Discriminator loss
        logit_fake = ctx.discriminator(y.detach())
        logit_true = ctx.discriminator(x)
        d_loss = 0.5 * (
            nn.functional.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))+ 
            nn.functional.binary_cross_entropy_with_logits(logit_true, torch.ones_like(logit_true))
        )
        losses["train/disc"] = d_loss.item()
        (d_loss / ctx.config.grad_acc).backward()
        

    (loss / ctx.config.grad_acc).backward()
    return x, y, loss, losses


def model_valid_step(ctx: TrainContext, batch_idx: int, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["inputs"].to(ctx.device, non_blocking=True, dtype=torch.float32)
    mu, sigma = ctx.model.encode(x)
    z = ctx.model.sample(mu, sigma)
    y = ctx.model.decode(z)
    loss = mse_loss(x, y)
    return x, y, loss


# ----------------------------------------------------------------
# Plot
# ----------------------------------------------------------------
def plot_valid_batch(ctx: TrainContext, x: torch.Tensor, x_hat: torch.Tensor):
    """Grid: top row = input frames, bottom row = reconstructed. First 4 samples, first variable."""
    import matplotlib

    matplotlib.use("Agg")

    n = min(8, x.shape[0])
    # first variable (channel 0)
    inp = x[:n, 0].cpu().numpy()
    rec = x_hat[:n, 0].cpu().numpy()

    vmin = min(inp.min(), rec.min())
    vmax = max(inp.max(), rec.max())

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        axes[0, i].imshow(inp[i], vmin=vmin, vmax=vmax, cmap="RdBu_r")
        axes[1, i].imshow(rec[i], vmin=vmin, vmax=vmax, cmap="RdBu_r")
        axes[0, i].set_title(f"Sample {i}")
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Input", fontsize=12)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=12)
    fig.suptitle(f"Epoch {ctx.curr_iter + 1}", fontsize=14)
    fig.tight_layout()

    dst = ctx.plot_dir / f"epoch_{ctx.curr_iter:03d}.png"
    fig.savefig(dst)
    plt.close(fig)

    return dst


# ----------------------------------------------------------------
# Dataset setup
# ----------------------------------------------------------------
def schedule_setup_dataset(ctx: TrainContext, config: IterableSimpleNamespace):
    LOGGER.info(f"Loading dataset from: {config.dataset}")
    data = xr.open_dataset(config.dataset)
    stats = dataset_stats_load(config.stats_path, data, config.variables, overwrite=True)

    train_bound = [int(config.train[0]), int(config.train[1])]
    valid_bound = [int(config.valid[0]), int(config.valid[1])]

    LOGGER.info(
        f"Train set size: {train_bound[1] - train_bound[0]}  "
        f"Valid set size: {valid_bound[1] - valid_bound[0]}"
    )

    reader_args = dict(vars=config.variables, normalize=config.normalize, stats=stats)
    loader_args = dict(batch_size=config.batch_size, num_workers=config.workers, pin_memory=False)

    if ctx.device.type not in {"cpu", "mps"}:
        loader_args["pin_memory"] = True

    ctx.train_set = FrameDataset(data, bound=train_bound, **reader_args)
    ctx.valid_set = FrameDataset(data, bound=valid_bound, **reader_args)

    ctx.train_loader = DataLoader(ctx.train_set, shuffle=True, **loader_args)
    ctx.valid_loader = DataLoader(
        ctx.valid_set, shuffle=False, drop_last=False, **loader_args
    )

    return ctx


# ----------------------------------------------------------------
# Train epoch
# ----------------------------------------------------------------
def schedule_train_epoch(ctx: TrainContext):
    ctx.model.train()
    running_loss = 0.0
    running_norm = 0.0
    running_comp = defaultdict(float)

    progress = enumerate(ctx.train_loader)
    if RANK in {-1, 0}:
        LOGGER.info(("\n" + "%11s" * 4) % ("Epoch", "GPU_mem", "Loss", "Norm"))
        progress = tqdm.tqdm(progress, total=len(ctx.train_loader))

    ctx.optimizer.zero_grad()

    if ctx.enable_discriminator:
        ctx.optimizer_d.zero_grad()

    for batch_idx, batch in progress:
        step_idx = (batch_idx + 1) // ctx.config.grad_acc
        x, y, loss, losses = model_train_step(ctx, batch_idx, batch)

        if (batch_idx + 1) % ctx.config.grad_acc == 0:
            norm = torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), max_norm=10.0)
            ctx.optimizer.step()
            ctx.optimizer.zero_grad()
            running_norm = (running_norm * step_idx + norm.item()) / (step_idx + 1)
            running_loss = (running_loss * step_idx + loss.item()) / (step_idx + 1)

            if ctx.enable_discriminator:
                torch.nn.utils.clip_grad_norm_(ctx.discriminator.parameters(), max_norm=10.0)
                ctx.optimizer_d.step()
                ctx.optimizer_d.zero_grad()

            for k, v in losses.items():
                running_comp[k] = (running_comp[k] * step_idx + v) / (step_idx + 1)

            if RANK in {-1, 0}:
                epoch_desc = f"{ctx.curr_iter + 1}/{ctx.config.epochs}"
                memory_used = device_memory_used(ctx.device)
                progress.set_description("%11s%11.4g%11.4g%11.4g" % (epoch_desc, memory_used, running_loss, running_norm))

    ctx.metrics["train/loss"] = running_loss
    ctx.metrics["train/norm"] = running_norm

    for k, v in running_comp.items():
        ctx.metrics[k] = v


# ----------------------------------------------------------------
# Valid epoch
# ----------------------------------------------------------------
@torch.inference_mode()
def schedule_valid_epoch(ctx: TrainContext):
    ctx.model.eval()

    progress = enumerate(ctx.valid_loader)
    if RANK in {-1, 0}:
        LOGGER.info(("%11s" * 2) % ("Val.MSE", "Val.SSIM"))
        progress = tqdm.tqdm(progress, total=len(ctx.valid_loader))

    running_mse = 0.0
    running_ssim = 0.0

    for batch_idx, batch in progress:
        x, y, loss = model_valid_step(ctx, batch_idx, batch)
        ssim_val = ssim_metric(y, x)

        running_mse = (running_mse * batch_idx + loss.item()) / (batch_idx + 1)
        running_ssim = (running_ssim * batch_idx + ssim_val.item()) / (batch_idx + 1)

        if RANK in {-1, 0}:
            progress.set_description("%11.4g%11.4f" % (running_mse, running_ssim))
            if batch_idx == 0:
                plot_valid_batch(ctx, x, y)

    ctx.metrics["valid/mse"] = running_mse
    ctx.metrics["valid/ssim"] = running_ssim
    ctx.metrics["valid/loss"] = running_mse
    ctx.tracker.log(ctx.metrics, step=ctx.curr_iter)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main(config: IterableSimpleNamespace):
    ctx = TrainContext(config)
    schedule_setup_dataset(ctx, config)
    num_vars = len(ctx.config.variables)
    model_overrides = {"in_channels": num_vars, "out_channels": num_vars}
    ctx.model = load_model(ctx.config.model, config=model_overrides).to(ctx.device)
    ctx.optimizer = optim.Adam(ctx.model.parameters(), lr=ctx.config.lr, weight_decay=ctx.config.weight_decay)
    ctx.scheduler = optim.lr_scheduler.ReduceLROnPlateau(ctx.optimizer, mode="min", factor=0.5, patience=5)
    g_num_params = model_get_num_params(ctx.model)
    ctx.metrics["model/params"] = g_num_params


    ctx.enable_discriminator = ctx.config.w_adv > 0
    if ctx.enable_discriminator:
        ctx.discriminator = Discriminator(in_channels=num_vars).to(ctx.device)
        disc_lr = config.get("disc_lr", ctx.config.lr)
        ctx.optimizer_d = optim.Adam(ctx.discriminator.parameters(), lr=disc_lr, weight_decay=ctx.config.weight_decay)
        d_num_params = model_get_num_params(ctx.discriminator)
        LOGGER.info(f"Discriminator: {d_num_params:,} params, lr={disc_lr}, start_epoch={config.disc_start}")

    LOGGER.info(f"Model: {config.model}, Params: {g_num_params:,}")
    LOGGER.info(f"Device: {ctx.device}")
    LOGGER.info(f"Training for {ctx.config.epochs} epochs...")

    for epoch in range(ctx.start_iter, ctx.config.epochs):
        ctx.curr_iter = epoch
        if epoch > ctx.start_iter:
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

    parser = argparse.ArgumentParser(
        description="Variational Autoencoder: reconstruction training with SSIM evaluation"
    )

    parser.add_argument("--config", action="append", default=[], help="Configuration file path")
    parser.add_argument("--name", type=str, help="Run name")
    parser.add_argument("--tracker", type=str, help="Tracker")

    # Data configuration
    parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    parser.add_argument("--variables", type=str, nargs="+", help="Variables to use")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="Enable normalization")

    # Model configuration
    parser.add_argument("--model", type=str, help="Model name in registry")

    # Loss weights
    parser.add_argument("--w_recon", type=float, help="Weight: reconstruction of input frame")
    parser.add_argument("--w_kl", type=float, help="Weight: KL divergence regularization")
    parser.add_argument("--w_grad", type=float, help="Weight: spatial gradient loss")
    parser.add_argument("--w_adv", type=float, help="Weight: adversarial generator loss")

    # Discriminator
    parser.add_argument("--disc_lr", type=float, help="Discriminator learning rate (default: same as --lr)")
    parser.add_argument("--disc_start", type=int, help="Epoch to start discriminator training (default: 0)")

    # Training configuration
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, help="Weight decay for optimizer")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--patience", type=int, help="Patience for early stopping")
    parser.add_argument("--grad_acc", type=int, help="Gradient accumulation steps")

    # System configuration
    parser.add_argument("--workers", type=int, help="Number of dataloader worker processes")
    parser.add_argument("--device", type=str, help="Device identifier")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_dir", type=str, help="Directory for saving checkpoints")

    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}

    configs = args.pop("config")
    base = {}
    for c in configs:
        base.update(yaml_load(c))

    args = {**base, **args}
    config = IterableSimpleNamespace(**args)
    main(config)
