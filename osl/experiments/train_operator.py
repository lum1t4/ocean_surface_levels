import math
import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import tqdm
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cmocean

from osl.core.pytorch import RANK, device_memory_used, model_get_num_params
from osl.core.train import TrainContext
from osl.core.utils import LOGGER, IterableSimpleNamespace, yaml_load
from osl.model import load_model
from osl.data import SequenceDataset, dataset_stats_load
from osl.model.autoencoderkl import AutoencoderKLFlux2
from osl.core.metrics import ssim_metric

matplotlib.use("Agg")


# ----------------------------------------------------------------
# Time operators
# ----------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10_000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FiLM(nn.Module):
    def __init__(self, time_emb_dim: int = 256, latent_channels: int = 32):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.film = nn.Linear(time_emb_dim, latent_channels * 2)

    def forward(self, z: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(time)
        t_emb = self.time_mlp(t_emb)
        film_params = self.film(t_emb)
        gamma, beta = film_params.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        z = z * (1 + gamma) + beta
        return z

    def proximity_loss(self, zi: torch.Tensor, zj: torch.Tensor, zt: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return nn.functional.smooth_l1_loss(zt, zj)


class Koopman(nn.Module):
    def __init__(self, latent_channels: int = 32):
        super().__init__()
        self.G = nn.Parameter(torch.eye(latent_channels) + torch.randn(latent_channels, latent_channels) * 0.01)

    def forward(self, z: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape
        z_pred = torch.zeros_like(z)
        for i in range(B):
            k_i = int(k[i].item())
            K = torch.linalg.matrix_power(self.G, k_i)
            z_flat = z[i].view(C, H * W)
            z_k_flat = torch.matmul(K, z_flat)
            z_pred[i] = z_k_flat.view(C, H, W)
        return z_pred

    def proximity_loss(self, zi: torch.Tensor, zj: torch.Tensor, zt: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return nn.functional.smooth_l1_loss(zt, zj)


class FlowMatchingOperator(nn.Module):
    """
    Conditional Flow Matching time operator.

    Learns a velocity field v(z_t, t, k) that transports z_0 -> z_k.
    - Training: conditional flow matching loss on the velocity (no ODE solve).
    - Inference: Euler integration of the learned velocity field.
    """

    def __init__(self, latent_channels: int = 32, time_emb_dim: int = 128, num_inference_steps: int = 10):
        super().__init__()
        self.num_inference_steps = num_inference_steps

        self.t_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.k_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, latent_channels * 2),
        )

        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels * 2, 3, padding=1),
            nn.GroupNorm(8, latent_channels * 2),
            nn.GELU(),
            nn.Conv2d(latent_channels * 2, latent_channels * 2, 3, padding=1),
            nn.GroupNorm(8, latent_channels * 2),
            nn.GELU(),
            nn.Conv2d(latent_channels * 2, latent_channels, 3, padding=1),
        )

    def velocity(self, z: torch.Tensor, t: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_emb(t * 1000)
        k_emb = self.k_emb(k.float())
        cond = self.cond_mlp(torch.cat([t_emb, k_emb], dim=-1))
        gamma, beta = cond.chunk(2, dim=-1)
        z_mod = z * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.net(z_mod)

    def flow_loss(self, z_0: torch.Tensor, z_1: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B = z_0.shape[0]
        t = torch.rand(B, device=z_0.device)
        t_spatial = t[:, None, None, None]
        z_t = (1 - t_spatial) * z_0 + t_spatial * z_1
        target_v = z_1 - z_0
        pred_v = self.velocity(z_t, t, k)
        return nn.functional.mse_loss(pred_v, target_v)

    def forward(self, z: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        dt = 1.0 / self.num_inference_steps
        z_t = z
        for i in range(self.num_inference_steps):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            v = self.velocity(z_t, t, k)
            z_t = z_t + dt * v
        return z_t

    def proximity_loss(self, zi: torch.Tensor, zj: torch.Tensor, zt: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return self.flow_loss(zi, zj, k)


class ResBlock(nn.Module):
    """Residual block with time conditioning via additive projection."""

    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class ResNetOperator(nn.Module):
    """
    ResNet-style latent time operator with time-conditioned residual blocks.

    Predicts the residual delta in latent space: z_k = z_0 + f(z_0, k).
    Spatial conv layers allow learning complex spatiotemporal transformations.
    """

    def __init__(
        self,
        latent_channels: int = 32,
        hidden_channels: int = 128,
        num_blocks: int = 6,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.in_conv = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(hidden_channels, time_emb_dim) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(8, hidden_channels)
        self.out_conv = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        self.act = nn.GELU()

        # Zero-init the last conv so the residual starts as identity
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(k.float()))
        h = self.in_conv(z)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.act(self.out_norm(h))
        h = self.out_conv(h)
        return z + h

    def proximity_loss(self, zi: torch.Tensor, zj: torch.Tensor, zt: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return nn.functional.smooth_l1_loss(zt, zj)


# ----------------------------------------------------------------
# Losses
# ----------------------------------------------------------------
def gradient_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_dx = preds[..., :, 1:] - preds[..., :, :-1]
    target_dx = targets[..., :, 1:] - targets[..., :, :-1]
    pred_dy = preds[..., 1:, :] - preds[..., :-1, :]
    target_dy = targets[..., 1:, :] - targets[..., :-1, :]
    diff_dx = (pred_dx - target_dx).abs()
    diff_dy = (pred_dy - target_dy).abs()
    return 0.5 * (diff_dx.mean() + diff_dy.mean())


def mse_loss(inputs: torch.Tensor, targets: torch.Tensor):
    return nn.functional.mse_loss(inputs, targets)



# ----------------------------------------------------------------
# Train step
# ----------------------------------------------------------------
def model_train_step(ctx: TrainContext, batch_idx: int, batch: dict) -> tuple:
    inputs = batch["inputs"]
    B, T = inputs.shape[:2]

    ts = torch.randint(1, T, (B,))
    xi = inputs[:, 0]
    xj = inputs[torch.arange(B), ts]

    ts = ts.to(ctx.device, non_blocking=True, dtype=torch.long)
    xi = xi.to(ctx.device, non_blocking=True, dtype=torch.float32)
    xj = xj.to(ctx.device, non_blocking=True, dtype=torch.float32)

    w_pred = ctx.config.get("w_pred", 1.0)
    w_prox = ctx.config.get("w_prox", 0.0)
    w_grad = ctx.config.get("w_grad", 0.0)

    # Encode source frame (frozen encoder)
    with torch.no_grad():
        mu_i, sigma_i = ctx.vae.encode(xi)
        zi = ctx.vae.sample(mu_i, sigma_i)

    # Advance in latent space with the time operator
    zt = ctx.top(zi, ts)

    # Decode prediction (frozen decoder)
    with torch.no_grad():
        yt = ctx.vae.decode(zt)

    losses = {}

    # Prediction loss in pixel space
    prediction = mse_loss(yt, xj)
    losses["prediction"] = prediction
    loss = w_pred * prediction

    if w_grad > 0:
        grad = gradient_loss(yt, xj)
        losses["grad"] = grad
        loss = loss + w_grad * grad

    # Proximity loss in latent space
    if w_prox > 0:
        with torch.no_grad():
            mu_j, sigma_j = ctx.vae.encode(xj)
            zj = ctx.vae.sample(mu_j, sigma_j)
        prox = ctx.top.proximity_loss(zi, zj, zt, ts)
        losses["proximity"] = prox
        loss = loss + w_prox * prox

    return yt, xj, loss, losses


# ----------------------------------------------------------------
# Generation (rollout)
# ----------------------------------------------------------------
@torch.inference_mode()
def model_generate(autoencoder: AutoencoderKLFlux2, inputs: torch.Tensor, top: nn.Module, horizon: int) -> torch.Tensor:
    batch_size = inputs.shape[0]
    outputs = []
    for i in range(1, horizon + 1):
        step = torch.full((batch_size,), i, device=inputs.device, dtype=torch.long)
        mu, sigma = autoencoder.encode(inputs)
        z = autoencoder.sample(mu, sigma)
        z = top(z, step)
        prediction = autoencoder.decode(z)
        outputs.append(prediction)
    return torch.stack(outputs, dim=1)


def model_valid_step(ctx: TrainContext, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["inputs"].to(ctx.device, non_blocking=True, dtype=torch.float32)
    targets = batch["targets"].to(ctx.device, non_blocking=True, dtype=torch.float32)
    G = targets.shape[1]
    H = ctx.config.seq_length - 1
    chunks = math.ceil(G / H)

    outputs = []
    frame = inputs.squeeze(1)
    for chunk_idx in range(chunks):
        chunk_length = min(H, G - chunk_idx * H)
        chunk = model_generate(ctx.vae, frame, ctx.top, chunk_length)
        outputs.append(chunk)
        frame = chunk[:, -1]

    outputs = torch.concat(outputs, dim=1)
    return outputs, targets


# ----------------------------------------------------------------
# Plot
# ----------------------------------------------------------------
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

        mse_val = np.nanmean((p - t) ** 2)

        axes[0, j].imshow(t, vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance, origin="upper")
        axes[0, j].set_title(f"Target day {day}")
        axes[0, j].axis("off")

        axes[1, j].imshow(p, vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance, origin="upper")
        axes[1, j].set_title(f"Pred day {day}\nMSE={mse_val:.6f}")
        axes[1, j].axis("off")

    fig.tight_layout()
    dst = ctx.plot_dir / f"valid_epoch_{ctx.curr_iter}.png"
    fig.savefig(dst, dpi=120)
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

    LOGGER.info(f"Train set size: {train_bound[1] - train_bound[0]}  "
                f"Valid set size: {valid_bound[1] - valid_bound[0]}")

    reader_args = dict(vars=config.variables, seq_length=config.seq_length, seq_stride=config.seq_length, normalize=config.normalize, stats=stats)
    roller_args = {**reader_args, **dict(seq_seek=1, seq_length=30, seq_stride=30)}
    loader_args = dict(batch_size=config.batch_size, num_workers=config.workers, pin_memory=False)

    if ctx.device.type not in {"cpu", "mps"}:
        loader_args["pin_memory"] = True

    ctx.train_set = SequenceDataset(data, bound=train_bound, **reader_args)
    ctx.valid_set = SequenceDataset(data, bound=valid_bound, **roller_args)

    ctx.train_loader = DataLoader(ctx.train_set, shuffle=True, **loader_args)
    ctx.valid_loader = DataLoader(ctx.valid_set, shuffle=False, drop_last=False, **loader_args)
    return ctx


# ----------------------------------------------------------------
# Train epoch
# ----------------------------------------------------------------
def schedule_train_epoch(ctx: TrainContext):
    ctx.top.train()

    running_loss = 0.0
    running_norm = 0.0
    running_comp = defaultdict(float)

    progress = enumerate(ctx.train_loader)
    if RANK in {-1, 0}:
        LOGGER.info(("\n" + "%11s" * 4) % ("Epoch", "GPU_mem", "Loss", "Norm"))
        progress = tqdm.tqdm(progress, total=len(ctx.train_loader))

    ctx.optimizer.zero_grad()

    for batch_idx, batch in progress:
        yt, xj, gen_loss, losses = model_train_step(ctx, batch_idx, batch)
        (gen_loss / ctx.config.grad_acc).backward()

        if (batch_idx + 1) % ctx.config.grad_acc == 0:
            step_idx = (batch_idx + 1) // ctx.config.grad_acc

            norm = torch.nn.utils.clip_grad_norm_(ctx.top.parameters(), max_norm=10.0)
            ctx.optimizer.step()
            ctx.optimizer.zero_grad()

            running_norm = (running_norm * step_idx + norm.item()) / (step_idx + 1)
            running_loss = (running_loss * step_idx + gen_loss.item()) / (step_idx + 1)
            for k, v in losses.items():
                running_comp[k] = (running_comp[k] * step_idx + v.item()) / (step_idx + 1)

        if RANK in {-1, 0}:
            epoch_desc = f"{ctx.curr_iter + 1}/{ctx.config.epochs}"
            memory_used = device_memory_used(ctx.device)
            progress.set_description("%11s%11.4g%11.4g%11.4g" % (epoch_desc, memory_used, running_loss, running_norm))

    ctx.metrics["train/loss"] = running_loss
    ctx.metrics["train/norm"] = running_norm
    for k, v in running_comp.items():
        ctx.metrics[f"train/{k}"] = v


# ----------------------------------------------------------------
# Valid epoch (rollout forecasting evaluation)
# ----------------------------------------------------------------
@torch.inference_mode()
def schedule_valid_epoch(ctx: TrainContext):
    ctx.top.eval()

    progress = enumerate(ctx.valid_loader)
    if RANK in {-1, 0}:
        LOGGER.info("%11s" % "Val. Loss")
        progress = tqdm.tqdm(progress, total=len(ctx.valid_loader))

    lead_days = [1, 7, 30]
    running_loss = 0.0
    running_leads_mse = torch.zeros(len(lead_days), device=ctx.device)
    running_leads_ssim = torch.zeros(len(lead_days), device=ctx.device)

    for batch_idx, batch in progress:
        outputs, targets = model_valid_step(ctx, batch)
        loss = nn.functional.mse_loss(outputs, targets)
        running_loss = (running_loss * batch_idx + loss.item()) / (batch_idx + 1)

        if RANK in {-1, 0}:
            progress.set_description("%11.4g" % running_loss)
            if batch_idx == 0:
                plot_valid_batch(ctx, outputs, targets)

        for j, lead_day in enumerate(lead_days):
            pred_frame = outputs[:, lead_day - 1]
            tgt_frame = targets[:, lead_day - 1]
            lead_mse = nn.functional.mse_loss(pred_frame, tgt_frame)
            lead_ssim = ssim_metric(pred_frame, tgt_frame)
            running_leads_mse[j] = (running_leads_mse[j] * batch_idx + lead_mse) / (batch_idx + 1)
            running_leads_ssim[j] = (running_leads_ssim[j] * batch_idx + lead_ssim) / (batch_idx + 1)

    for j, lead_day in enumerate(lead_days):
        ctx.metrics[f"valid/MSE@{lead_day}"] = running_leads_mse[j].item()
        ctx.metrics[f"valid/RMSE@{lead_day}"] = torch.sqrt(running_leads_mse[j]).item()
        ctx.metrics[f"valid/SSIM@{lead_day}"] = running_leads_ssim[j].item()

    ctx.metrics["valid/loss"] = running_loss
    ctx.tracker.log(ctx.metrics, step=ctx.curr_iter)


def get_time_operator(config):
    if config.time_operator == "koopman":
        return Koopman()
    if config.time_operator == "flow_matching":
        return FlowMatchingOperator(num_inference_steps=config.get("fm_inference_steps", 10))
    if config.time_operator == "film":
        return FiLM()
    if config.time_operator == "resnet":
        return ResNetOperator(
            hidden_channels=config.get("op_hidden_channels", 128),
            num_blocks=config.get("op_num_blocks", 6),
        )


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main(config: IterableSimpleNamespace):
    ctx = TrainContext(config)
    schedule_setup_dataset(ctx, config)

    num_vars = len(ctx.config.variables)
    model_overrides = {"in_channels": num_vars, "out_channels": num_vars}

    # Load pretrained autoencoder and freeze
    ctx.vae = load_model(ctx.config.model, config=model_overrides, weights=ctx.config.autoencoder_weights).to(ctx.device)
    ctx.vae.eval()
    
    for p in ctx.vae.parameters():
        p.requires_grad = False
    LOGGER.info(f"Loaded frozen autoencoder from {ctx.config.autoencoder_weights}")

    # Time operator (this is what we train)
    ctx.top = get_time_operator(config).to(ctx.device)
    ctx.model = ctx.top  # for checkpointing compatibility

    np_ae = model_get_num_params(ctx.vae)
    np_top = model_get_num_params(ctx.top)
    ctx.metrics["model/ae_params"] = np_ae
    ctx.metrics["model/top_params"] = np_top

    ctx.optimizer = optim.Adam(ctx.top.parameters(), lr=ctx.config.lr, weight_decay=ctx.config.weight_decay)
    ctx.scheduler = optim.lr_scheduler.ReduceLROnPlateau(ctx.optimizer, mode="min", factor=0.5, patience=5)

    LOGGER.info(f"Autoencoder: {config.model} ({np_ae:,} params, frozen)")
    LOGGER.info(f"Time operator: {config.time_operator} ({np_top:,} params, trainable)")
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

    parser = argparse.ArgumentParser(description="Time operator training with frozen autoencoder")

    parser.add_argument("--config", action="append", default=[], help="Configuration file path")
    parser.add_argument("--name", type=str, help="Run name")
    parser.add_argument("--tracker", type=str, help="Tracker")

    # Data configuration
    parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    parser.add_argument("--seq_length", type=int, help="Sequence length for training")
    parser.add_argument("--seq_stride", type=int, help="Stride for sequence sampling")
    parser.add_argument("--variables", type=str, nargs="+", help="Variables to use")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="Enable normalization")

    # Model configuration
    parser.add_argument("--model", type=str, help="Autoencoder model name in registry")
    parser.add_argument("--autoencoder_weights", type=str, required=True, help="Path to pretrained autoencoder checkpoint")

    # Time operator
    parser.add_argument("--time_operator", type=str, help="Time operator: film, koopman, flow_matching, resnet")

    # Loss weights
    parser.add_argument("--w_pred", type=float, help="Weight: prediction loss (pixel space)")
    parser.add_argument("--w_prox", type=float, help="Weight: latent proximity loss")
    parser.add_argument("--w_grad", type=float, help="Weight: spatial gradient loss")

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
