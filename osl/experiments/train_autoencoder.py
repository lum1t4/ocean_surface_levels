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
from osl.core.metrics import ssim_metric
from osl.model.autoencoderkl import AutoencoderKLFlux2

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

    Args:
        latent_channels: Number of channels in the latent space.
        time_emb_dim: Dimensionality of time embeddings.
        num_inference_steps: Number of Euler steps during inference.
    """

    def __init__(self, latent_channels: int = 32, time_emb_dim: int = 128, num_inference_steps: int = 10):
        super().__init__()
        self.num_inference_steps = num_inference_steps

        # Separate embeddings for interpolation time t and step-ahead k
        self.t_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.k_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim * 2, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, latent_channels * 2),
        )

        # Lightweight velocity network (spatial convolutions + FiLM conditioning)
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
        """Predict velocity field v(z, t, k).

        Args:
            z: (B, C, H, W) latent at interpolation time t.
            t: (B,) continuous interpolation time in [0, 1].
            k: (B,) integer step-ahead (used as conditioning).
        """
        t_emb = self.t_emb(t * 1000)  # scale for richer sinusoidal features
        k_emb = self.k_emb(k.float())
        cond = self.cond_mlp(torch.cat([t_emb, k_emb], dim=-1))
        gamma, beta = cond.chunk(2, dim=-1)
        z_mod = z * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.net(z_mod)

    def flow_loss(self, z_0: torch.Tensor, z_1: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Conditional flow matching loss (direct velocity supervision, no ODE solve).

        Optimal transport interpolation: z_t = (1-t)*z_0 + t*z_1
        Target velocity: v* = z_1 - z_0

        Args:
            z_0: (B, C, H, W) source latent.
            z_1: (B, C, H, W) target latent.
            k: (B,) integer step-ahead.
        """
        B = z_0.shape[0]
        t = torch.rand(B, device=z_0.device)
        t_spatial = t[:, None, None, None]

        z_t = (1 - t_spatial) * z_0 + t_spatial * z_1
        target_v = z_1 - z_0
        pred_v = self.velocity(z_t, t, k)
        return nn.functional.mse_loss(pred_v, target_v)

    def forward(self, z: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Euler integration from z_0 over the learned velocity field.

        Args:
            z: (B, C, H, W) source latent.
            k: (B,) integer step-ahead.
        """
        dt = 1.0 / self.num_inference_steps
        z_t = z
        for i in range(self.num_inference_steps):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            v = self.velocity(z_t, t, k)
            z_t = z_t + dt * v
        return z_t

    def proximity_loss(self, zi: torch.Tensor, zj: torch.Tensor, zt: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return self.flow_loss(zi, zj, k)


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
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i], stride=strides[i],
                          padding=paddings[i],
                          bias=False if i != 0 else True),
                nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                activation if i != len(layers_dim) - 2 else nn.Identity(),
            )
            for i in range(len(layers_dim) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ----------------------------------------------------------------
# Losses
# ----------------------------------------------------------------
def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for the discriminator."""
    return 0.5 * (torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean())


def _broadcast_mask(mask: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    while mask.ndim < ref_tensor.ndim:
        mask = mask.unsqueeze(0)
    return mask


def gradient_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
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


def mse_loss(inputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None):
    if mask is None:
        return nn.functional.mse_loss(inputs, targets)
    diff = (inputs - targets) * mask
    denom = (mask.sum() * inputs.shape[0] * inputs.shape[1] * inputs.shape[2]).clamp_min(1.0)
    return (diff ** 2).sum() / denom


# ----------------------------------------------------------------
# Train step (generator)
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

    # Encode source frame
    mu_i, sigma_i = ctx.model.encode(xi)
    zi = ctx.model.sample(mu_i, sigma_i)

    # Advance in latent space and decode
    zt = ctx.top(zi, ts)
    yt = ctx.model.decode(zt)

    # Read loss weights from config
    w_pred = ctx.config.get('w_pred', 0.0)
    w_prox = ctx.config.get('w_prox', 0.0)
    w_grad = ctx.config.get('w_grad', 0.0)
    w_recon_xi = ctx.config.get('w_recon_xi', 0.0)
    w_recon_xj = ctx.config.get('w_recon_xj', 0.0)
    w_adv = ctx.config.get('w_adv', 0.0)
    w_kl = ctx.config.get('w_kl', 0.0)


    # Encode target frame when needed for latent-space losses
    mu_j, sigma_j = ctx.model.encode(xj)
    zj = ctx.model.sample(mu_j, sigma_j)

    def similarity_loss(inputs: torch.Tensor, targets: torch.Tensor, mask=None) -> torch.Tensor:
        loss = mse_loss(inputs, targets, mask)
        if w_grad > 0:
            loss += w_grad * gradient_loss(inputs, targets, mask)
        return loss

    # 1. Prediction loss: decode(time_op(encode(xi))) vs xj
    prediction = similarity_loss(yt, xj)
    losses = {"prediction": prediction}
    loss = w_pred * prediction

    # 2. Reconstruction loss on xi: decode(encode(xi)) vs xi
    if w_recon_xi > 0:
        yi = ctx.model.decode(zi)
        recon_i = similarity_loss(yi, xi)
        losses["recon_input"] = recon_i
        loss = loss + w_recon_xi * recon_i

    # 3. Reconstruction loss on xj: decode(encode(xj)) vs xj
    if w_recon_xj > 0:
        yj = ctx.model.decode(zj)
        recon_j = similarity_loss(yj, xj)
        losses["recon_target"] = recon_j
        loss = loss + w_recon_xj * recon_j

    # 4. Proximity loss (dispatched to time operator: smooth_l1 for FiLM/Koopman, flow matching for FM)
    if w_prox > 0:
        prox = ctx.top.proximity_loss(zi, zj, zt, ts)
        losses["proximity"] = prox
        loss = loss + w_prox * prox

    # 5. KL divergence
    if w_kl > 0:
        kl = kl_divergence(mu_i, sigma_i)
        losses["kl"] = kl
        loss = loss + w_kl * kl

    # 7. Adversarial generator loss: fool the discriminator
    if w_adv > 0 and ctx.discriminator is not None:
        disc_start = ctx.config.get('disc_start', 0)
        if ctx.curr_iter >= disc_start:
            fake_logits = ctx.discriminator(yt)
            adv_g_loss = -fake_logits.mean()
            losses["adv_gen"] = adv_g_loss
            loss = loss + w_adv * adv_g_loss

    return yt, xj, loss, losses


# ----------------------------------------------------------------
# Train step (discriminator)
# ----------------------------------------------------------------
def disc_train_step(ctx: TrainContext, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    real_logits = ctx.discriminator(real)
    fake_logits = ctx.discriminator(fake.detach())
    loss = hinge_d_loss(real_logits, fake_logits)
    return loss


# ----------------------------------------------------------------
# Generation
# ----------------------------------------------------------------
@torch.inference_mode()
def model_generate(model: AutoencoderKLFlux2, inputs: torch.Tensor, top: nn.Module, horizon: int) -> torch.Tensor:
    batch_size = inputs.shape[0]
    outputs = []
    for i in range(1, horizon + 1):
        step = torch.full((batch_size,), i, device=inputs.device, dtype=torch.long)
        mu, sigma = model.encode(inputs)
        z = model.sample(mu, sigma)
        z = top(z, step)
        prediction = model.decode(z)
        outputs.append(prediction)
    return torch.stack(outputs, dim=1)


def model_valid_step(ctx: TrainContext, batch_idx: int, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["inputs"].to(ctx.device, non_blocking=True, dtype=torch.float32)
    targets = batch["targets"].to(ctx.device, non_blocking=True, dtype=torch.float32)
    G = targets.shape[1]
    H = ctx.config.seq_length - 1
    outputs = []
    chunks = math.ceil(G / H)

    inputs = inputs.squeeze(1)
    for chunk_idx in range(chunks):
        chunk_length = min(H, G - chunk_idx * H)
        chunk = model_generate(ctx.model, inputs, ctx.top, chunk_length)
        outputs.append(chunk)
        inputs = chunk[:, -1]

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

    if ctx.device.type not in {'cpu', 'mps'}:
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
    ctx.model.train()
    ctx.top.train()

    has_disc = ctx.discriminator is not None and ctx.config.get('w_adv', 0.0) > 0
    disc_active = has_disc and ctx.curr_iter >= ctx.config.get('disc_start', 0)
    if disc_active:
        ctx.discriminator.train()

    running_loss = 0.0
    running_disc_loss = 0.0
    running_norm = 0.0
    running_comp = defaultdict(int)
    progress = enumerate(ctx.train_loader)
    if RANK in {-1, 0}:
        LOGGER.info(("\n" + "%11s" * 4) % ("Epoch", "GPU_mem", "Loss", "Norm"))
        progress = tqdm.tqdm(progress, total=len(ctx.train_loader))

    ctx.optimizer.zero_grad()
    if disc_active:
        ctx.optimizer_d.zero_grad()

    for batch_idx, batch in progress:
        # --- Generator step ---
        yt, xj, gen_loss, losses = model_train_step(ctx, batch_idx, batch)
        (gen_loss / ctx.config.grad_acc).backward()

        # --- Discriminator step ---
        if disc_active:
            d_loss = disc_train_step(ctx, xj, yt)
            (d_loss / ctx.config.grad_acc).backward()

        if (batch_idx + 1) % ctx.config.grad_acc == 0:
            step_idx = (batch_idx + 1) // ctx.config.grad_acc

            # Update generator (model + time_operator)
            gen_params = list(ctx.model.parameters()) + list(ctx.top.parameters())
            norm = torch.nn.utils.clip_grad_norm_(gen_params, max_norm=10.0)
            ctx.optimizer.step()
            ctx.optimizer.zero_grad()

            running_norm = (running_norm * step_idx + norm.item()) / (step_idx + 1)
            running_loss = (running_loss * step_idx + gen_loss.item()) / (step_idx + 1)
            for k, v in losses.items():
                running_comp[k] = (running_comp[k] * step_idx + v.item()) / (step_idx + 1)

            # Update discriminator
            if disc_active:
                torch.nn.utils.clip_grad_norm_(ctx.discriminator.parameters(), max_norm=10.0)
                ctx.optimizer_d.step()
                ctx.optimizer_d.zero_grad()
                running_disc_loss = (running_disc_loss * step_idx + d_loss.item()) / (step_idx + 1)

        if RANK in {-1, 0}:
            epoch_desc = f"{ctx.curr_iter + 1}/{ctx.config.epochs}"
            memory_used = device_memory_used(ctx.device)
            progress.set_description("%11s%11.4g%11.4g%11.4g" % (epoch_desc, memory_used, running_loss, running_norm))

    ctx.metrics["train/loss"] = running_loss
    ctx.metrics["train/norm"] = running_norm
    if disc_active:
        ctx.metrics["train/disc_loss"] = running_disc_loss
    for k, v in running_comp.items():
        ctx.metrics[f"train/{k}"] = v


# ----------------------------------------------------------------
# Valid epoch
# ----------------------------------------------------------------
@torch.inference_mode()
def schedule_valid_epoch(ctx: TrainContext):
    ctx.model.eval()
    ctx.top.eval()

    progress = enumerate(ctx.valid_loader)
    if RANK in {-1, 0}:
        LOGGER.info("%11s" % "Val. Loss")
        progress = tqdm.tqdm(progress, total=len(ctx.valid_loader))

    lead_days = [1, 7, 30]
    running_loss = 0.0
    running_leads = torch.zeros(len(lead_days), device=ctx.device)
    running_leads_ssim = torch.zeros(len(lead_days), device=ctx.device)
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


def get_time_operator(config):
    if config.time_operator == "koopman":
        return Koopman()
    if config.time_operator == "flow_matching":
        return FlowMatchingOperator(num_inference_steps=config.get('fm_inference_steps', 10))
    if config.time_operator == "film":
        return FiLM()


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main(config: IterableSimpleNamespace):
    ctx = TrainContext(config)
    schedule_setup_dataset(ctx, config)

    num_vars = len(ctx.config.variables)
    model_overrides = {"in_channels": num_vars, "out_channels": num_vars}


    ctx.model = load_model(ctx.config.model, config=model_overrides).to(ctx.device)
    ctx.top = get_time_operator(config).to(ctx.device)
    

    np_model = ctx.metrics["model/params"] = model_get_num_params(ctx.model)
    np_top = ctx.metrics["model/top/params"] = model_get_num_params(ctx.top)

    param_group = list(ctx.model.parameters()) + list(ctx.top.parameters())
    ctx.optimizer = optim.Adam(param_group, lr=ctx.config.lr, weight_decay=ctx.config.weight_decay)
    ctx.scheduler = optim.lr_scheduler.ReduceLROnPlateau(ctx.optimizer, mode="min", factor=0.5, patience=5)


    # Discriminator (only instantiated when adversarial weight > 0)
    w_adv = config.get('w_adv', 0.0)
    disc_lr = config.get('disc_lr', ctx.config.lr)
    disc_start = config.get('disc_start', 0)

    if w_adv > 0:
        ctx.discriminator = Discriminator(in_channels=num_vars).to(ctx.device)
        ctx.optimizer_d = optim.Adam(ctx.discriminator.parameters(), lr=disc_lr, weight_decay=ctx.config.weight_decay)
        num_params = model_get_num_params(ctx.discriminator)
        ctx.metrics["model/disc/params"] = num_params
        LOGGER.info(f"Discriminator: {num_params:,} params, lr={disc_lr}, start_epoch={disc_start}")
    else:
        ctx.discriminator = None
        ctx.optimizer_d = None

    LOGGER.info(f"Model: {config.model}, Params: {np_model:,}, TimeOp: {config.time_operator} ({np_top:,})")
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Temporal VAE: progressive forecasting in latent space")

    parser.add_argument("--config", action="append", default=[], help="Configuration file path")
    parser.add_argument("--name", type=str, help="Run name")
    parser.add_argument("--tracker", type=str, help="Tracker")

    # Data configuration
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--seq_length', type=int, help='Sequence length for training')
    parser.add_argument("--seq_stride", type=int, help="Stride for sequence sampling")
    parser.add_argument('--variables', type=str, nargs='+', help='Variables to use')
    parser.add_argument('--normalize', action=argparse.BooleanOptionalAction, help='Enable normalization')

    # Model configuration
    parser.add_argument('--model', type=str, help='Model name in registry')

    # Time operator
    parser.add_argument('--time_operator', type=str, help="Time operator (film, koopman, flow_matching)")
    parser.add_argument('--fm_inference_steps', type=int, help="Euler steps for flow matching inference (default: 10)")

    # Loss weights (semantic names)
    parser.add_argument('--w_pred', type=float, help="Weight: prediction loss")
    parser.add_argument('--w_recon_xi', type=float, help="Weight: reconstruction of input frame")
    parser.add_argument('--w_recon_xj', type=float, help="Weight: reconstruction of target frame")
    parser.add_argument('--w_prox', type=float, help="Weight: latent proximity (time_op output vs encoded target)")
    parser.add_argument('--w_kl', type=float, help="Weight: KL divergence regularization")
    parser.add_argument('--w_grad', type=float, help="Weight: spatial gradient loss")
    parser.add_argument('--w_adv', type=float, help="Weight: adversarial generator loss")

    # Discriminator
    parser.add_argument('--disc_lr', type=float, help="Discriminator learning rate (default: same as --lr)")
    parser.add_argument('--disc_start', type=int, help="Epoch to start discriminator training (default: 0)")

    # Training configuration
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--patience', type=int, help='Patience for early stopping')
    parser.add_argument('--grad_acc', type=int, help='Gradient accumulation steps')

    # System configuration
    parser.add_argument("--workers", type=int, help="Number of dataloader worker processes")
    parser.add_argument("--device", type=str, help="Device identifier")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_dir", type=str, help="Directory for saving checkpoints")

    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}

    configs = args.pop('config')
    base = {}
    for c in configs:
        base.update(yaml_load(c))

    args = {**base, **args}
    config = IterableSimpleNamespace(**args)
    main(config)
