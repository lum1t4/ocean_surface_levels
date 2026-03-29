import torch
from einops import rearrange, repeat
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F


def get_time_embedding(time_steps: torch.Tensor, emb_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    if emb_dim % 2 != 0:
        raise ValueError("Embedding dimension must be divisible by 2.")

    half = emb_dim // 2
    factor = 10000 ** (
        torch.arange(start=0, end=half, dtype=torch.float32, device=time_steps.device)
        / half
    )
    angles = time_steps[:, None].float() / factor[None]
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class LatteConfig(BaseModel):
    num_channels: int = 4
    image_size: int = 32
    patch_size: int = 2
    out_channels: int | None = None

    num_layers: int = 12
    hidden_size: int = 384
    num_attention_heads: int = 6
    mlp_ratio: float = 4.0

    timestep_emb_dim: int = 384
    max_temporal_positions: int = 16
    hidden_dropout_prob: float = 0.0
    qkv_bias: bool = True


class PatchEmbedder(nn.Module):
    def __init__(self, config: LatteConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class TimestepEmbedder(nn.Module):
    def __init__(self, config: LatteConfig):
        super().__init__()
        self.timestep_emb_dim = config.timestep_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(config.timestep_emb_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(get_time_embedding(t, self.timestep_emb_dim))


class Attention(nn.Module):
    def __init__(self, config: LatteConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={config.hidden_size} must be divisible by "
                f"num_attention_heads={config.num_attention_heads}"
            )
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout_p = config.hidden_dropout_prob

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, seq_len, 3, self.num_attention_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(batch, seq_len, hidden)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config: LatteConfig):
        super().__init__()
        hidden_size = config.hidden_size
        mlp_dim = int(hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(config.hidden_dropout_prob)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.drop2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, config: LatteConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(config)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )

        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        x = x + gate_msa.unsqueeze(1) * h

        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class FinalLayer(nn.Module):
    def __init__(self, config: LatteConfig):
        super().__init__()
        out_channels = config.num_channels if config.out_channels is None else config.out_channels
        self.norm_final = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True),
        )
        self.linear = nn.Linear(
            config.hidden_size,
            config.patch_size * config.patch_size * out_channels,
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class Latte(nn.Module):
    """
    Video Latent Diffusion Transformer with alternating spatial/temporal DiT blocks.
    Input/Output shape: (B, T, C, H, W).
    """

    def __init__(self, config: LatteConfig):
        super().__init__()
        if config.num_layers % 2 != 0:
            raise ValueError("num_layers must be even to alternate spatial/temporal blocks.")
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={config.hidden_size} must be divisible by "
                f"num_attention_heads={config.num_attention_heads}"
            )

        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.out_channels = config.num_channels if config.out_channels is None else config.out_channels

        if config.image_size % config.patch_size != 0:
            raise ValueError(
                f"image_size={config.image_size} must be divisible by patch_size={config.patch_size}"
            )
        self.grid_size = config.image_size // config.patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.x_embedder = PatchEmbedder(config)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.hidden_size))
        self.temp_embed = nn.Parameter(
            torch.zeros(1, config.max_temporal_positions, config.hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temp_embed, std=0.02)

        self.t_embedder = TimestepEmbedder(config)

        self.blocks = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_layers)]
        )
        self.final_layer = FinalLayer(config)

    def _prepare_timesteps(self, t: torch.Tensor | int | list[int], batch: int, device):
        t = torch.as_tensor(t, device=device, dtype=torch.long)
        if t.ndim == 0:
            t = t.repeat(batch)
        if t.ndim != 1 or t.shape[0] != batch:
            raise ValueError(f"Expected timesteps shape ({batch},), got {tuple(t.shape)}")
        return t

    def _get_temporal_embedding(self, num_frames: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        temp_embed = self.temp_embed.to(device=device, dtype=dtype)
        if num_frames == temp_embed.shape[1]:
            return temp_embed
        temp_embed = temp_embed.transpose(1, 2)
        temp_embed = F.interpolate(
            temp_embed, size=num_frames, mode="linear", align_corners=False
        )
        return temp_embed.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor, t: torch.Tensor | int | list[int]):
        batch_size, num_frames, channels, height, width = x.shape
        if channels != self.config.num_channels:
            raise ValueError(
                f"Expected {self.config.num_channels} channels, got {channels}."
            )
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Spatial shape {(height, width)} must be divisible by patch_size={self.patch_size}."
            )

        x = rearrange(x, "b f c h w -> (b f) c h w")
        out = self.x_embedder(x)
        num_patch_tokens = out.shape[1]
        if num_patch_tokens != self.num_patches:
            raise ValueError(
                f"Patch count mismatch ({num_patch_tokens} != {self.num_patches}). "
                "Check image_size/patch_size."
            )
        out = out + self.pos_embed

        timesteps = self._prepare_timesteps(t, batch_size, x.device)
        t_emb = self.t_embedder(timesteps)
        t_emb_spatial = repeat(t_emb, "b d -> (b f) d", f=num_frames)
        t_emb_temporal = repeat(t_emb, "b d -> (b p) d", p=num_patch_tokens)

        temporal_pos_embed = self._get_temporal_embedding(
            num_frames=num_frames, dtype=out.dtype, device=out.device
        )

        for layer_idx in range(0, len(self.blocks), 2):
            spatial_layer = self.blocks[layer_idx]
            temporal_layer = self.blocks[layer_idx + 1]

            out = spatial_layer(out, t_emb_spatial)
            out = rearrange(out, "(b f) p d -> (b p) f d", b=batch_size, f=num_frames)

            if layer_idx == 0:
                out = out + temporal_pos_embed.expand(out.shape[0], -1, -1)

            out = temporal_layer(out, t_emb_temporal)
            out = rearrange(
                out,
                "(b p) f d -> (b f) p d",
                b=batch_size,
                p=num_patch_tokens,
                f=num_frames,
            )

        out = self.final_layer(out, t_emb_spatial)
        out = rearrange(
            out,
            "b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)",
            ph=self.patch_size,
            pw=self.patch_size,
            nh=height // self.patch_size,
            nw=width // self.patch_size,
            c=self.out_channels,
        )
        out = out.reshape(batch_size, num_frames, self.out_channels, height, width)
        return out
