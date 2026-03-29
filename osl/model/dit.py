import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


class DiTConfig(BaseModel):
    image_size: int | Sequence[int] = 32
    patch_size: int = 2
    in_channels: int = 4
    out_channels: int | None = None

    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0

    timestep_emb_dim: int = 256
    num_classes: int = 0
    class_dropout_prob: float = 0.0
    qkv_bias: bool = True
    attn_dropout_prob: float = 0.0
    mlp_dropout_prob: float = 0.0
    learn_sigma: bool = False


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _get_1d_sincos_pos_embed(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    half_dim = embed_dim // 2
    omega = torch.arange(half_dim, dtype=torch.float32, device=positions.device)
    omega = 1.0 / (10000 ** (omega / half_dim))
    out = positions.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
    grid_h_t = torch.arange(grid_h, dtype=torch.float32)
    grid_w_t = torch.arange(grid_w, dtype=torch.float32)
    yy = grid_h_t[:, None].expand(grid_h, grid_w).reshape(-1)
    xx = grid_w_t[None, :].expand(grid_h, grid_w).reshape(-1)
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, yy)
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, xx)
    return torch.cat([emb_h, emb_w], dim=1).unsqueeze(0)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 != 0:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(timesteps, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + int(use_cfg_embedding), hidden_size)

    def token_drop(self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.to(device=labels.device, dtype=torch.bool)
        dropped = torch.full_like(labels, self.num_classes)
        return torch.where(drop_ids, dropped, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True, dropout_prob: float = 0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout_prob = dropout_prob
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_prob if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(bsz, seq_len, hidden)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, dropout_prob: float = 0.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(mlp_hidden_dim, hidden_size)
        self.drop2 = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_dropout_prob: float = 0.0,
        mlp_dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=qkv_bias, dropout_prob=attn_dropout_prob)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio, dropout_prob=mlp_dropout_prob)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    """
    Diffusion Transformer compatible with official DiT checkpoints while remaining
    independent from Hugging Face and timm runtime dependencies.
    """

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"hidden_size={config.hidden_size} must be divisible by num_heads={config.num_heads}"
            )

        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.learn_sigma = config.learn_sigma

        if config.out_channels is not None:
            self.out_channels = config.out_channels
        else:
            self.out_channels = config.in_channels * 2 if config.learn_sigma else config.in_channels

        base_h, base_w = self._normalize_image_size(config.image_size)
        if base_h % self.patch_size != 0 or base_w % self.patch_size != 0:
            raise ValueError(
                f"image_size={config.image_size} must be divisible by patch_size={self.patch_size}"
            )
        self.base_grid_size = (base_h // self.patch_size, base_w // self.patch_size)

        self.x_embedder = PatchEmbed(
            in_channels=config.in_channels,
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
        )
        self.t_embedder = TimestepEmbedder(config.hidden_size, frequency_embedding_size=config.timestep_emb_dim)
        self.y_embedder = (
            LabelEmbedder(config.num_classes, config.hidden_size, config.class_dropout_prob)
            if config.num_classes > 0
            else None
        )

        num_patches = self.base_grid_size[0] * self.base_grid_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    config.hidden_size,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    attn_dropout_prob=config.attn_dropout_prob,
                    mlp_dropout_prob=config.mlp_dropout_prob,
                )
                for _ in range(config.depth)
            ]
        )
        self.final_layer = FinalLayer(config.hidden_size, config.patch_size, self.out_channels)
        self.initialize_weights()

    @staticmethod
    def _normalize_image_size(image_size: int | Sequence[int]) -> tuple[int, int]:
        if isinstance(image_size, int):
            return image_size, image_size
        if len(image_size) != 2:
            raise ValueError(f"image_size must be int or tuple/list(H, W), got {image_size}")
        return int(image_size[0]), int(image_size[1])

    def initialize_weights(self):
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = _get_2d_sincos_pos_embed(
            self.config.hidden_size,
            self.base_grid_size[0],
            self.base_grid_size[1],
        )
        self.pos_embed.data.copy_(pos_embed)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.x_embedder.proj.bias is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _prepare_timesteps(self, t: torch.Tensor | int | list[int], batch: int, device: torch.device) -> torch.Tensor:
        t = torch.as_tensor(t, device=device, dtype=torch.long)
        if t.ndim == 0:
            t = t.repeat(batch)
        if t.ndim != 1 or t.shape[0] != batch:
            raise ValueError(f"Expected timesteps shape ({batch},), got {tuple(t.shape)}")
        return t

    def _prepare_labels(
        self,
        y: torch.Tensor | int | list[int] | None,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        if y is None:
            return torch.zeros(batch, device=device, dtype=torch.long)
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        if y.ndim == 0:
            y = y.repeat(batch)
        if y.ndim != 1 or y.shape[0] != batch:
            raise ValueError(f"Expected labels shape ({batch},), got {tuple(y.shape)}")
        return y

    def _pad_to_patch_size(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, pad_h, pad_w

    def _get_pos_embed(self, grid_h: int, grid_w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pos_embed = self.pos_embed.to(device=device, dtype=dtype)
        base_h, base_w = self.base_grid_size
        if grid_h == base_h and grid_w == base_w:
            return pos_embed
        pos_embed = pos_embed.reshape(1, base_h, base_w, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        return pos_embed.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1).contiguous()

    def unpatchify(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        batch, tokens, patch_dim = x.shape
        expected_tokens = grid_h * grid_w
        expected_patch_dim = self.patch_size * self.patch_size * self.out_channels
        if tokens != expected_tokens:
            raise ValueError(f"Token count mismatch: got {tokens}, expected {expected_tokens}")
        if patch_dim != expected_patch_dim:
            raise ValueError(f"Patch dim mismatch: got {patch_dim}, expected {expected_patch_dim}")

        x = x.reshape(batch, grid_h, grid_w, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(batch, self.out_channels, grid_h * self.patch_size, grid_w * self.patch_size)
        return x

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | int | list[int],
        y: torch.Tensor | int | list[int] | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}")

        x, pad_h, pad_w = self._pad_to_patch_size(x)
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size

        x = self.x_embedder(x)
        x = x + self._get_pos_embed(grid_h, grid_w, dtype=x.dtype, device=x.device)

        t = self._prepare_timesteps(timesteps, batch, device=x.device)
        cond = self.t_embedder(t)

        if self.y_embedder is not None:
            labels = self._prepare_labels(y, batch, device=x.device)
            cond = cond + self.y_embedder(labels, self.training)

        for block in self.blocks:
            x = block(x, cond)

        x = self.final_layer(x, cond)
        x = self.unpatchify(x, grid_h, grid_w)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :height, :width]
        return x
