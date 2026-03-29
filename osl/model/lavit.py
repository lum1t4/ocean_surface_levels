from pydantic import BaseModel
from transformers.activations import ACT2FN
import torch
from torch import Tensor, nn


def eager_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    training: bool = True,
):
    L, S = query.size(-2), key.size(-2)
    scale_factor = query.size(-1) ** -0.5 if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        attn_mask = attn_mask.to(query.device)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = nn.functional.dropout(attn_weight, p=dropout_p, training=training)
    return attn_weight @ value


scaled_dot_product = eager_attention_forward

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(
        enabled=torch.backends.cuda.is_flash_attention_available()
    )
    scaled_dot_product = nn.functional.scaled_dot_product_attention


class LaViTConfig(BaseModel):
    num_channels: int = 32
    image_size: int = 28
    patch_size: list[int] = [4, 4]

    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    hidden_act: str = "gelu_fast"
    hidden_dropout_prob: float = 0.0
    max_temporal_positions: int = 64


class PatchEmbeddings(nn.Module):
    def __init__(self, config: LaViTConfig):
        super().__init__()
        self.patch_h = config.patch_size[0]
        self.patch_w = config.patch_size[1]
        self.proj = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        B, T, C, H, W = x.shape
        if H % self.patch_h != 0 or W % self.patch_w != 0:
            raise ValueError(
                f"Input spatial shape {(H, W)} must be divisible by patch_size {(self.patch_h, self.patch_w)}"
            )
        x = x.view(B * T, C, H, W)
        x = self.proj(x)
        Hp, Wp = x.shape[-2], x.shape[-1]
        D = x.shape[1]
        x = x.flatten(2).transpose(1, 2).contiguous()
        N = x.shape[1]
        x = x.view(B, T, N, D)
        return x, Hp, Wp


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), *args, **kwargs)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: LaViTConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = scaled_dot_product(q, k, v, dropout_p=0.0, attn_mask=mask)
        return attn.transpose(1, 2).contiguous().reshape(B, T, -1)


class MLP(nn.Module):
    def __init__(self, config: LaViTConfig):
        super().__init__()
        intermediate_size = int(config.hidden_size * config.mlp_ratio)
        self.dense_mid = nn.Linear(config.hidden_size, intermediate_size)
        self.dense_out = nn.Linear(intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.act(self.dense_mid(x)))
        x = self.dropout(self.dense_out(x))
        return x


class TemporalBlock(nn.Module):
    def __init__(self, config: LaViTConfig):
        super().__init__()
        self.attn = PreNorm(config.hidden_size, MultiHeadSelfAttention(config))
        self.mlp = PreNorm(config.hidden_size, MLP(config))

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, T, N, D = x.shape
        y = x.permute(0, 2, 1, 3).contiguous()
        y = y.view(B * N, T, D)
        y = y + self.attn(y, mask=mask)
        y = y + self.mlp(y)
        y = y.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return y


class SpatialBlock(nn.Module):
    def __init__(self, config: LaViTConfig):
        super().__init__()
        self.attn = PreNorm(config.hidden_size, MultiHeadSelfAttention(config))
        self.mlp = PreNorm(config.hidden_size, MLP(config))

    def forward(self, x: torch.Tensor):
        B, T, N, D = x.shape
        y = x.reshape(B * T, N, D)
        y = y + self.attn(y)
        y = y + self.mlp(y)
        return y.reshape(B, T, N, D)


class LaViT(nn.Module):
    def __init__(self, config: LaViTConfig):
        super().__init__()
        self.config = config
        self.patch_h = config.patch_size[0]
        self.patch_w = config.patch_size[1]
        self.num_patches = (config.image_size // self.patch_h) * (
            config.image_size // self.patch_w
        )

        self.pos_spatial = nn.Parameter(torch.zeros(1, 1, self.num_patches, config.hidden_size))
        self.pos_temporal = nn.Parameter(torch.zeros(1, config.max_temporal_positions, 1, config.hidden_size))
        nn.init.trunc_normal_(self.pos_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_temporal, std=0.02)

        self.patch_emb = PatchEmbeddings(config)
        self.s_blocks = nn.ModuleList(
            [SpatialBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.t_blocks = nn.ModuleList(
            [TemporalBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)

        self.decode = nn.Sequential(
            nn.Conv2d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                in_channels=config.hidden_size,
                out_channels=config.num_channels,
                kernel_size=(self.patch_h, self.patch_w),
                stride=(self.patch_h, self.patch_w),
            ),
        )

    def forward(self, tokens: torch.Tensor):
        B, T, C, H, W = tokens.shape
        if T > self.pos_temporal.shape[1]:
            raise ValueError(
                f"Sequence length {T} exceeds max_temporal_positions={self.pos_temporal.shape[1]}"
            )

        mask = self.make_mask(T).to(tokens.device)
        tokens, Hp, Wp = self.patch_emb(tokens)

        if tokens.shape[2] != self.pos_spatial.shape[2]:
            raise ValueError(
                f"Patch count mismatch: tokens has {tokens.shape[2]} patches, expected {self.pos_spatial.shape[2]}. "
                "Check image_size and patch_size."
            )

        tokens = tokens + self.pos_spatial + self.pos_temporal[:, :T]

        for sblk, tblk in zip(self.s_blocks, self.t_blocks):
            tokens = sblk(tokens)
            tokens = tblk(tokens, mask)

        tokens = self.norm(tokens)
        _, _, N, D = tokens.shape
        grid = tokens.view(B * T, N, D).transpose(1, 2).contiguous().view(B * T, D, Hp, Wp)
        out = self.decode(grid)

        if out.shape[-2] != H or out.shape[-1] != W:
            raise RuntimeError(
                f"Decoded spatial shape {tuple(out.shape[-2:])} does not match input shape {(H, W)}. "
                "Check image_size and patch_size divisibility."
            )

        return out.view(B, T, self.config.num_channels, H, W)

    def make_mask(self, size: int):
        return torch.ones(size, size, dtype=torch.bool).tril(diagonal=0)


if __name__ == "__main__":
    x = torch.rand((2, 8, 16, 28, 28))
    model = LaViT(LaViTConfig())
    y = model(x)
    print(y.shape)
