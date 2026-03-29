"""
Adaptive Fourier Neural Operator (AFNO) for spatiotemporal prediction.

Inspired by FourCastNet (Pathak et al., 2022). The core idea is to replace
self-attention with frequency-domain token mixing via 2D FFT, which is both
computationally efficient (O(N log N) vs O(N^2)) and naturally suited to
geophysical fields on regular grids.

Architecture:
    1. Patch embedding (Conv2d) to convert (C, H, W) -> (N, D) tokens
    2. Stack of AFNO blocks:
       - LayerNorm -> 2D FFT -> learnable complex frequency filter -> 2D IFFT
       - Soft thresholding for sparsity in frequency domain
       - LayerNorm -> MLP
       - FiLM conditioning from sinusoidal time embedding (optional)
    3. Unpatchify back to (C_out, H, W)

Reference:
    Pathak, J., et al. "FourCastNet: A Global Data-driven High-Resolution
    Weather Model using Adaptive Fourier Neural Operators." arXiv:2202.11214, 2022.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


# --------------------- /
# Config
# --------------------- /
class AFNOConfig(BaseModel):
    image_size: int = 224
    patch_size: int = 8          # 224/8 = 28 tokens per axis
    in_channels: int = 1
    out_channels: int = 1
    hidden_size: int = 256
    num_layers: int = 8
    mlp_ratio: float = 4.0
    num_blocks: int = 8          # number of frequency blocks for channel mixing
    sparsity_threshold: float = 0.01  # soft-thresholding in freq domain
    with_time_emb: bool = True


# --------------------- /
# Time embeddings
# --------------------- /
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10_000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeMLPEmbedding(nn.Module):
    """Time embedding with MLP projection."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(dim_in)
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


# --------------------- /
# AFNO Block
# --------------------- /
class AFNOBlock(nn.Module):
    """
    Single AFNO layer:
    1. LayerNorm
    2. 2D FFT on spatial dims
    3. Learnable complex-valued weight multiplication (channel mixing in freq domain)
    4. Soft thresholding (sparsity)
    5. 2D IFFT
    6. Residual connection
    7. LayerNorm + MLP with optional FiLM conditioning
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        mlp_ratio: float,
        sparsity_threshold: float,
        time_emb_dim: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.sparsity_threshold = sparsity_threshold

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        # Frequency domain weights — complex valued
        # We process hidden_size in `num_blocks` groups
        block_size = hidden_size // num_blocks
        # Weight shape: (2, num_blocks, block_size, block_size) for real+imag
        self.weight1 = nn.Parameter(torch.randn(2, num_blocks, block_size, block_size) * 0.02)
        self.bias1 = nn.Parameter(torch.zeros(2, num_blocks, block_size))
        self.weight2 = nn.Parameter(torch.randn(2, num_blocks, block_size, block_size) * 0.02)
        self.bias2 = nn.Parameter(torch.zeros(2, num_blocks, block_size))

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # Optional FiLM conditioning from time embedding
        self.film = None
        if time_emb_dim is not None:
            self.film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, hidden_size * 4),  # gamma1, beta1, gamma2, beta2
            )

    def _freq_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learnable frequency-domain mixing.
        x: (B, H, W, C) in spatial domain
        """
        B, H, W, C = x.shape

        # 2D FFT on spatial dimensions
        x_ft = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  # (B, H, W//2+1, C)

        # Reshape for block-wise processing
        x_ft = x_ft.reshape(B, x_ft.shape[1], x_ft.shape[2], self.num_blocks, -1)

        # Complex multiplication: (real + i*imag) * (w_real + i*w_imag)
        x_real = x_ft.real
        x_imag = x_ft.imag

        # First complex linear layer
        o_real = torch.einsum('bhwnc,ncd->bhwnd', x_real, self.weight1[0]) - \
                 torch.einsum('bhwnc,ncd->bhwnd', x_imag, self.weight1[1]) + self.bias1[0]
        o_imag = torch.einsum('bhwnc,ncd->bhwnd', x_real, self.weight1[1]) + \
                 torch.einsum('bhwnc,ncd->bhwnd', x_imag, self.weight1[0]) + self.bias1[1]

        # ReLU on real and imaginary separately
        o_real = F.relu(o_real)
        o_imag = F.relu(o_imag)

        # Second complex linear layer
        o_real2 = torch.einsum('bhwnc,ncd->bhwnd', o_real, self.weight2[0]) - \
                  torch.einsum('bhwnc,ncd->bhwnd', o_imag, self.weight2[1]) + self.bias2[0]
        o_imag2 = torch.einsum('bhwnc,ncd->bhwnd', o_real, self.weight2[1]) + \
                  torch.einsum('bhwnc,ncd->bhwnd', o_imag, self.weight2[0]) + self.bias2[1]

        # Soft thresholding (sparsity in frequency domain)
        o_real2 = torch.sign(o_real2) * F.relu(o_real2.abs() - self.sparsity_threshold)
        o_imag2 = torch.sign(o_imag2) * F.relu(o_imag2.abs() - self.sparsity_threshold)

        # Reconstruct complex tensor
        x_ft_out = torch.complex(o_real2, o_imag2)
        x_ft_out = x_ft_out.reshape(B, x_ft_out.shape[1], x_ft_out.shape[2], C)

        # Inverse FFT
        x_out = torch.fft.irfft2(x_ft_out, s=(H, W), dim=(1, 2), norm='ortho')
        return x_out

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, N, C) where N = H*W (flattened spatial tokens)
        t_emb: (B, time_emb_dim) or None
        """
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # assume square grid of patches

        # Get FiLM parameters if available
        gamma1, beta1, gamma2, beta2 = None, None, None, None
        if self.film is not None and t_emb is not None:
            film_params = self.film(t_emb)  # (B, C*4)
            gamma1, beta1, gamma2, beta2 = film_params.chunk(4, dim=-1)

        # Frequency mixing branch
        residual = x
        x_norm = self.norm1(x)
        if gamma1 is not None:
            x_norm = x_norm * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)

        x_spatial = x_norm.reshape(B, H, W, C)
        x_freq = self._freq_mixing(x_spatial)
        x_freq = x_freq.reshape(B, N, C)
        x = residual + x_freq

        # MLP branch
        residual = x
        x_norm = self.norm2(x)
        if gamma2 is not None:
            x_norm = x_norm * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        x = residual + self.mlp(x_norm)

        return x


# --------------------- /
# AFNO Model
# --------------------- /
class AFNO(nn.Module):
    """
    Adaptive Fourier Neural Operator for spatiotemporal prediction.

    Inspired by FourCastNet (Pathak et al., 2022).
    Uses FFT-based token mixing instead of self-attention.
    """

    def __init__(self, config: AFNOConfig):
        super().__init__()
        self.config = config
        num_patches = (config.image_size // config.patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.in_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Time embedding
        time_emb_dim = None
        self.time_mlp = None
        if config.with_time_emb:
            time_emb_dim = config.hidden_size * 4
            self.time_mlp = TimeMLPEmbedding(config.hidden_size, time_emb_dim)

        # AFNO blocks
        self.blocks = nn.ModuleList([
            AFNOBlock(
                hidden_size=config.hidden_size,
                num_blocks=config.num_blocks,
                mlp_ratio=config.mlp_ratio,
                sparsity_threshold=config.sparsity_threshold,
                time_emb_dim=time_emb_dim,
            )
            for _ in range(config.num_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(config.hidden_size)
        patch_dim = config.patch_size * config.patch_size * config.out_channels
        self.head = nn.Linear(config.hidden_size, patch_dim)

    def forward(self, x: torch.Tensor, time: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
            time: (B,) long tensor for time conditioning (optional)

        Returns:
            (B, C_out, H, W) prediction
        """
        B, C, H, W = x.shape
        pH = H // self.config.patch_size
        pW = W // self.config.patch_size

        # Patch embed: (B, C, H, W) -> (B, D, pH, pW) -> (B, N, D)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        # Time embedding
        t_emb = None
        if self.time_mlp is not None and time is not None:
            t_emb = self.time_mlp(time.float())

        # AFNO blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Output: (B, N, D) -> (B, N, p*p*C_out) -> (B, C_out, H, W)
        x = self.norm(x)
        x = self.head(x)
        x = x.reshape(B, pH, pW, self.config.patch_size, self.config.patch_size, self.config.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, self.config.out_channels, H, W)

        return x


if __name__ == "__main__":
    from osl.core.pytorch import model_get_num_params

    # Test with default config (224x224, patch_size=8, 1ch in/out)
    config = AFNOConfig()
    model = AFNO(config)
    num_params = model_get_num_params(model)
    print(f"AFNO default config: {num_params:,} params")

    # Forward pass with time conditioning
    img = torch.randn(2, 1, 224, 224)
    t = torch.randint(0, 100, (2,))
    out = model(img, t)
    print(f"Input:  {img.shape}")
    print(f"Time:   {t.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 1, 224, 224), f"Shape mismatch: {out.shape}"

    # Forward pass without time conditioning
    out_no_time = model(img)
    print(f"Output (no time): {out_no_time.shape}")
    assert out_no_time.shape == (2, 1, 224, 224)

    # Test with time embedding disabled
    config_no_time = AFNOConfig(with_time_emb=False)
    model_no_time = AFNO(config_no_time)
    out2 = model_no_time(img)
    print(f"Output (no time emb): {out2.shape}")
    assert out2.shape == (2, 1, 224, 224)

    print("All tests passed.")
