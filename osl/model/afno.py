"""
AFNO: Adaptive Fourier Neural Operator

Reference:
 - FourCastNet: https://arxiv.org/abs/2202.11214
 - AFNO: https://arxiv.org/abs/2111.13587

The AFNO performs global mixing in the Fourier domain, which:
 - Naturally handles periodic boundaries (important for ocean grids)
 - Has O(N log N) complexity vs O(N^2) for standard attention
 - Produces smooth outputs without patch artifacts
 - Captures multi-scale patterns efficiently

This implementation adapts AFNO for video prediction of ocean surface currents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


class AFNOConfig(BaseModel):
    """Configuration for AFNO model.

    Args:
        in_channels: Number of input channels (e.g., 3 for sla, ugos, vgos)
        hidden_dim: Hidden dimension for AFNO blocks
        num_blocks: Number of AFNO blocks
        num_layers: Number of encoder/decoder layers
        mlp_ratio: MLP expansion ratio
        drop_rate: Dropout rate
        sparsity_threshold: Soft thresholding for frequency filtering
        hard_thresholding_fraction: Fraction of frequencies to keep
        in_frames: Number of input frames
    """
    in_channels: int = 3
    hidden_dim: int = 256
    num_blocks: int = 8
    num_layers: int = 4
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    sparsity_threshold: float = 0.01
    hard_thresholding_fraction: float = 1.0
    in_frames: int = 16


class AFNO2D(nn.Module):
    """Adaptive Fourier Neural Operator for 2D spatial data.

    Performs token mixing in Fourier space with learnable frequency filters.
    This is the core building block that avoids patch artifacts by operating
    globally in the frequency domain.
    """

    def __init__(self, hidden_dim: int, num_blocks: int = 8,
                 sparsity_threshold: float = 0.01,
                 hard_thresholding_fraction: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction

        # Block-diagonal weight matrices for frequency mixing
        # Each block processes hidden_dim // num_blocks channels
        self.block_size = hidden_dim // num_blocks
        self.scale = 0.02

        # Learnable complex weights for Fourier domain
        # Shape: (num_blocks, block_size, block_size, 2) for real/imag
        self.w1 = nn.Parameter(
            self.scale * torch.randn(num_blocks, self.block_size, self.block_size, 2)
        )
        self.w2 = nn.Parameter(
            self.scale * torch.randn(num_blocks, self.block_size, self.block_size, 2)
        )
        self.b1 = nn.Parameter(
            self.scale * torch.randn(num_blocks, self.block_size, 2)
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(num_blocks, self.block_size, 2)
        )

    def _complex_mul(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex matrix multiplication.

        Args:
            x: (B, blocks, H, W, block_size, 2) complex tensor
            weights: (blocks, block_size, block_size, 2) complex weights
        Returns:
            (B, blocks, H, W, block_size, 2) result
        """
        # Extract real and imaginary parts
        x_real, x_imag = x[..., 0], x[..., 1]
        w_real, w_imag = weights[..., 0], weights[..., 1]

        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        out_real = torch.einsum('bnhwi,nio->bnhwo', x_real, w_real) - \
                   torch.einsum('bnhwi,nio->bnhwo', x_imag, w_imag)
        out_imag = torch.einsum('bnhwi,nio->bnhwo', x_real, w_imag) + \
                   torch.einsum('bnhwi,nio->bnhwo', x_imag, w_real)

        return torch.stack([out_real, out_imag], dim=-1)

    def _complex_add(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Add complex bias."""
        return x + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
        Returns:
            (B, C, H, W) output features
        """
        B, C, H, W = x.shape
        assert C == self.hidden_dim, f"Expected {self.hidden_dim} channels, got {C}"

        # Reshape to blocks: (B, num_blocks, block_size, H, W)
        x = x.view(B, self.num_blocks, self.block_size, H, W)

        # FFT along spatial dimensions
        x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')

        # Reshape for block-wise processing: (B, blocks, H, W//2+1, block_size, 2)
        x_fft = x_fft.permute(0, 1, 3, 4, 2)  # (B, blocks, H, W//2+1, block_size)
        x_fft = torch.stack([x_fft.real, x_fft.imag], dim=-1)

        # Apply hard thresholding (keep fraction of frequencies)
        if self.hard_thresholding_fraction < 1.0:
            H_fft, W_fft = x_fft.shape[2], x_fft.shape[3]
            keep_h = int(H_fft * self.hard_thresholding_fraction)
            keep_w = int(W_fft * self.hard_thresholding_fraction)
            x_fft = x_fft[:, :, :keep_h, :keep_w]

        # Two-layer MLP in Fourier space with block-diagonal weights
        x_fft = self._complex_mul(x_fft, self.w1)
        x_fft = self._complex_add(x_fft, self.b1)

        # ReLU on magnitude (soft thresholding)
        magnitude = torch.sqrt(x_fft[..., 0]**2 + x_fft[..., 1]**2 + 1e-8)
        mask = (magnitude > self.sparsity_threshold).float()
        x_fft = x_fft * mask.unsqueeze(-1)

        x_fft = self._complex_mul(x_fft, self.w2)
        x_fft = self._complex_add(x_fft, self.b2)

        # Pad back if hard thresholding was applied
        if self.hard_thresholding_fraction < 1.0:
            x_fft_padded = torch.zeros(
                B, self.num_blocks, H, W // 2 + 1, self.block_size, 2,
                device=x_fft.device, dtype=x_fft.dtype
            )
            x_fft_padded[:, :, :keep_h, :keep_w] = x_fft
            x_fft = x_fft_padded

        # Convert back to complex tensor
        x_fft = torch.complex(x_fft[..., 0], x_fft[..., 1])
        x_fft = x_fft.permute(0, 1, 4, 2, 3)  # (B, blocks, block_size, H, W//2+1)

        # Inverse FFT
        x = torch.fft.irfft2(x_fft, s=(H, W), dim=(-2, -1), norm='ortho')

        # Reshape back: (B, C, H, W)
        x = x.view(B, C, H, W)

        return x


class AFNOBlock(nn.Module):
    """AFNO Transformer block with Fourier mixing and MLP.

    Combines global Fourier mixing with local MLP processing.
    """

    def __init__(self, hidden_dim: int, num_blocks: int = 8,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 sparsity_threshold: float = 0.01):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.afno = AFNO2D(hidden_dim, num_blocks, sparsity_threshold)

        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input features
        Returns:
            (B, C, H, W) output features
        """
        B, C, H, W = x.shape

        # AFNO branch
        residual = x
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.afno(x)
        x = x + residual

        # MLP branch
        residual = x
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = x + residual

        return x


class PatchEmbed(nn.Module):
    """Patch embedding using convolution (with overlap for smoothness)."""

    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size,
            stride=patch_size // 2,  # Overlap
            padding=patch_size // 4
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class AFNONet(nn.Module):
    """AFNO Network for spatiotemporal prediction.

    A video prediction model using AFNO blocks for global spatial mixing
    and 3D convolutions for temporal modeling.
    """

    def __init__(self, config: AFNOConfig):
        super().__init__()
        self.config = config

        # Temporal fusion: combine frames along channel dimension
        # then use 3D conv to mix temporal information
        self.temporal_embed = nn.Sequential(
            nn.Conv3d(
                config.in_channels,
                config.hidden_dim // 4,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)
            ),
            nn.GroupNorm(8, config.hidden_dim // 4),
            nn.GELU(),
            nn.Conv3d(
                config.hidden_dim // 4,
                config.hidden_dim // 2,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)
            ),
            nn.GroupNorm(8, config.hidden_dim // 2),
            nn.GELU(),
        )

        # Spatial embedding per frame
        self.spatial_embed = nn.Conv2d(
            config.hidden_dim // 2, config.hidden_dim,
            kernel_size=3, padding=1
        )

        # AFNO blocks
        self.blocks = nn.ModuleList([
            AFNOBlock(
                config.hidden_dim,
                config.num_blocks,
                config.mlp_ratio,
                config.drop_rate,
                config.sparsity_threshold
            )
            for _ in range(config.num_layers)
        ])

        # Decoder: upsample back to original channels
        self.decoder = nn.Sequential(
            nn.Conv2d(config.hidden_dim, config.hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, config.hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim // 2, config.in_channels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) input frames
        Returns:
            (B, T, C, H, W) predicted frames
        """
        B, T, C, H, W = x.shape

        # Temporal embedding: (B, T, C, H, W) -> (B, C, T, H, W) -> (B, hidden//2, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.temporal_embed(x)

        # Process each frame through AFNO blocks
        # (B, hidden//2, T, H, W) -> process per frame
        outputs = []
        for t in range(T):
            frame = x[:, :, t]  # (B, hidden//2, H, W)
            frame = self.spatial_embed(frame)  # (B, hidden, H, W)

            # AFNO blocks
            for block in self.blocks:
                frame = block(frame)

            # Decode
            frame = self.decoder(frame)  # (B, C, H, W)
            outputs.append(frame)

        # Stack outputs: (B, T, C, H, W)
        x = torch.stack(outputs, dim=1)

        return x
