"""
SimVP: Simpler yet Better Video Prediction

Reference:
 - https://arxiv.org/abs/2206.05099 (SimVP: Simpler yet Better Video Prediction)
 - https://arxiv.org/abs/2206.12126 (SimVP v2: TAU - Temporal Attention Unit)
 - https://github.com/chengtan9907/OpenSTL

A fully convolutional architecture for video prediction that avoids patch artifacts
by using spatial convolutions instead of patch embeddings. The architecture consists of:
 - Spatial Encoder: Extracts features from each frame independently
 - Temporal Module: Models temporal dynamics across frames
 - Spatial Decoder: Reconstructs output frames from features

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


class SimVPConfig(BaseModel):
    """Configuration for SimVP model.

    Args:
        in_channels: Number of input channels (e.g., 3 for sla, ugos, vgos)
        hidden_dim: Hidden dimension in the encoder/decoder
        num_layers: Number of layers in encoder and decoder
        kernel_size: Convolution kernel size
        temporal_module: Type of temporal module ('inception', 'conv', 'tau')
        drop_path: Drop path rate for regularization
        in_frames: Number of input frames (required for 'inception' and 'tau' modules)
    """
    in_channels: int = 3
    hidden_dim: int = 64
    num_layers: int = 4
    kernel_size: int = 5
    temporal_module: str = "inception"  # 'inception', 'conv', 'tau'
    drop_path: float = 0.0
    in_frames: int = 16


# ----------------------------
# Basic Building Blocks
# ----------------------------

class ConvBlock(nn.Module):
    """Basic convolutional block with GroupNorm and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvSC(nn.Module):
    """Convolutional block with optional stride for downsampling/upsampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 downsample: bool = False, upsample: bool = False):
        super().__init__()
        padding = kernel_size // 2
        stride = 2 if downsample else 1

        if upsample:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size,
                stride=2, padding=padding, output_padding=1
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding
            )
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ----------------------------
# Spatial Encoder / Decoder
# ----------------------------

class SpatialEncoder(nn.Module):
    """Encodes each frame independently using 2D convolutions.

    Progressively downsamples spatial dimensions while increasing channels.
    """

    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int, kernel_size: int):
        super().__init__()
        layers = [ConvSC(in_channels, hidden_dim, kernel_size)]
        for _ in range(1, num_layers):
            layers.append(ConvSC(hidden_dim, hidden_dim, kernel_size, downsample=True))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) input frames
        Returns:
            (B, T, hidden_dim, H', W') encoded features
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        for layer in self.layers:
            x = layer(x)

        _, C_out, H_out, W_out = x.shape
        return x.view(B, T, C_out, H_out, W_out)


class SpatialDecoder(nn.Module):
    """Decodes features back to frame space using 2D convolutions.

    Progressively upsamples spatial dimensions back to original resolution.
    """

    def __init__(self, hidden_dim: int, out_channels: int, num_layers: int, kernel_size: int):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(ConvSC(hidden_dim, hidden_dim, kernel_size, upsample=True))
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, hidden_dim, H', W') encoded features
        Returns:
            (B, T, C, H, W) decoded frames
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        for layer in self.layers:
            x = layer(x)

        _, C_out, H_out, W_out = x.shape
        return x.reshape(B, T, C_out, H_out, W_out)


# ----------------------------
# Temporal Modules
# ----------------------------

class InceptionModule(nn.Module):
    """Inception-style module with multi-scale convolutions.

    Captures temporal patterns at different scales using parallel branches
    with different kernel sizes.
    """

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        # Branch with 1x1 conv
        self.branch1 = nn.Conv2d(channels, channels // 4, kernel_size=1)

        # Branch with 3x3 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.Conv2d(channels // 4, channels // 4, kernel_size=3, padding=1)
        )

        # Branch with 5x5 conv (or kernel_size)
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.Conv2d(channels // 4, channels // 4, kernel_size=kernel_size, padding=kernel_size // 2)
        )

        # Branch with max pooling
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, channels // 4, kernel_size=1)
        )

        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        out = torch.cat([b1, b3, b5, bp], dim=1)
        return self.act(self.norm(out))


class TemporalInceptionModule(nn.Module):
    """Temporal module using stacked inception blocks.

    Processes features across time using inception modules that capture
    multi-scale spatial-temporal patterns.
    """

    def __init__(self, num_frames: int, hidden_dim: int, num_blocks: int = 4, kernel_size: int = 5):
        super().__init__()
        # Merge time into channels: (B, T, C, H, W) -> (B, T*C, H, W)
        self.num_frames = num_frames
        merged_channels = num_frames * hidden_dim

        blocks = []
        for _ in range(num_blocks):
            blocks.append(InceptionModule(merged_channels, kernel_size))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        # Merge time into channels
        x = x.view(B, T * C, H, W)
        x = self.blocks(x)
        # Split back
        return x.view(B, T, C, H, W)


class TemporalAttentionUnit(nn.Module):
    """TAU: Temporal Attention Unit from SimVP v2.

    Uses attention mechanism to model temporal dependencies with separate
    intra-frame and inter-frame attention paths.
    """

    def __init__(self, num_frames: int, hidden_dim: int, kernel_size: int = 5):
        super().__init__()
        self.num_frames = num_frames
        merged_channels = num_frames * hidden_dim

        # Temporal attention (across frames)
        self.temporal_fc = nn.Sequential(
            nn.Linear(num_frames, num_frames),
            nn.GELU(),
            nn.Linear(num_frames, num_frames),
            nn.Sigmoid()
        )

        # Spatial processing
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(merged_channels, merged_channels, kernel_size=kernel_size,
                     padding=kernel_size // 2, groups=num_frames),
            nn.GroupNorm(8, merged_channels),
            nn.GELU()
        )

        # Output projection
        self.proj = nn.Conv2d(merged_channels, merged_channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, merged_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        residual = x

        # Temporal attention
        x_t = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, C, H, W, T)
        x_t = self.temporal_fc(x_t)  # Attention over time dimension
        x_t = x_t.permute(0, 4, 1, 2, 3).contiguous()  # (B, T, C, H, W)

        # Spatial processing
        x_s = x_t.reshape(B, T * C, H, W)
        x_s = self.spatial_conv(x_s)
        x_s = self.proj(x_s)
        x_s = self.norm(x_s)
        x_s = x_s.reshape(B, T, C, H, W)

        return x_s + residual


class TemporalConvModule(nn.Module):
    """Simple temporal convolution module.

    Uses 3D convolutions to model temporal dynamics directly.
    """

    def __init__(self, num_frames: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, kernel_size, kernel_size),
                               padding=(1, kernel_size // 2, kernel_size // 2))
        self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, kernel_size, kernel_size),
                               padding=(1, kernel_size // 2, kernel_size // 2))
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        residual = x

        # (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        # (B, C, T, H, W) -> (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return x + residual


# ----------------------------
# Main SimVP Model
# ----------------------------

class SimVP(nn.Module):
    """SimVP: Simpler yet Better Video Prediction.

    A fully convolutional architecture for spatiotemporal prediction that
    avoids patch artifacts by using spatial convolutions throughout.

    Architecture:
        1. Spatial Encoder: Extracts features from each frame
        2. Temporal Module: Models dynamics across frames
        3. Spatial Decoder: Reconstructs output frames

    Args:
        config: SimVPConfig with model hyperparameters
    """

    def __init__(self, config: SimVPConfig):
        super().__init__()
        self.config = config
        self.in_frames = config.in_frames
        self.out_frames = config.in_frames

        # Spatial encoder
        self.encoder = SpatialEncoder(
            config.in_channels, config.hidden_dim,
            config.num_layers, config.kernel_size
        )

        # Temporal module (processes all frames together)
        if config.temporal_module == "inception":
            self.temporal = TemporalInceptionModule(
                config.in_frames, config.hidden_dim,
                num_blocks=4, kernel_size=config.kernel_size
            )
        elif config.temporal_module == "tau":
            self.temporal = TemporalAttentionUnit(
                config.in_frames, config.hidden_dim, config.kernel_size
            )
        elif config.temporal_module == "conv":
            self.temporal = TemporalConvModule(
                config.in_frames, config.hidden_dim, config.kernel_size
            )
        else:
            raise ValueError(f"Unknown temporal module: {config.temporal_module}")

        # Spatial decoder
        self.decoder = SpatialDecoder(
            config.hidden_dim, config.in_channels,
            config.num_layers, config.kernel_size
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) input frames
        Returns:
            (B, T, C, H, W) predicted frames (shifted by 1 timestep)
        """
        B, T, C, H, W = x.shape

        # Encode each frame
        encoded = self.encoder(x)  # (B, T, hidden_dim, H', W')

        # Model temporal dynamics
        temporal = self.temporal(encoded)  # (B, T, hidden_dim, H', W')

        # Decode to output frames
        decoded = self.decoder(temporal)  # (B, T, C, H'', W'')

        # Interpolate to match original size if needed
        if decoded.shape[-2:] != (H, W):
            decoded = decoded.view(B * T, C, decoded.shape[-2], decoded.shape[-1])
            decoded = F.interpolate(decoded, size=(H, W), mode='bilinear', align_corners=False)
            decoded = decoded.view(B, T, C, H, W)

        return decoded
