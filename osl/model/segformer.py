"""
Reference:
 - https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/segformer/modeling_segformer.py#L675
 - https://arxiv.org/pdf/2105.15203.pdf
 - https://github.com/FrancescoSaverioZuppichini/SegFormer
"""

from dataclasses import dataclass
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Configuration
# ----------------------------
@dataclass
class SegformerConfig:
    num_channels = 3
    num_encoder_blocks = 4
    dephts = [2, 2, 2, 2]
    hidden_sizes = [32, 64, 160, 256]
    patch_sizes = [7, 3, 3, 3]
    strides = [4, 2, 2, 2]
    num_attention_heads = [1, 2, 5, 8]
    mlp_ratios = [4, 4, 4, 4]
    hidden_act = "gelu"
    hidden_dropout_prob = 0.0
    attention_probs_dropout_prob = 0.0
    classifier_dropout_prob = 0.1
    classifier_dropout_prob = 0.1
    initializer_range = 0.02
    drop_path_rate = 0.1
    layer_norm_eps = 1e-6
    decoder_hidden_size = 256
    reshape_last_stage: Optional[bool] = True
    sr_ratios = [8, 4, 2, 1]
    depths = [2, 2, 2, 2]
    num_labels = 1


# ----------------------------
# Basic utilities
# ----------------------------
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


# ----------------------------
# Patch Embeddings
# ----------------------------
class SegformerOverlapPatchEmbeddings(nn.Module):
    def __init__(self, patch_size, stride, in_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        x = self.proj(x)  # (B, hidden_size, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        x = self.layer_norm(x)
        return x, H, W


# ----------------------------
# Efficient Self-Attention
# ----------------------------
class SegformerEfficientSelfAttention(nn.Module):
    def __init__(
        self,
        config: SegformerConfig,
        hidden_size: int,
        num_attention_heads: int,
        sr_ratio: int,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = math.sqrt(self.head_dim)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, H, W):
        # x: (B, N, C)
        B, N, C = x.shape
        q = (
            self.query(x)
            .reshape(B, N, self.num_attention_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_)
            B, C, H_sr, W_sr = x_.shape
            x_ = x_.flatten(2).transpose(1, 2)
            x_ = self.layer_norm(x_)
            k = (
                self.key(x_)
                .reshape(B, -1, self.num_attention_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.value(x_)
                .reshape(B, -1, self.num_attention_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
        else:
            k = (
                self.key(x)
                .reshape(B, N, self.num_attention_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.value(x)
                .reshape(B, N, self.num_attention_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return out


class SegformerSelfOutput(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerAttention(nn.Module):
    def __init__(
        self, config, hidden_size, num_attention_heads, sequence_reduction_ratio
    ):
        super().__init__()
        self.self = SegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sr_ratio=sequence_reduction_ratio,
        )
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)
        self.pruned_heads = set()

    def forward(self, hidden_states, height, width):
        x = self.self(hidden_states, height, width)
        x = self.output(x, hidden_states)
        return x


class SegformerDWConv(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, height, width):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, height, width)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ----------------------------
# Mix-FFN (MLP)
# ----------------------------
class SegformerMixFFN(nn.Module):
    def __init__(self, config: SegformerConfig, in_features: int, mlp_ratio: float):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        self.act = nn.GELU() if config.hidden_act == "gelu" else nn.ReLU()
        self.dense2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, H, W):
        x = self.dense1(x)  # (B, N, hidden_features)
        x = self.dwconv(x, H, W)  # Apply depthwise convolution
        x = self.act(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


# ----------------------------
# Transformer Block
# ----------------------------
class SegformerLayer(nn.Module):
    def __init__(
        self,
        config: SegformerConfig,
        hidden_size: int,
        num_attention_heads: int,
        sr_ratio: int,
        mlp_ratio: float,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = SegformerAttention(
            config, hidden_size, num_attention_heads, sr_ratio
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.mlp = SegformerMixFFN(config, hidden_size, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, H, W):
        shortcut = x
        x = self.layer_norm_1(x)
        x = self.attention(x, H, W)
        x = self.drop_path(x) + shortcut

        shortcut = x
        x = self.layer_norm_2(x)
        x = self.mlp(x, H, W)  # Pass H and W to mlp
        x = self.drop_path(x) + shortcut
        return x


# ----------------------------
# Encoder: Stacked Stages
# ----------------------------
class SegformerEncoder(nn.Module):
    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.reshape_last_stage = config.reshape_last_stage
        drop_path_decays = [
            x.item()
            for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))
        ]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    config.patch_sizes[i],
                    config.strides[i],
                    config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    config.hidden_sizes[i],
                )
            )

        self.patch_embeddings = nn.ModuleList(embeddings)
        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    SegformerLayer(
                        config,
                        config.hidden_sizes[i],
                        config.num_attention_heads[i],
                        config.sr_ratios[i],
                        config.mlp_ratios[i],
                        drop_path=drop_path_decays[cur + j],
                    )
                )
            blocks.append(nn.ModuleList(layers))
        self.block = nn.ModuleList(blocks)
        self.layer_norm = nn.ModuleList(
            [nn.LayerNorm(hs) for hs in config.hidden_sizes]
        )

    def forward(self, x):
        B = x.size(0)
        hidden_states = []
        for i, (embed, blocks, norm) in enumerate(
            zip(self.patch_embeddings, self.block, self.layer_norm)
        ):
            x, H, W = embed(x)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x)
            if i != len(self.block) - 1 or (
                i == len(self.block) - 1 and self.reshape_last_stage
            ):
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            hidden_states.append(x)
        return hidden_states


# ----------------------------
# Minimal Decode Head for Segmentation
# ----------------------------


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHead(nn.Module):
    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.linear_c = nn.ModuleList(
            [SegformerMLP(config, input_dim=dim) for dim in config.hidden_sizes]
        )
        # the following 3 layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(
            config.decoder_hidden_size, config.num_labels, kernel_size=1
        )

        self.config = config

    def forward(self, features):
        batch_size = features[-1].shape[0]

        hidden_states = []
        for feat, mlp in zip(features, self.linear_c):
            if self.config.reshape_last_stage is False and feat.ndim == 3:
                h = w = int(math.sqrt(feat.shape[-1]))
                feat = (
                    feat.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2).contiguous()
                )
            h, w = feat.shape[2], feat.shape[3]
            feat = mlp(feat)
            feat = feat.permute(0, 2, 1).reshape(batch_size, -1, h, w)
            feat = F.interpolate(
                feat, size=features[0].size()[2:], mode="bilinear", align_corners=False
            )
            hidden_states.append(feat)

        x = self.linear_fuse(torch.cat(hidden_states[::-1], dim=1))
        x = x.contiguous()
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.classifier(x)


class Segformer(nn.Module):
    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.config = config
        self.encoder = SegformerEncoder(config)
        self.apply(self._init_weights)  # Apply weight initialization

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)


class SegformerForSemanticSegmentation(nn.Module):
    def __init__(self, config: SegformerConfig):
        super().__init__()
        self.config = config
        self.segformer = Segformer(config)
        self.decode_head = SegformerDecodeHead(config)
        self.apply(self._init_weights)  # Apply weight initialization

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        encoder_outs = self.segformer(pixel_values)
        logits = self.decode_head(encoder_outs)
        return logits
