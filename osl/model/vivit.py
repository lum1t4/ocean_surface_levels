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
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = nn.functional.dropout(attn_weight, p=dropout_p, training=training)
    return attn_weight @ value


scaled_dot_product = eager_attention_forward

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(enabled=torch.backends.cuda.is_flash_attention_available())
    scaled_dot_product = nn.functional.scaled_dot_product_attention
    


class VivitConfig(BaseModel):
    num_frames: int = 32
    image_size: int = 224
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-6
    num_attention_heads: int = 12
    num_channels: int = 3
    num_hidden_layers: int = 12
    qkv_bias: bool = True
    tubelet_size: list[int] = [2, 16, 16]
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    hidden_size: int = 768
    hidden_act: str = "gelu_fast"
    
    

class VivitTubeletEmbeddings(nn.Module):
    """
    Construct Vivit Tubelet embeddings.

    This module turns a batch of videos of shape (batch_size, num_frames, num_channels, height, width) into a tensor of
    shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size[0]) * (height // tubelet_size[1]) *
    (width // tubelet_size[2]).
    """

    def __init__(self, config: VivitConfig):
        super().__init__()
        self.num_frames = config.num_frames
        self.image_size = config.image_size
        self.patch_size = config.tubelet_size
        self.embed_dim = config.hidden_size

        self.num_patches = (
            (self.image_size // self.patch_size[2])
            * (self.image_size // self.patch_size[1])
            * (self.num_frames // self.patch_size[0])
        )
        self.projection = nn.Conv3d(config.num_channels, config.hidden_size, kernel_size=config.tubelet_size, stride=config.tubelet_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
            raise ValueError(
                f"Image image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        # (B, T, C, H, W) -> (B, C, T, H, W)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        # (B, C, T, H, W) -> (B, C', T', H', W')
        # ex: [1, 3, 4, 224, 224] -> [1, 768, 2, 14, 14]
        x = self.projection(pixel_values)
        # (B, C', T', H', W') -> (B, C', T'*H'*W') -> (B, T'*H'*W', C')
        # ex. [1, 768, 2, 14, 14] -> [1, 768, 392] -> transpose(1, 2) -> [1, 392, 768]
        x = x.flatten(2).transpose(1, 2)
        return x


class VivitEmbeddings(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.tubelet_size[1:]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = VivitTubeletEmbeddings(config)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.patch_embeddings.num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        B, _, _, H, W = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        #cls_tokens = self.cls_token.tile([B, 1, 1])
        #embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        return embeddings
        position_embeddings = self.interpolate_pos_encoding(embeddings, H, W) if interpolate_pos_encoding else self.position_embeddings
        print(embeddings.shape, position_embeddings.shape)
        embeddings = embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings 

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int):
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings
        

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]
        new_height = height // self.patch_size[0]
        new_width = width // self.patch_size[1]

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
    


class VivitSelfAttention(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.query  = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.key    = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.value  = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x)  .view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = scaled_dot_product(q, k, v, dropout_p=0.0,is_causal=True)
        return att.transpose(1, 2).contiguous().reshape(B, T, -1)



class VivitSelfOutput(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x: torch.Tensor):
        x = self.dense(x)
        x = self.dropout(x)
        return x
    

class VivitAttention(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.attention = VivitSelfAttention(config)
        self.output = VivitSelfOutput(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self_attn_output = self.attention(hidden_states)
        output = self.output(self_attn_output)
        return output


class VivitIntermediate(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class VivitOutput(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states
    

class VivitLayer(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.attention = VivitAttention(config)
        self.intermediate = VivitIntermediate(config)
        self.output = VivitOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm = self.layernorm_before(hidden_states)
        hidden_states = self.attention(norm) + hidden_states
        output = self.layernorm_after(hidden_states)
        output = self.intermediate(output)
        output = self.output(output, hidden_states)
        return output


class VivitVideoRegressionHead(nn.Module):
    def __init__(self, config, out_channels: int = 3):
        super().__init__()
        self.image_size = config.image_size
        self.tubelet_size = config.tubelet_size
        self.hidden_size = config.hidden_size

        self.feat_h = self.image_size // self.tubelet_size[1]
        self.feat_w = self.image_size // self.tubelet_size[2]
        self.projection = nn.ConvTranspose3d(
            in_channels=config.hidden_size,
            out_channels=out_channels,
            kernel_size=config.tubelet_size,
            stride=config.tubelet_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # (B, T'*H'*W', C') -> (B, C', T'*H'*W') -> (B, C', T', H', W')
        x = x.transpose(1, 2).view(B, C, -1, self.feat_h, self.feat_w)
        # (B, C', T', H', W') -> (B, C, T, H, W)
        x = self.projection(x)
        return x




class VivitDecoder(nn.Module):
    def __init__(self, config: VivitConfig):
        super().__init__()
        self.embeddings = VivitEmbeddings(config)
        self.layers = nn.ModuleList([VivitLayer(config) for _ in range(config.num_hidden_layers)])
        self.head = VivitVideoRegressionHead(config)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.embeddings(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.head(hidden_states)


