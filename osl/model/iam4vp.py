############################################################
#  IAM4VP‑ME - A model for video prediction with metadata
############################################################
import math

import torch
import torch.nn as nn
from einops import rearrange


# ---------------------------------------------------------------------------
#  Utility layers
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, n: int, eps: float = 1e-6, data_format: str = "channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.eps = eps
        assert data_format in {"channels_first", "channels_last"}
        self.data_format = data_format
        self.shape = (n,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return torch.nn.functional.layer_norm(x, self.shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class BasicConv2d(nn.Module):
    def __init__(
        self, cin: int, cout: int, *, stride: int, transpose: bool = False, act_norm: bool = True
    ):
        super().__init__()
        if stride == 1:
            transpose = False
        if transpose:
            self.conv = nn.Sequential(
                nn.Conv2d(cin, cout * 4, 3, 1, 1, bias=False),
                nn.PixelShuffle(2),
            )
        else:
            self.conv = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.norm = LayerNorm(cout)
        self.act = nn.SiLU(True)
        self.act_norm = act_norm

    def forward(self, x):
        x = self.conv(x)
        if self.act_norm:
            x = self.act(self.norm(x))
        return x


class ConvSC(nn.Module):
    def __init__(self, cin: int, cout: int, *, stride: int, transpose: bool = False):
        super().__init__()
        self.core = BasicConv2d(cin, cout, stride=stride, transpose=transpose)

    def forward(self, x):
        return self.core(x)


# ---------------------------------------------------------------------------
#  Positional & metadata embedding
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        self.register_buffer("freq", freq, persistent=False)

    def forward(self, x):  # (B,)
        x = x[:, None] * self.freq[None, :]
        return torch.cat([x.sin(), x.cos()], dim=1)


class MetadataEmbedding(nn.Module):
    """Encode (t, lon, lat) scalars → 64D vector."""

    def __init__(self, meta: int, dim: int = 64):
        """
        Args:
            meta: Number of metadata features (e.g., 3 for (t, lon, lat)).
            dim: Output dimension of the embedding.
        """
        super().__init__()
        self.pe = SinusoidalPosEmb(dim)
        self.net = nn.Sequential(nn.Linear(meta * dim, 256), nn.GELU(), nn.Linear(256, dim))

    def forward(self, meta):
        parts = [self.pe(meta[:, i]) for i in range(meta.shape[1])]
        return self.net(torch.cat(parts, dim=1))


# ---------------------------------------------------------------------------
#  ConvNeXt blocks
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1 - self.p
        mask = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) < keep
        return x.div(keep) * mask


class LKA(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv0 = nn.Conv2d(d, d, 5, 1, 2, groups=d)
        self.conv1 = nn.Conv2d(d, d, 7, 1, 9, dilation=3, groups=d)
        self.conv2 = nn.Conv2d(d, d, 1)

    def forward(self, x):
        w = self.conv2(self.conv1(self.conv0(x)))
        return x * w


class ConvNeXtBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dw = LKA(d)
        self.norm = LayerNorm(d, data_format="channels_last")
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.gamma = nn.Parameter(1e-6 * torch.ones(d))
        self.dp = DropPath(0.0)
        self.meta = nn.Sequential(nn.GELU(), nn.Linear(64, d))

    def forward(self, x, m):
        h = self.dw(x) + rearrange(self.meta(m), "b c -> b c 1 1")
        h = h.permute(0, 2, 3, 1)
        h = self.ffn(self.norm(h))
        h = h.permute(0, 3, 1, 2)
        return x + self.dp(self.gamma[None, :, None, None] * h)


class ConvNeXtBottle(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.meta = nn.Sequential(nn.GELU(), nn.Linear(64, dim))
        self.dw = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = LayerNorm(dim, data_format="channels_last")
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x, m):
        h = self.dw(x) + rearrange(self.meta(m), "b c -> b c 1 1")
        h = h.permute(0, 2, 3, 1)
        h = self.ffn(self.norm(h))
        h = h.permute(0, 3, 1, 2)
        return x + self.gamma[None, :, None, None] * h


# ---------------------------------------------------------------------------
#  Encoder / Decoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, cin=3, ch=128):
        super().__init__()
        self.layers = nn.Sequential(
            ConvSC(cin, ch, stride=1),
            ConvSC(ch, ch, stride=2),
            ConvSC(ch, ch, stride=2),
            ConvSC(ch, ch, stride=1),
        )

    def forward(self, x):
        skip = self.layers[0](x)
        x = self.layers[1](skip)
        x = self.layers[2](x)
        x = self.layers[3](x)
        return x, skip


class Decoder(nn.Module):
    def __init__(self, ch: int = 128, seq_len: int = 10):
        super().__init__()
        self.up1 = ConvSC(ch, ch, stride=2, transpose=True)
        self.up2 = ConvSC(ch, ch, stride=2, transpose=True)
        self.fuse = ConvSC(ch * 2, ch, stride=1)
        self.collapse = nn.Conv2d(seq_len * ch, 64, 1)
        self.attn = LKA(64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, lat, skip):
        b, s, c, h, w = lat.shape
        x = lat.reshape(b * s, c, h, w)
        x = self.up1(x)
        x = self.up2(x)
        skip = skip.view(b, -1, c, 64, 64)[:, -1]
        x = torch.cat([x, skip], 1)
        x = self.fuse(x)
        x = x.view(b, s * c, 64, 64)
        x = self.attn(self.collapse(x))
        return self.out(x)


# ---------------------------------------------------------------------------
#  Temporal predictor
# ---------------------------------------------------------------------------
class Predictor(nn.Module):
    def __init__(self, ch_in: int = 128, seq_len: int = 10, depth=5):
        super().__init__()
        dim = ch_in * seq_len * 2  # channels after cat(latent, mask)
        self.bottle = ConvNeXtBottle(dim)
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(depth)])
        self.slots = seq_len
        self.ch = ch_in

    def forward(self, z, m):
        b, s2, c, h, w = z.shape
        x = z.view(b, s2 * c, h, w)
        x = self.bottle(x, m)
        for blk in self.blocks:
            x = blk(x, m)
        x = x.view(b, s2, c, h, w)
        return x[:, self.slots :]


# ---------------------------------------------------------------------------
#  Top‑level model
# ---------------------------------------------------------------------------
class IAM4VP_ME(nn.Module):
    def __init__(self, ch_in=3, seq_len: int = 10, meta: int = 3, dim: int = 128, depth: int = 5):
        super().__init__()
        self.enc = Encoder(ch_in, dim)
        self.meta = MetadataEmbedding(meta)
        self.mask_token = nn.Parameter(torch.zeros(seq_len, dim, 16, 16))
        self.pred = Predictor(dim, seq_len, depth)
        self.dec = Decoder(dim, seq_len)
        self.slots = seq_len
        nn.init.zeros_(self.mask_token)

    def forward(self, x, meta, y_lat: list = []):
        B, S, C, H, W = x.shape
        assert S == self.slots
        x = x.reshape(B * S, C, H, W)
        lat, skip = self.enc(x)
        lat = lat.reshape(B, S, -1, 16, 16)
        mask = self.mask_token.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        for i, p in enumerate(y_lat):
            mask[:, i] = p
        z = torch.cat([lat, mask], 1)
        mvec = self.meta(meta)

        pred_lat_all = self.pred(z, mvec)  # (B, S, C, 16, 16)
        print(pred_lat_all.shape, "pred_lat_all")
        return pred_lat_all  # self.dec(pred_lat_all, skip)


# ---------------------------------------------------------------------------
#  Smoke‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B = 2
    model = IAM4VP_ME()
    x = torch.randn(B, 10, 3, 64, 64)
    meta = torch.tensor([[0.5, 0.6, 0.4]]).expand(B, -1)
    y = model(x, meta)
    print("OK →", y.shape)  # should be (B,1,64,64)
