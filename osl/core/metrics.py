import torch
from torch import nn


def ssim_metric(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute Structural Similarity Index between two batches."""
    data_range = max((x.max() - x.min()).item(), (y.max() - y.min()).item())
    if data_range < 1e-8:
        return torch.tensor(1.0, device=x.device)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    C = x.shape[1]
    coords = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(C, 1, -1, -1)
    pad = window_size // 2

    mu_x = nn.functional.conv2d(x, kernel, padding=pad, groups=C)
    mu_y = nn.functional.conv2d(y, kernel, padding=pad, groups=C)

    sigma_x_sq = nn.functional.conv2d(x ** 2, kernel, padding=pad, groups=C) - mu_x ** 2
    sigma_y_sq = nn.functional.conv2d(y ** 2, kernel, padding=pad, groups=C) - mu_y ** 2
    sigma_xy = nn.functional.conv2d(x * y, kernel, padding=pad, groups=C) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return ssim_map.mean()


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
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        layers_dim[i],
                        layers_dim[i + 1],
                        kernel_size=kernels[i],
                        stride=strides[i],
                        padding=paddings[i],
                        bias=False if i != 0 else True,
                    ),
                    nn.BatchNorm2d(layers_dim[i + 1])
                    if i != len(layers_dim) - 2 and i != 0
                    else nn.Identity(),
                    activation if i != len(layers_dim) - 2 else nn.Identity(),
                )
                for i in range(len(layers_dim) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def adversarial_loss(self, fake: torch.Tensor) -> torch.Tensor:
        for p in self.parameters():
            p.requires_grad_ = False
        logits = self(fake)
        return nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    
    def discriminative_loss(self, fake: torch.Tensor, true: torch.Tensor):
        for p in self.parameters():
            p.requires_grad_ = True
        # Discriminator loss
        logit_fake = self(fake.detach())
        logit_true = self(true)
        return 0.5 * (
            nn.functional.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))+ 
            nn.functional.binary_cross_entropy_with_logits(logit_true, torch.ones_like(logit_true))
        )

