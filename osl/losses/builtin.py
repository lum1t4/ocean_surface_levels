"""Built-in loss functions for ocean surface level prediction."""

from typing import Optional
import torch
from torch import nn

from .registry import register_loss


def _broadcast_mask(mask: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    """
    Helper to align mask dimensions with the reference tensor.
    Example: Broadcasts mask (H, W) to match ref (B, T, C, H, W).
    """
    mask = mask.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    while mask.ndim < ref_tensor.ndim:
        mask = mask.unsqueeze(0)
    return mask


@register_loss("masked_mse", aliases=["masked_mse_loss"])
class MaskedMSELoss(nn.Module):
    """MSE loss with spatial masking (e.g., for ocean/land masks)."""

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return nn.functional.mse_loss(preds, targets)

        mask = _broadcast_mask(mask, preds)
        assert mask.dtype == preds.dtype, f"Mask dtype {mask.dtype} != preds dtype {preds.dtype}"

        diff = (preds - targets) * mask
        # Normalize by: mask_sum * B * T * C
        denom = (mask.sum() * preds.shape[0] * preds.shape[1] * preds.shape[2]).clamp_min(1.0)
        return (diff ** 2).sum() / denom


@register_loss("gradient_l1", aliases=["gradient_loss", "grad_l1"])
class GradientL1Loss(nn.Module):
    """
    L1 loss on spatial gradients (finite differences).
    Ensures gradients are only calculated between two valid pixels when masked.
    """

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute spatial gradients
        # dx: difference along last dim (W); dy: difference along 2nd to last dim (H)
        pred_dx = preds[..., :, 1:] - preds[..., :, :-1]
        target_dx = targets[..., :, 1:] - targets[..., :, :-1]

        pred_dy = preds[..., 1:, :] - preds[..., :-1, :]
        target_dy = targets[..., 1:, :] - targets[..., :-1, :]

        # L1 errors on gradients
        diff_dx = (pred_dx - target_dx).abs()
        diff_dy = (pred_dy - target_dy).abs()

        if mask is None:
            return 0.5 * (diff_dx.mean() + diff_dy.mean())

        # Create gradient masks - valid only if BOTH adjacent pixels are valid
        mask = mask.to(device=preds.device, dtype=preds.dtype)

        # Valid x-edges (cols 1..W AND cols 0..W-1)
        mask_dx_raw = mask[..., :, 1:] * mask[..., :, :-1]
        # Valid y-edges (rows 1..H AND rows 0..H-1)
        mask_dy_raw = mask[..., 1:, :] * mask[..., :-1, :]

        # Broadcast and normalize
        mask_dx = _broadcast_mask(mask_dx_raw, diff_dx).expand_as(diff_dx)
        mask_dy = _broadcast_mask(mask_dy_raw, diff_dy).expand_as(diff_dy)

        loss_dx = (diff_dx * mask_dx).sum() / mask_dx.sum().clamp_min(1.0)
        loss_dy = (diff_dy * mask_dy).sum() / mask_dy.sum().clamp_min(1.0)

        return 0.5 * (loss_dx + loss_dy)


@register_loss("mse", aliases=["mse_loss", "mean_squared_error"])
def mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss."""
    return nn.functional.mse_loss(preds, targets)


@register_loss("focal_loss", aliases=["focal"])
class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
