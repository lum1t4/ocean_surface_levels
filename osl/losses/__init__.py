"""
Loss functions with registry and factory pattern.

Usage:
    from osl.losses import load_criterion, LossRegistry

    # Simple loss
    loss_fn = load_criterion("focal_loss", {"gamma": 2.0})
    total, parts = loss_fn(preds, targets)

    # Composite loss with weights
    loss_fn = load_criterion("0.9 * masked_mse_loss + 0.1 * gradient_loss")
    total, parts = loss_fn(preds, targets, mask=ocean_mask)

    # Register custom loss
    @register_loss("my_loss", aliases=["ml"])
    class MyLoss(nn.Module):
        ...
"""

from .registry import register_loss, load_criterion


# Import builtin losses to trigger registration
from . import builtin as builtin  # noqa: F401

__all__ = ["register_loss", "load_criterion"]
