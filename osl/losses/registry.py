from typing import Any, Callable, Dict, List, Type, Union
from dataclasses import dataclass
import torch
from torch import nn
import inspect
from typing import Tuple
import re


# nn.Module class, nn.Module instance, etc.
LossSpec = Union[nn.Module, Type[nn.Module], Callable[..., torch.Tensor]]
# TODO: Callable[..., nn.Module] factory function returning nn.Module

@dataclass(frozen=True)
class LossTerm:
    """A parsed loss term with weight and name."""
    weight: float
    name: str


_TERM_PATTERN = re.compile(r"""
^\s*
(?:                             # optional weighted form
    (?:\[\s*)?                  # optional opening [
    (?P<w>[0-9]*\.?[0-9]+)      # weight number
    (?:\s*\])?                  # optional closing ]
    \s*\*\s*                    # *
)?
(?P<n>[A-Za-z_]\w*)             # variable name
\s*$
""", re.VERBOSE)

ERROR_INVALID_LOSS_OUTPUT = NotImplementedError("Loss automatic evalutation currenty does not support losses with outputs different from torch.Tensor")


def _accepts_kwargs(object: Any) -> bool:
    """Check if a callable/module's forward accepts **kwargs."""
    try:
        function = object.forward if isinstance(object, nn.Module) else object
        signature = inspect.signature(function)
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())

    except Exception:
        return False
    

def _instantiate_loss_(spec: LossSpec, kwargs: Dict[str, Any]) -> nn.Module:
    """
    Turn registry spec into an nn.Module instance.

    - If spec is an nn.Module instance -> return it (ignore kwargs)
    - If spec is nn.Module class -> instantiate with kwargs
    """
    if isinstance(spec, nn.Module):
        return spec

    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec(**kwargs)
    
    if callable(spec):
        return spec

    raise TypeError(f"Unsupported loss spec type: {type(spec)}")


class LossRegistry:
    """
    A registry for loss functions that supports nn.Module classes,
    instances, and factory functions.
    """
    _registry: Dict[str, LossSpec] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, spec: LossSpec, aliases: List[str] = None) -> None:
        """
        Register a loss function.
        Args:
            name: Primary name for the loss
            spec: nn.Module class, instance, or factory function
            aliases: Optional list of alternative names
        """
        key = name.strip().lower()
        cls._registry[key] = spec
        for alias in (aliases or []):
            cls._aliases[alias.strip().lower()] = key

    @classmethod
    def _retrieve(cls, name: str) -> LossSpec:
        """Get a loss spec by name or alias."""
        key = name.strip().lower()
        return cls._registry[cls._aliases.get(key, key)]

    @classmethod
    def _instantiate(cls, name: str, **kwargs) -> nn.Module:
        """Create a loss instance by name with given parameters."""
        return _instantiate_loss_(cls._retrieve(name), kwargs)

    @classmethod
    def has(cls, name: str) -> bool:
        """Check if a loss is registered."""
        key = name.strip().lower()
        return key in cls._registry or key in cls._aliases

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available loss names including aliases."""
        return sorted(set(cls._registry.keys()) | set(cls._aliases.keys()))

    @classmethod
    def _clear_(cls) -> None:
        """Clear the registry (useful for testing)."""
        cls._registry.clear()
        cls._aliases.clear()


class EvaluatedLoss(nn.Module):
    def __init__(self, names: List[str], weights: List[float], losses: List[nn.Module]):
        super().__init__()
        assert len(names) == len(weights) == len(losses)
        self.loss_names = names
        self.loss_funs = losses
        self.register_buffer("_weights", torch.tensor(weights, dtype=torch.float32))


    @property
    def weights(self) -> List[float]:
        return self._weights.tolist()

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros(())
        weights = self._weights

        for idx, (name, fn) in enumerate(zip(self.loss_names, self.loss_funs)):
            loss = fn(*args, **kwargs)
            if not isinstance(loss, torch.Tensor):
                raise ERROR_INVALID_LOSS_OUTPUT
            if total.dtype != loss.dtype or total.device != loss.device:
                total = total.to(dtype=loss.dtype, device=loss.device)
                weights = weights.to(device=loss.device)

            losses[name] = loss
            total = total + losses[name] * weights[idx]

        return total, losses

    def __len__(self) -> int:
        return len(self.loss_funs)

    def __repr__(self) -> str:
        parts = [f"{w:.3g}*{n}" for w, n in zip(self.weights, self.loss_names)]
        return f"EvaluatedLoss({' + '.join(parts)})"


def _parse_term(expr: str) -> LossTerm:
    match = _TERM_PATTERN.match(expr)
    if not match:
        raise ValueError(f"Could not parse loss term: {expr!r}")
    w = float(match.group("w") or "1.0")   # default weight
    return LossTerm(weight=w, name=match.group("n"))


def _parse_expr(expr: str) -> List[LossTerm]:
    return [_parse_term(p.strip()) for p in expr.split('+') if p.strip()]


def register_loss(name: str, aliases: List[str] = None) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """
    Decorator to register a loss function.

    Args:
        name: Primary name for the loss
        aliases: Optional list of alternative names

    Example:
        @register_loss("focal_loss", aliases=["focal"])
        class FocalLoss(nn.Module):
            ...
    """
    def decorator(loss_cls: Type[nn.Module]) -> Type[nn.Module]:
        LossRegistry.register(name, loss_cls, aliases=aliases)
        return loss_cls
    return decorator


def load_criterion(expr: str, params: dict | None = None, **kwargs) -> EvaluatedLoss:
    loss_n: List[str] = []
    loss_w: List[float] = []
    loss_f: List[nn.Module] = []

    for term in _parse_expr(expr):
        loss_p = {}
        if params and term.name in params and isinstance(params[term.name], dict):
            loss_p.update(params[term.name])
        loss_p.update(kwargs)

        loss_n.append(term.name)
        loss_w.append(term.weight)
        loss_f.append(LossRegistry._instantiate(term.name, **loss_p))

    return EvaluatedLoss(loss_n, loss_w, loss_f)
