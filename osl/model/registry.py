import os
from typing import Any, Dict, Type, Union
from urllib.request import urlretrieve

import torch
from osl.core.utils import LOGGER
from osl.core.pytorch import intersect_dicts, RANK
from pathlib import Path


CACHE_DIR = "~/.cache/model_registry"
CACHE_DIR = Path(CACHE_DIR).expanduser()

# --- Registry Singleton ---
class ModelRegistry:
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_model(
        cls,
        name: str,
        model_class: Type,
        model_config: Any,
        model_weights: Union[str, None] = None,
    ):
        """Register a model in the registry."""
        if name in cls._registry:
            raise ValueError(f"Model '{name}' is already registered.")
        cls._registry[name] = {
            "class": model_class,
            "config": model_config,
            "weights": model_weights,
        }

    @classmethod
    def get_model_entry(cls, name: str) -> Dict[str, Any]:
        if name not in cls._registry:
            raise KeyError(f"Model '{name}' not found in registry.")
        return cls._registry[name]


def retrive_weights(path_or_url: str, model_name: str) -> str:
    """Download weights if it's a URL, otherwise verify path exists."""
    if path_or_url.startswith("http"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        slug = model_name.replace(os.sep, '-') + Path(path_or_url).suffix
        local_path = CACHE_DIR / slug
        if not os.path.exists(local_path):
            print(f"Downloading weights from {path_or_url} to {local_path.as_posix()}...")
            urlretrieve(path_or_url, local_path)
        return local_path
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"Weight file not found: {path_or_url}")
    return Path(path_or_url)


def load_weights(model: torch.nn.Module, weight_path: Path, strict: bool = False) -> torch.nn.Module:
    if weight_path.suffix in {'.pt', '.pth', '.bin'}:
        ckpt = torch.load(weight_path, map_location="cpu")
    elif weight_path.suffix in {".safetensors"}:
        from safetensors.torch import load_file
        ckpt = load_file(weight_path, device="cpu")
    else:
        raise ValueError(f"Unsupported weight file format: {weight_path}")
    
    ckpt = ckpt['model'] if 'model' in ckpt else ckpt
    csd = intersect_dicts(ckpt, model.state_dict())  # intersect
    model.load_state_dict(csd, strict=strict)  # load

    # Warn only if there is a mismatch (intention if loading from a pretrained model)
    if RANK in {-1, 0}:
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights from {weight_path}")
    return model


def load_model(name: str, config: dict = {}, strict: bool = False, weights: str = None) -> torch.nn.Module:
    entry = ModelRegistry.get_model_entry(name)
    model_class = entry["class"]
    model_config = entry["config"]
    model_weights = weights if weights is not None else entry.get("weights", None)

    for k, v in config.items():
        if hasattr(model_config, k):
            setattr(model_config, k, v)
    
    model = model_class(model_config)
    if model_weights is not None:
        weight_path = retrive_weights(model_weights, name)
        model = load_weights(model, weight_path, strict=strict)

    return model

