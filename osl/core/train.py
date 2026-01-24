from copy import deepcopy
from datetime import datetime, timedelta
import gc
import io
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from osl.core.pytorch import (
    LOCAL_RANK,
    RANK,
    WORLD_SIZE,
    convert_optimizer_state_dict_to_fp16,
    de_parallel,
    device_memory_clear,
    intersect_dicts,
    rank_zero_only,
    set_seeds
)
from osl.core.trackers import Tracker
from osl.core.utils import LOGGER, IterableSimpleNamespace
from osl.core.trackers import load_tracker


class TrainContext:
    config: IterableSimpleNamespace
    model: nn.Module | nn.parallel.DistributedDataParallel
    device: torch.device
    tracker: Tracker
    save_dir: Path
    fitness: float
    train_dataset: Dataset
    valid_dataset: Dataset | None
    train_loader: DataLoader
    valid_loader: DataLoader | None
    metrics: dict[str, Any] = {}
    start_iter: int = 0
    best_iter: int = 0
    curr_iter: int = None
    stop: bool = False
    name: str | None = None

    def __init__(self, config: IterableSimpleNamespace):
        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        # Set seed
        set_seeds(config.seed, deterministic=config.deterministic)

        self.config = config
        self.device = torch.device(config.device)
        self.save_dir = Path(config.save_dir)
        self.fitness = float("-inf") if config.mode == "max" else float("inf")
        self.tracker = load_tracker(name=self.config.tracker, config=vars(self.config))

        # Distributed init
        if WORLD_SIZE > 1:
            torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=10800)) # 3 hours
            torch.cuda.set_device(LOCAL_RANK)
            self.device = torch.device(f"cuda:{LOCAL_RANK}")
        
        # Checks and patches on config
        if self.device.type in {"cpu", "mps"}:
            self.config.workers = 0
            self.config.amp = False


    @property
    def weights_dir(self) -> Path:
        name = self.config.name if self.config.name else self.config.model
        w = self.save_dir / name / 'weights'
        w.mkdir(parents=True, exist_ok=True)
        return w
    
    @property
    def plt_dir(self) -> Path:
        name = self.config.name if self.config.name else self.config.model
        p = self.save_dir / name / 'plots'
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def last_checkpoint(self) -> Path:
        return self.weights_dir / "last.pth"
    
    @property
    def best_checkpoint(self) -> Path:
        return self.weights_dir / 'best.pth'
    
    @property
    def current_checkpoint(self) -> Path:
        return self.weights_dir / f"epoch_{self.curr_iter}.pt"


    def iteration_end(self):
        gc.collect()
        device_memory_clear(self.device)



    @rank_zero_only
    def checkpointing(self):
        buffer = io.BytesIO()
        model = de_parallel(self.model) if WORLD_SIZE > 1 else self.model
        optim = convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict()))

        torch.save({
            "epoch": self.curr_iter,
            "model": model.state_dict(),
            "optimizer": optim,
            "metrics": self.metrics,
            "config": vars(self.config),
            "date": datetime.now().isoformat(),
        }, buffer)

        ckpt = buffer.getvalue()
        self.last_checkpoint.write_bytes(ckpt)
        if self.curr_iter == self.best_iter:
            self.best_checkpoint.write_bytes(ckpt)
            self.tracker.log_model(self.best_checkpoint, aliases=["best"])
        if self.config.save_period > 0 and (self.curr_iter + 1) % self.config.save_period == 0:
            self.current_checkpoint.write_bytes(ckpt)
        return self
    

    def resume(self):
        """Load pretrained or resume from checkpoint """
        if self.config.weights is None:
            return self

        weights = Path(self.config.weights)
        
        if not weights.exists():
            LOGGER.warning(f"Could not find specified weights at {weights}")
            return self
        
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        if self.config.resume:
            LOGGER.info("Resuming training")
            msd = ckpt["model"]

            if isinstance(ckpt["optimizer"], tuple):
                ckpt["optimizer"] = ckpt["optimizer"][0]
            
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_iter = ckpt["epoch"] + 1
            LOGGER.info(f"Resuming training from epoch {self.start_iter}")
        else:
            msd = ckpt
        csd = intersect_dicts(msd, self.model.state_dict())  # intersect
        self.model.load_state_dict(csd, strict=False)  # load
        if RANK in {-1, 0}:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")
        return self
    

    def early_stopping(self):
        x = self.metrics[self.config.monitor]
        y = self.fitness
        if (self.config.mode == "max" and x >= y) or (self.config.mode == "min" and x <= y):
            self.fitness = x
            self.best_iter = self.curr_iter

        if self.curr_iter - self.best_iter == self.config.patience:
            LOGGER.info(f"Triggered Early Stopping at epoch {self.curr_iter + 1}")
            self.stop = True
        return self