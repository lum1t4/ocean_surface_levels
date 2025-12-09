from pathlib import Path
from typing import List

from osl.core.pytorch import RANK, rank_zero_only

WANDB_AVAILABLE = False


class Tracker:
    def __init__(self, config: dict = None):
        pass

    @rank_zero_only
    def log(self, x: dict, step: int = None):
        pass

    @rank_zero_only
    def log_model(self, checkpoint: Path, aliases: List[str] = ["last"]):
        pass


class TrackerRegistry:
    _registry: dict[str, Tracker] = {}

    @classmethod
    def register_tracker(cls, name: str, tracker: Tracker):
        cls._registry[name] = tracker

    @classmethod
    def list_trackers(cls):
        return list(cls._registry.keys())
    
    @classmethod
    def get_tracker(cls, name: str):
        return cls._registry[name]


def load_tracker(name: str | None, config: dict):
    if name is None or RANK not in {-1, 0}:
        return Tracker(config)
    return TrackerRegistry.get_tracker(name)(config)



try:
    import wandb
    WANDB_AVAILABLE = True

    class WandbTracker(Tracker):
        def __init__(self, config: dict):
            super().__init__(config)
            if not WANDB_AVAILABLE:
                raise ImportError("wandb is not available. Please install it to use WandbTracker.")

            if RANK not in {-1, 0}:
                self.run = None
            else:
                self.run = wandb.init(
                project=config['project'],
                name=config['name'],
                config=config,
                allow_val_change=True
            )

        @rank_zero_only
        def log(self, x, step: int = None):
            self.run.log(x, step=step)

        @rank_zero_only
        def log_model(self, checkpoint, aliases = ["last"]):
            artifact = wandb.Artifact(f"run_{wandb.run.id}_model", type="model")
            artifact.add_file(checkpoint, name=checkpoint.name)
            wandb.run.log_artifact(artifact, aliases=aliases)

    if RANK in {-1, 0}:
        TrackerRegistry.register_tracker("wandb", WandbTracker)

except ImportError:
    wandb = None



