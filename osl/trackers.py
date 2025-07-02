from pathlib import Path
from typing import List

from osl.dtypes import IterableSimpleNamespace
from osl.torch_utils import rank_zero_only

_WANDB_AVAILABLE = False

try:
    import wandb

    # TODO: Leverage wandb typing and uncomment the following lines
    # from wandb import Artifact
    # from wandb.sdk.lib import RunDisabled
    # from wandb.wandb_run import Run

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None


class Tracker:
    
    def __init__(self, config: dict | IterableSimpleNamespace):
        pass

    @rank_zero_only
    def log(self, x, y = None, step: int = None):
        pass

    @rank_zero_only
    def log_model(self, checkpoint: Path, aliases: List[str] = ["last"]):
        pass

    @rank_zero_only
    def finish(self):
        pass


class WandbTracker(Tracker):
    def __init__(self, project: str, name: str, config: dict):
        super().__init__(config)
        self.run = None
        if _WANDB_AVAILABLE:
            self.run = wandb.init(project=project, name=name, config=config, allow_val_change=True)
            if "monitor" in config and "mode" in config:
                self.run.define_metric(config["monitor"], summary=config["mode"])
    
    @rank_zero_only
    def log(self, x, y = None, step: int = None):
        if _WANDB_AVAILABLE:
            if isinstance(x, dict):
                self.run.log(x, step=step)
            else:
                self.run.log({x: y}, step=step)
        else:
            raise ImportError("wandb is not available. Please install it to use WandbTracker.")
    
    @rank_zero_only
    def log_model(self, checkpoint, aliases = ["last"]):
        if _WANDB_AVAILABLE:
            artifact = wandb.Artifact(f"run_{wandb.run.id}_model", type="model")
            artifact.add_file(checkpoint, name=checkpoint.name)
            wandb.run.log_artifact(artifact, aliases=aliases)

    @rank_zero_only
    def finish(self):
        if _WANDB_AVAILABLE:
            wandb.finish()
        else:
            raise ImportError("wandb is not available. Please install it to use WandbTracker.")
