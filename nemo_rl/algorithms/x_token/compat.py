"""Adapt v0.7 X-Token bookkeeping to the production checkpoint API."""

import os
from pathlib import Path

import torch
import yaml

from nemo_rl.utils.checkpoint import CheckpointManager as ProductionCheckpointManager


class CheckpointManager(ProductionCheckpointManager):
    """Keep the production checkpoint layout and accept a validated config."""

    def __init__(self, config):
        self.save_optimizer = config["save_optimizer"]
        super().__init__(config)

    def get_resume_paths(self, checkpoint_path):
        """Resolve weights and optimizer paths only for an existing checkpoint."""
        if checkpoint_path is None:
            return None, None
        weights = Path(checkpoint_path) / "policy/weights"
        optimizer = Path(checkpoint_path) / "policy/optimizer"
        if not weights.exists():
            raise FileNotFoundError(weights)
        if self.save_optimizer and not optimizer.exists():
            raise FileNotFoundError(optimizer)
        return str(weights), str(optimizer) if self.save_optimizer else None

    def init_tmp_checkpoint(self, step, training_info, run_config=None):
        """Serialize the same effective config used by the training loop."""
        if run_config is not None:
            run_config = run_config.model_dump()
        return super().init_tmp_checkpoint(step, training_info, run_config)


def load_dataloader_state(dataloader, checkpoint_path, data_config):
    """Restore the dataloader from the production checkpoint layout."""
    with (Path(checkpoint_path) / "config.yaml").open() as config_file:
        saved_config = yaml.safe_load(config_file)
    if saved_config["data"] != data_config:
        raise ValueError("Refusing dataloader resume with a different data configuration")
    state_path = os.path.join(checkpoint_path, "train_dataloader.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    dataloader.load_state_dict(state)
