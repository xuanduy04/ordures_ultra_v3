from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


class BaseAdvantageEstimator(ABC):
    name: str = "[Unknown Advantage Estimator Name]"

    @property
    def is_group_normalization(self) -> bool:
        return False

    def __init__(self, estimator_config: dict | None = None, loss_config: dict | None = None):
        self.estimator_config = estimator_config

    @abstractmethod
    def compute_advantage(self, prompt_ids: Tensor, rewards: "Tensor", mask: "Tensor", **kwargs) -> "Tensor":
        """Compute advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding. Used
                for: (1) expanding advantages to token-level shape, (2) global normalization that only considers valid
                tokens.
            **kwargs: Additional arguments.

        Returns:
            Advantages tensor of shape [batch_size, seq_len], globally normalized across valid tokens.
        """
        ...

    def _ignore_non_default(self, key: str, default: bool, reason: str):
        if bool(self.estimator_config.get(key, default)) != default:
            print(f"[WARNING] Ignoring {key}={not default}. {reason}")
