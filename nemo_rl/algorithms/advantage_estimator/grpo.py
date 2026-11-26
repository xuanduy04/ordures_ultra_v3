from __future__ import annotations

import torch

from nemo_rl.algorithms.advantage_estimator.base import BaseAdvantageEstimator
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt


class GRPOAdvantageEstimator(BaseAdvantageEstimator):
    name = "GRPO"
    """GRPO-style advantage estimator with leave-one-out baseline.

    Note: GRPO computes advantages over all responses for each prompt.
    """

    @property
    def is_group_normalization(self) -> bool:
        return True

    def __init__(self, estimator_config: dict, loss_config: dict):
        super().__init__(estimator_config, loss_config)
        # Baseline usage
        self.use_leave_one_out_baseline = estimator_config.get("use_leave_one_out_baseline", False)

        self._ignore_non_default("minus_baseline", True, "GRPO without baseline is unheard of.")
        self._ignore_non_default("batch_level_baseline", False, "GRPO with batch-level baseline is unheard of.")

        # Normalization
        self.normalize_rewards = estimator_config.get("normalize_rewards", True)

        # Per-environment
        self._ignore_non_default("per_environment", False, "GRPO is innately prompt-level.")

        # Name
        if not self.normalize_rewards:
            self.name = "DR.GRPO"
        if self.use_leave_one_out_baseline:
            self.name += " (LOO baseline)"

    def compute_advantage(self, prompt_ids, rewards, mask, **kwargs):
        """Compute GRPO advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding. Used
                only for expanding advantages to token-level shape.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        baseline, std = calculate_baseline_and_std_per_prompt(
            prompt_ids, rewards, torch.ones_like(rewards), leave_one_out_baseline=self.use_leave_one_out_baseline
        )
        advantages = (rewards - baseline).unsqueeze(-1)

        if self.normalize_rewards:
            # don't sharpen the ones with no variation
            epsilon = 1e-6
            non_zero_std_mask = std > 0
            advantages[non_zero_std_mask] = advantages[non_zero_std_mask] / (
                std.unsqueeze(-1)[non_zero_std_mask] + epsilon
            )

        return advantages.expand(mask.shape)
