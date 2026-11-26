from __future__ import annotations

import torch

from nemo_rl.algorithms.advantage_estimator.base import BaseAdvantageEstimator


class KalmanFilterAdvantageEstimator(BaseAdvantageEstimator):
    name = "KRPO"
    """KRPO advantage estimator: GRPO with the group mean and std replaced by a per-prompt Kalman filter.

    As in the official code, the rewards of each prompt are fed in batch order to a fresh scalar Kalman filter with
    initial mean 0, initial variance 1, process noise variance Q = 1e-5 and measurement noise variance R = 1e-2. The
    advantage of response i is (r_i - x_i) / (sqrt(P_i) + 1e-8), where x_i and P_i are the filter's posterior mean and
    variance after observing r_i.
    """

    @property
    def is_group_normalization(self) -> bool:
        return True

    def __init__(self, estimator_config: dict, loss_config: dict):
        super().__init__(estimator_config, loss_config)
        # Baseline usage
        self._ignore_non_default(
            "use_leave_one_out_baseline", False, "KRPO with leave-one-out baseline is not defined."
        )
        self._ignore_non_default("minus_baseline", True, "KRPO without baseline is not defined.")
        self._ignore_non_default("batch_level_baseline", False, "KRPO with batch-level baseline is not defined.")

        # Normalization
        self._ignore_non_default("normalize_rewards", True, "KRPO without reward normalization is not defined.")

        # Per-environment
        self._ignore_non_default("per_environment", False, "KRPO is innately prompt-level.")

    def compute_advantage(self, prompt_ids, rewards, mask, **kwargs):
        """Compute KRPO advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size, ...] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding. Used
                only for expanding advantages to token-level shape.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        _, group_ids = torch.unique(prompt_ids.to(rewards.device), dim=0, return_inverse=True)
        one_hot = torch.nn.functional.one_hot(group_ids)
        positions = (one_hot.cumsum(0) * one_hot).sum(-1) - 1

        x = rewards.new_zeros(one_hot.shape[1])
        p = rewards.new_tensor(1.0)
        advantages = torch.empty_like(rewards)
        for i in range(int(positions.max()) + 1):
            at_i = positions == i
            groups_at_i = group_ids[at_i]
            p_pred = p + 1e-5
            k = p_pred / (p_pred + 1e-2)
            x[groups_at_i] = x[groups_at_i] + k * (rewards[at_i] - x[groups_at_i])
            p = (1 - k) * p_pred
            advantages[at_i] = (rewards[at_i] - x[groups_at_i]) / (p.sqrt() + 1e-8)

        return advantages.unsqueeze(-1).expand(mask.shape)
