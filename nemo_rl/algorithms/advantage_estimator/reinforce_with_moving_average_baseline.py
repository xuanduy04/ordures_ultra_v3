from __future__ import annotations

from typing import Any, Mapping

from nemo_rl.algorithms.advantage_estimator.base import BaseAdvantageEstimator


class ReinforceWithMovingAverageBaselineAdvantageEstimator(BaseAdvantageEstimator):
    name = "REINFORCE w/ moving average baseline"
    """REINFORCE with a moving average baseline (Ahmadian et al., 2024, Eq. 9).

    The baseline is the mean of every reward seen so far in training, the current step included. Rewards are not
    normalized. The running reward sums and counts are saved in and restored from checkpoints.

    Args:
        per_environment (Optional, default False): If True, keep one moving average per environment (samples sharing an
            ``agent_ref["name"]``) instead of one for all samples.
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        super().__init__(estimator_config, loss_config)
        self.reward_stats: dict[str, list[float]] = {}
        # Baseline usage
        self._ignore_non_default(
            "minus_baseline", True, "REINFORCE w/ moving average baseline without baseline is just 'REINFORCE (plain)'"
        )
        self._ignore_non_default(
            "use_leave_one_out_baseline", False, "The moving average baseline is not leave-one-out."
        )
        self._ignore_non_default(
            "batch_level_baseline", False, "The moving average baseline already spans all batches."
        )

        # Normalization
        self._ignore_non_default(
            "normalize_rewards", False, "REINFORCE w/ moving average baseline does not normalize rewards."
        )

        # Per-environment
        self.per_environment = estimator_config.get("per_environment", False)

        # Name
        if self.per_environment:
            self.name += " (per-environment baseline)"

    def state_dict(self) -> dict[str, Any]:
        return {"moving_average_baseline": self.reward_stats}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.reward_stats = state.get("moving_average_baseline", {})

    def compute_advantage(self, prompt_ids, rewards, mask, agent_refs=None, **kwargs):
        """Compute REINFORCE w/ moving average baseline advantages.

        Args:
            prompt_ids: Unused.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding. Used
                only for expanding advantages to token-level shape.
            agent_refs: List of batch_size NeMo-Gym agent references, required if `per_environment` is True.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        if self.per_environment:
            if agent_refs is None:
                raise ValueError("The per-environment moving average baseline requires `agent_refs`")
            keys = [agent_ref["name"] for agent_ref in agent_refs]
        else:
            keys = ["all"] * len(rewards)

        for key, reward in zip(keys, rewards.tolist()):
            stats = self.reward_stats.setdefault(key, [0.0, 0])
            stats[0] += reward
            stats[1] += 1

        baseline = rewards.new_tensor([self.reward_stats[key][0] / self.reward_stats[key][1] for key in keys])
        return (rewards - baseline).unsqueeze(-1).expand(mask.shape)
