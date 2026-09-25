"""Advantage Estimators for RL algorithms.

This module provides different advantage estimation strategies:
- GRPOAdvantageEstimator: Standard GRPO advantage with leave-one-out baseline
- ReinforcePlusPlusAdvantageEstimator: REINFORCE++ with optional baseline subtraction (minus_baseline) and KL penalty in reward
- OPDAdvantageEstimator: On-Policy Distillation (MOPD) token-level distillation advantages
Reference papers:
- ProRLv2: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
- REINFORCE++: https://arxiv.org/abs/2501.03262
- MOPD: https://arxiv.org/abs/2601.02780
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from nemo_rl.algorithms.utils import (
    calculate_baseline_and_std_per_prompt,
    calculate_batch_level_baseline,
    calculate_kl,
)

if TYPE_CHECKING:
    from torch import Tensor


class BaseAdvantageEstimator(ABC):
    def __init__(self, estimator_config: dict | None = None, loss_config: dict | None = None):
        pass

    @abstractmethod
    def compute_advantage(self, prompt_ids: Tensor, rewards: "Tensor", mask: "Tensor", **kwargs) -> "Tensor":
        """Compute advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
                    Used for: (1) expanding advantages to token-level shape, (2) global normalization
                    that only considers valid tokens.
            **kwargs: Additional arguments.

        Returns:
            Advantages tensor of shape [batch_size, seq_len], globally normalized across valid tokens.
        """
        ...


class GRPOAdvantageEstimator(BaseAdvantageEstimator):
    """GRPO-style advantage estimator with leave-one-out baseline.

    Note: GRPO computes advantages over all responses for each prompt.
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        self.use_leave_one_out_baseline = estimator_config["use_leave_one_out_baseline"]
        self.normalize_rewards = estimator_config["normalize_rewards"]

    def compute_advantage(self, prompt_ids, rewards, mask, **kwargs):
        """Compute GRPO advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
                  Used only for expanding advantages to token-level shape.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        baseline, std = calculate_baseline_and_std_per_prompt(
            prompt_ids,
            rewards,
            torch.ones_like(rewards),
            leave_one_out_baseline=self.use_leave_one_out_baseline,
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


class ReinforceAdvantageEstimator(BaseAdvantageEstimator):
    """Traditional REINFORCE advantage estimator.

    Args:
        minus_baseline (Optional, default False): If True, subtract the per-prompt mean reward baseline.
    """

    def __init__(self, estimator_config: dict, loss_config: dict | None = None):
        # We allow .get() here, this is an exception.
        self.minus_baseline = estimator_config.get("minus_baseline", False)
        self.leave_one_out_baseline = estimator_config.get("leave_one_out_baseline", False)
        self.batch_level_baseline = estimator_config.get("batch_level_baseline", False)

        if (not self.minus_baseline) and(self.batch_level_baseline or self.leave_one_out_baseline):
            print("[WARNING] minus_baseline=False. Ignoring batch_level_baseline and leave_one_out_baseline settings")

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        **kwargs,
    ):
        """Compute traditional REINFORCE advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt
                each sample belongs to.
            rewards: Tensor of shape [batch_size] containing the sequence reward
                for each sample.
            mask: Response token mask of shape [batch_size, seq_len], with 1 for
                valid response tokens and 0 for padding.
            **kwargs: Additional arguments (unused).

        Returns:
            Tensor of shape [batch_size, seq_len].
        """
        if self.minus_baseline:
            if self.batch_level_baseline:
                baseline = calculate_batch_level_baseline(
                    rewards,
                    torch.ones_like(rewards),
                    leave_one_out_baseline=self.leave_one_out_baseline,
                )
            else:
                baseline, _ = calculate_baseline_and_std_per_prompt(
                    prompt_ids,
                    rewards,
                    torch.ones_like(rewards),
                    leave_one_out_baseline=self.leave_one_out_baseline,
                )
            advantages = rewards - baseline
        else:
            advantages = rewards
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        # In sequence-level REINFORCE, every action/token in the sampled
        # response receives the same sequence return.
        advantages = advantages.unsqueeze(-1).expand_as(mask)

        return advantages * mask


class ReinforcePlusPlusAdvantageEstimator(BaseAdvantageEstimator):
    """REINFORCE++ advantage estimator with optional baseline subtraction and KL penalty in reward.

    Args:
        minus_baseline: If True, subtract per-prompt mean baseline from rewards.
        use_kl_in_reward: If True, add KL penalty to reward instead of loss.
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        self.minus_baseline = estimator_config["minus_baseline"]
        self.use_kl_in_reward = loss_config["use_kl_in_reward"]
        self.kl_coef = loss_config["reference_policy_kl_penalty"]
        self.kl_type = loss_config["reference_policy_kl_type"]

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        logprobs_policy=None,
        logprobs_reference=None,
        **kwargs,
    ):
        """Compute REINFORCE++ advantages with optional KL penalty.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding.
                  Used for: (1) expanding advantages to token-level shape, (2) global normalization
                  that only considers valid tokens.
            logprobs_policy: Policy log probabilities of shape [batch_size, seq_len], required if use_kl_in_reward.
            logprobs_reference: Reference policy log probabilities of shape [batch_size, seq_len], required if use_kl_in_reward.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len], globally normalized across valid tokens.
        """
        # minus baseline
        if self.minus_baseline:
            mean, _ = calculate_baseline_and_std_per_prompt(
                prompt_ids,
                rewards,
                torch.ones_like(rewards),
                leave_one_out_baseline=False,
            )
            adv = rewards - mean
        else:
            adv = rewards

        adv = adv.unsqueeze(-1)
        adv = adv.expand(mask.shape)

        # add kl penalty to reward (token-level)
        if (
            self.use_kl_in_reward
            and logprobs_policy is not None
            and logprobs_reference is not None
        ):
            kl = calculate_kl(
                logprobs_policy,
                logprobs_reference,
                kl_type=self.kl_type,
            )
            adv = adv - self.kl_coef * kl

        # global normalization across the batch
        adv_mean = (adv * mask).sum() / mask.sum()
        adv_var = ((adv - adv_mean).pow(2) * mask).sum() / mask.sum()
        adv_rstd = adv_var.clamp(min=1e-8).rsqrt()
        adv = (adv - adv_mean) * adv_rstd

        return adv


class OPDAdvantageEstimator(BaseAdvantageEstimator):
    """On-Policy Distillation advantage estimator (MOPD, arXiv:2601.02780).

    Computes token-level distillation advantages:
        Â_MOPD,t = sg[log π_teacher - log π_student]

    This is Equation 8 from the MOPD paper. The `IS` truncation (w_t, the
    hard gate on the training-to-inference ratio) is handled separately by
    ICE-POP mode in ClippedPGLoss — not here.

    Optionally blends in ORM reward advantages:
        Â = opd_advantage_weight * Â_MOPD + orm_advantage_weight * Â_ORM
    where the ORM could be one of any supported AdvantageEstimator classes.

    The loss function should be configured with:
        disable_ppo_ratio: true               (REINFORCE, no PPO ratio)
        use_importance_sampling_correction: true
        truncated_importance_sampling_type: icepop
        truncated_importance_sampling_ratio_min: <eps_low>
        truncated_importance_sampling_ratio: <eps_high>

    Required kwargs in compute_advantage:
        teacher_logprobs: [B, S] teacher model log probabilities
        prev_logprobs: [B, S] student training-engine log probabilities
    """

    ORM_ESTIMATOR_MAPPING: dict[str, type[BaseAdvantageEstimator]] = {
        "grpo": GRPOAdvantageEstimator,
        "reinforce": ReinforceAdvantageEstimator,
        "reinforceplusplus": ReinforcePlusPlusAdvantageEstimator,
        "reinforce++": ReinforcePlusPlusAdvantageEstimator,
    }

    def __init__(self, estimator_config: dict, loss_config: dict):
        self.opd_advantage_weight = float(estimator_config["opd_advantage_weight"])
        self.opd_advantage_clip_low = float(estimator_config.get("opd_advantage_clip_low", -6767))
        self.opd_advantage_clip_high = float(estimator_config.get("opd_advantage_clip_high", 6767))
        self.orm_advantage_weight = float(estimator_config.get("orm_advantage_weight", 0.0))
        self.orm_advantage_clip_low = float(estimator_config.get("orm_advantage_clip_low", -6767))
        self.orm_advantage_clip_high = float(estimator_config.get("orm_advantage_clip_high", 6767))
        self.zero_out_of_bounds_advantages = estimator_config.get("zero_out_of_bounds_advantages", False)

        if self.orm_advantage_weight > 0:
            orm_estimator_cfg = estimator_config.get("orm_advantage_estimator")
            if not orm_estimator_cfg:
                raise ValueError(
                    "OPD requires 'orm_advantage_estimator' in the estimator config "
                    "when orm_advantage_weight > 0."
                )
            self.orm_estimator_name: str = orm_estimator_cfg["orm_estimator_name"]
            if self.orm_estimator_name not in self.ORM_ESTIMATOR_MAPPING:
                raise ValueError(
                    f"Unsupported ORM advantage estimator: {self.orm_estimator_name!r}. "
                    f"Supported estimators are: {sorted(self.ORM_ESTIMATOR_MAPPING.keys())}"
                )
            self.orm_estimator = self.ORM_ESTIMATOR_MAPPING[self.orm_estimator_name](
                orm_estimator_cfg, loss_config
            )

        self.last_metrics: dict[str, float] = {}

    def compute_advantage(
        self,
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=None,
        prev_logprobs=None,
        **kwargs,
    ):
        """Compute OPD distillation advantages, optionally blended with ORM.

        Args:
            prompt_ids: [B] prompt IDs.
            rewards: [B] rewards (used only when orm_advantage_weight > 0).
            mask: [B, S] token mask.
            teacher_logprobs: [B, S] teacher model logprobs (required).
            prev_logprobs: [B, S] student training-engine logprobs (required).

        Returns:
            [B, S] token-level advantages (stop-gradient).
        """
        if teacher_logprobs is None:
            raise ValueError("OPD requires teacher_logprobs")
        if prev_logprobs is None:
            raise ValueError("OPD requires prev_logprobs")

        # Â_MOPD,t = sg[log π_teacher - log π_student]  (Equation 8)
        # NaN = collection-time context-length sentinel → zero OPD contribution for
        # those rows; ORM blending below is unaffected.
        distill_advantages = torch.nan_to_num(
            (teacher_logprobs - prev_logprobs).detach(), nan=0.0
        )
        distill_advantages = self._apply_advantage_bounds(
            distill_advantages, self.opd_advantage_clip_low, self.opd_advantage_clip_high
        )
        combined = self.opd_advantage_weight * distill_advantages

        if self.orm_advantage_weight > 0:
            orm_advantages = self.orm_estimator.compute_advantage(
                prompt_ids, rewards, mask, **kwargs
            )
            orm_advantages = self._apply_advantage_bounds(
                orm_advantages, self.orm_advantage_clip_low, self.orm_advantage_clip_high
            )
            combined = combined + self.orm_advantage_weight * orm_advantages
        else:
            orm_advantages = None

        # Apply mask
        advantages = combined * mask
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        # Metrics
        self._compute_metrics(distill_advantages, advantages, mask, orm_advantages=orm_advantages)

        return advantages

    def _apply_advantage_bounds(self, advantages: "Tensor", clip_low: float, clip_high: float) -> "Tensor":
        """Apply [clip_low, clip_high] bounds to advantages.

        Clips by default; zeroes out-of-bounds values instead when
        zero_out_of_bounds_advantages is enabled.
        """
        if self.zero_out_of_bounds_advantages:
            return torch.where(
                (advantages >= clip_low) & (advantages <= clip_high),
                advantages,
                torch.zeros_like(advantages),
            )
        return advantages.clamp(min=clip_low, max=clip_high)

    def _compute_metrics(self, distill_advantages: "Tensor", advantages: "Tensor", mask: "Tensor", orm_advantages: "Tensor" | None = None):
        """Compute OPD logging metrics and store in self.last_metrics."""
        valid_bool = mask.bool()
        distill_valid = torch.masked_select(distill_advantages, valid_bool)
        adv_valid = torch.masked_select(advantages, valid_bool)

        distill_mean = distill_valid.mean().item() if distill_valid.numel() > 0 else 0.0
        adv_mean = adv_valid.mean().item() if adv_valid.numel() > 0 else 0.0
        adv_std = adv_valid.std().item() if adv_valid.numel() > 1 else 0.0

        self.last_metrics = {
            "on_policy_distillation/teacher_student_logprob_gap_mean": distill_mean,
            "on_policy_distillation/adv_mean": adv_mean,
            "on_policy_distillation/adv_std": adv_std,
        }

        if orm_advantages is not None:
            orm_valid = torch.masked_select(orm_advantages, valid_bool)
            self.last_metrics[f"on_policy_distillation/{self.orm_estimator_name}_adv_mean"] = (
                orm_valid.mean().item() if orm_valid.numel() > 0 else 0.0
            )
            self.last_metrics[f"on_policy_distillation/{self.orm_estimator_name}_adv_std"] = (
                orm_valid.std().item() if orm_valid.numel() > 1 else 0.0
            )
