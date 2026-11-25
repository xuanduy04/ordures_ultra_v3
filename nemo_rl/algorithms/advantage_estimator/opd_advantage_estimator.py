from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nemo_rl.algorithms.advantage_estimator.base_advantage_estimator import BaseAdvantageEstimator

if TYPE_CHECKING:
    from torch import Tensor


class OPDAdvantageEstimator(BaseAdvantageEstimator):
    name = "OPD"
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

    def __init__(self, estimator_config: dict, loss_config: dict):
        super().__init__(estimator_config, loss_config)
        self.opd_advantage_weight = float(estimator_config["opd_advantage_weight"])
        self.opd_advantage_clip_low = float(estimator_config.get("opd_advantage_clip_low", -6767))
        self.opd_advantage_clip_high = float(estimator_config.get("opd_advantage_clip_high", 6767))
        self.orm_advantage_weight = float(estimator_config.get("orm_advantage_weight", 0.0))
        self.orm_advantage_clip_low = float(estimator_config.get("orm_advantage_clip_low", -6767))
        self.orm_advantage_clip_high = float(estimator_config.get("orm_advantage_clip_high", 6767))
        self.zero_out_of_bounds_advantages = estimator_config.get("zero_out_of_bounds_advantages", False)

        if self.orm_advantage_weight > 0:
            from nemo_rl.algorithms.advantage_estimator import ADVANTAGE_ESTIMATOR_MAPPING

            orm_estimator_cfg = estimator_config.get("orm_advantage_estimator")
            if not orm_estimator_cfg:
                raise ValueError(
                    "OPD requires 'orm_advantage_estimator' in the estimator config when orm_advantage_weight > 0."
                )
            self.orm_estimator_name: str = orm_estimator_cfg["orm_estimator_name"]
            if self.orm_estimator_name not in ADVANTAGE_ESTIMATOR_MAPPING:
                raise ValueError(
                    f"Unsupported ORM advantage estimator: {self.orm_estimator_name!r}. "
                    f"Supported estimators are: {sorted(ADVANTAGE_ESTIMATOR_MAPPING.keys())}"
                )
            self.orm_estimator = ADVANTAGE_ESTIMATOR_MAPPING[self.orm_estimator_name](orm_estimator_cfg, loss_config)
            self.name += f"with {self.orm_estimator.name}"

        self.last_metrics: dict[str, float] = {}

    def compute_advantage(self, prompt_ids, rewards, mask, teacher_logprobs=None, prev_logprobs=None, **kwargs):
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
        distill_advantages = torch.nan_to_num((teacher_logprobs - prev_logprobs).detach(), nan=0.0)
        distill_advantages = self._apply_advantage_bounds(
            distill_advantages, self.opd_advantage_clip_low, self.opd_advantage_clip_high
        )
        combined = self.opd_advantage_weight * distill_advantages

        if self.orm_advantage_weight > 0:
            orm_advantages = self.orm_estimator.compute_advantage(prompt_ids, rewards, mask, **kwargs)
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

        Clips by default; zeroes out-of-bounds values instead when zero_out_of_bounds_advantages is enabled.
        """
        if self.zero_out_of_bounds_advantages:
            return torch.where(
                (advantages >= clip_low) & (advantages <= clip_high), advantages, torch.zeros_like(advantages)
            )
        return advantages.clamp(min=clip_low, max=clip_high)

    def _compute_metrics(
        self, distill_advantages: "Tensor", advantages: "Tensor", mask: "Tensor", orm_advantages: "Tensor" | None = None
    ):
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
