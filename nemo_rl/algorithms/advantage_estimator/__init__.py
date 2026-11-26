"""Advantage Estimators for RL algorithms.

This module provides different advantage estimation strategies:
- GRPOAdvantageEstimator: Standard GRPO advantage with leave-one-out baseline
- ReinforcePlusPlusAdvantageEstimator: REINFORCE++ with optional baseline subtraction (minus_baseline) and KL penalty in
  reward
- OPDAdvantageEstimator: On-Policy Distillation (MOPD) token-level distillation advantages
- JamesSteinAdvantageEstimator: JSPO, RLOO baseline shrunk toward the leave-one-prompt-out batch mean
- KalmanFilterAdvantageEstimator: KRPO, GRPO with the group mean and std replaced by a per-prompt Kalman filter
- ReinforceWithMovingAverageBaselineAdvantageEstimator: REINFORCE with the moving average of all past rewards as
  baseline
Reference papers:
- ProRLv2: https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/
- REINFORCE++: https://arxiv.org/abs/2501.03262
- MOPD: https://arxiv.org/abs/2601.02780
- JSPO: https://arxiv.org/abs/2511.03710
- KRPO: https://arxiv.org/abs/2505.07527
- REINFORCE w/ moving average baseline: https://aclanthology.org/2024.acl-long.662
"""

from __future__ import annotations

import warnings

from typing import Any, Mapping, NotRequired, TypedDict

from nemo_rl.algorithms.advantage_estimator.base_advantage_estimator import BaseAdvantageEstimator
from nemo_rl.algorithms.advantage_estimator.grpo_advantage_estimator import GRPOAdvantageEstimator
from nemo_rl.algorithms.advantage_estimator.jspo_advantage_estimator import JamesSteinAdvantageEstimator
from nemo_rl.algorithms.advantage_estimator.krpo_advantage_estimator import KalmanFilterAdvantageEstimator
from nemo_rl.algorithms.advantage_estimator.opd_advantage_estimator import OPDAdvantageEstimator
from nemo_rl.algorithms.advantage_estimator.reinforce_advantage_estimator import ReinforcePlusPlusAdvantageEstimator
from nemo_rl.algorithms.advantage_estimator.reinforce_with_moving_average_baseline_advantage_estimator import (
    ReinforceWithMovingAverageBaselineAdvantageEstimator,
)


class AdvantageEstimatorConfig(TypedDict):
    """Configuration for advantage estimator."""

    name: str

    # Baselines
    # whether to use baselines (default: depend on estimator)
    minus_baseline: NotRequired[bool]
    # enable LOO baseline (default: False)
    use_leave_one_out_baseline: NotRequired[bool]
    # enable 'larger-than-prompt'-level baselines (default: False) NOTE: only some environments have this option
    batch_level_baseline: NotRequired[bool]

    # Normalization
    normalize_rewards: NotRequired[bool]  # (default: True), false would turn off normalization

    # Per-environment
    per_environment: NotRequired[bool]  # enable to use the "per-env" variant of the algorithm, if it is implemented


ADVANTAGE_ESTIMATOR_MAPPING: dict[str, type[BaseAdvantageEstimator]] = {
    "grpo": GRPOAdvantageEstimator,
    "reinforce++": ReinforcePlusPlusAdvantageEstimator,
    "reinforceplusplus": ReinforcePlusPlusAdvantageEstimator,
    "reinforce_plus_plus": ReinforcePlusPlusAdvantageEstimator,
    "jspo": JamesSteinAdvantageEstimator,
    "james_stein": JamesSteinAdvantageEstimator,
    "krpo": KalmanFilterAdvantageEstimator,
    "kalman_filter": KalmanFilterAdvantageEstimator,
    "reinforce_with_moving_average_baseline": ReinforceWithMovingAverageBaselineAdvantageEstimator,
    # "opd": OPDAdvantageEstimator,  # OPD is removed from this dict due to it requiring unique configurations
}


def create_advantage_estimator(master_config: Mapping[str, Any]) -> BaseAdvantageEstimator:
    """Create and return an advantage estimator based on configuration.

    Args:
        master_config: The master configuration dictionary.

    Returns:
        An advantage estimator instance.

    Raises:
        ValueError: If the advantage estimator name is not recognized.
    """
    grpo_config = master_config["grpo"]
    loss_config = master_config["loss_fn"]

    # Provide backward-compatible defaults when advantage_estimator is not in config.
    advantage_estimator_config = grpo_config.get("advantage_estimator", "'advantage_estimator' not specified in config")

    advantage_estimator_name = advantage_estimator_config["name"]
    if advantage_estimator_name == "opd":
        opd_cfg = master_config.get("on_policy_distillation", {})
        opd_estimator_config = {
            "name": "opd",
            "opd_advantage_weight": opd_cfg.get("opd_advantage_weight", 1.0),
            "opd_advantage_clip_low": opd_cfg.get("opd_advantage_clip_low", -6767),
            "opd_advantage_clip_high": opd_cfg.get("opd_advantage_clip_high", 6767),
            "orm_advantage_estimator": opd_cfg.get("orm_advantage_estimator", {}),
            "orm_advantage_weight": opd_cfg.get("orm_advantage_weight", 0.0),
            "orm_advantage_clip_low": opd_cfg.get("orm_advantage_clip_low", -6767),
            "orm_advantage_clip_high": opd_cfg.get("orm_advantage_clip_high", 6767),
            "zero_out_of_bounds_advantages": opd_cfg.get("zero_out_of_bounds_advantages", False),
        }
        advantage_estimator = OPDAdvantageEstimator(opd_estimator_config, loss_config)
        print("  ✓ Using OPD advantage estimator")

        # Warn if loss_fn is not configured per MOPD paper recommendations
        if not loss_config.get("disable_ppo_ratio", False):
            warnings.warn("OPD recommends loss_fn.disable_ppo_ratio: true (REINFORCE-style, MOPD Eq. 7)")
        if not loss_config.get("use_importance_sampling_correction", False):
            warnings.warn("OPD recommends loss_fn.use_importance_sampling_correction: true (MOPD Eq. 8 w_t)")
        if loss_config.get("truncated_importance_sampling_type", "default") != "icepop":
            warnings.warn("OPD recommends loss_fn.truncated_importance_sampling_type: 'icepop' (hard gate, MOPD Eq. 8)")
    else:
        if advantage_estimator_name not in ADVANTAGE_ESTIMATOR_MAPPING:
            raise ValueError(
                f"Invalid advantage_estimator name: {advantage_estimator_name}. "
                f"Supported estimators are: {sorted(ADVANTAGE_ESTIMATOR_MAPPING.keys())} and 'opd'"
            )
        advantage_estimator = ADVANTAGE_ESTIMATOR_MAPPING[advantage_estimator_name](
            advantage_estimator_config, loss_config
        )
        print(f"  ✓ Using {advantage_estimator.name} advantage estimator")

    return advantage_estimator
