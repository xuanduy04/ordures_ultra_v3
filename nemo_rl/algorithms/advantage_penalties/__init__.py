"""Advantage penalties for RL algorithms.

GDPO-style advantage-level penalties. Each penalty is a self-contained advantage-level reward that:

1. Extracts a per-rollout severity signal (higher = worse) from rollout metadata (e.g. message flags).
2. Normalizes it through a shared advantage estimator: the negated severity is fed as a pseudo-reward and the GDPO
   "no-bonus" shift ``(a - a.max()) / 2`` maps the best rollouts in the normalization group to exactly 0.
3. Subtracts a weighted share from the offending messages' token advantages.

The normalizer is any estimator from `ADVANTAGE_ESTIMATOR_MAPPING`
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Mapping, Sequence, NotRequired, TypedDict

import torch

from nemo_rl.algorithms.advantage_penalties.base_advantage_penalty import BaseAdvantagePenalty, DEFAULT_CONFIG_KEYS
from nemo_rl.algorithms.advantage_penalties.flag_advantage_penalty import (
    FlagAdvantagePenalty,
    FlagAdvantagePenaltyConfig,
    InvalidToolCallAdvantagePenaltyConfig,
)
from nemo_rl.algorithms.advantage_penalties.n_gram_advantage_penalty import (
    NGramAdvantagePenalty,
    NGramAdvantagePenaltyConfig,
)

if TYPE_CHECKING:
    from torch import Tensor

    from nemo_rl.algorithms.advantage_estimator import BaseAdvantageEstimator
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class AdvantagePenaltyConfig(TypedDict):
    """Nested advantage-penalty configuration under ``grpo.advantage_penalty``.

    Each penalty sub-block enables one penalty; per-penalty settings fall back to the ``default_*`` keys.
    """

    # 1. Invalid tool call tokens
    invalid_tool_call: NotRequired[InvalidToolCallAdvantagePenaltyConfig]
    # 2. Messages with malformed thinking
    malformed_thinking: NotRequired[FlagAdvantagePenaltyConfig]
    # 3. From the first EOS token in a generation to the final token (assistant turns only)
    eos_token: NotRequired[FlagAdvantagePenaltyConfig]
    # 4. Final assistant message when the final answer is empty
    empty_final_answer: NotRequired[FlagAdvantagePenaltyConfig]
    # 5. N-gram repetition, subtracted from token advantages.
    n_gram_repetition: NotRequired[NGramAdvantagePenaltyConfig]
    # 6. Assistant turns without a tool call when the environment + row expects one
    #    (see nemo_rl/algorithms/advantage_penalties/tool_call_expectations.py)
    missing_tool_call: NotRequired[FlagAdvantagePenaltyConfig]

    # Weight used by any enabled penalty that does not set its own weight.
    default_weight: NotRequired[float]
    # Advantage estimator used to normalize the penalty channels' pseudo-rewards.
    # Batch-level by default ("reinforce_plus_plus", the original GDPO-style behavior);
    # "grpo" gives per-prompt-group normalization, "reinforce" no normalization.
    default_advantage_penalty_normalizer: NotRequired[str]
    # Value bounding the per-token advantage reduction for penalties that do not set their own.
    default_max_advantage_penalty: NotRequired[float]


ADVANTAGE_PENALTY_TYPES: tuple[type[BaseAdvantagePenalty[Any, Any]], ...] = (
    FlagAdvantagePenalty,
    NGramAdvantagePenalty,
)


def create_advantage_penalties(
    grpo_config: Mapping[str, Any], loss_config: dict
) -> list[BaseAdvantagePenalty[Any, Any]]:
    cfg = grpo_config.get("advantage_penalty") or {}
    _validate_config_keys(cfg)
    return [
        penalty
        for penalty_type in ADVANTAGE_PENALTY_TYPES
        if (penalty := penalty_type.from_config(cfg, loss_config)) is not None
    ]


def _validate_config_keys(cfg: Mapping[str, Any]) -> None:
    claimed: dict[str, str] = {}
    for penalty_type in ADVANTAGE_PENALTY_TYPES:
        for key in penalty_type.config_keys():
            if key in claimed:
                raise RuntimeError(f"Config key {key!r} claimed by both {claimed[key]} and {penalty_type.__name__}")
            claimed[key] = penalty_type.__name__
    if unknown := set(cfg) - claimed.keys() - DEFAULT_CONFIG_KEYS:
        raise ValueError(
            f"Unknown advantage_penalty keys: {sorted(unknown)}. Known: {sorted(claimed.keys() | DEFAULT_CONFIG_KEYS)}"
        )


def normalize_penalty_signal(
    normalizer: BaseAdvantageEstimator, prompt_ids: Optional[Tensor], signal: Tensor, max_advantage_penalty: float
) -> Tensor:
    """Normalize a penalty's per-rollout severity signal into per-rollout penalties.

    The signal (>= 0, higher = worse) is negated into a pseudo-reward and run through the shared normalizer. A ones-mask
    of shape [batch_size, 1] keeps the estimator's normalization statistics at rollout level without expanding to
    [batch_size, seq_len]. The GDPO no-bonus shift ``(a - a.max()) / 2`` then maps the best rollouts to exactly 0 and
    pushes the rest negative, bounded below by ``-abs(max_advantage_penalty)``. A degenerate signal (constant across the
    group) yields all zeros, i.e. no signal to learn from.

    The no-bonus max follows the normalizer's granularity: batch-level schemes take the batch-wide max, while
    group-normalizing schemes (e.g. "grpo") take the per-prompt-group max.

    Args:
        normalizer: Advantage estimator providing the normalization scheme.
        prompt_ids: Tensor of shape [batch_size, ...] identifying which prompt each rollout belongs to. Used by
            group-normalizing schemes for both the normalization and the per-group no-bonus max.
        signal: Per-rollout severities of shape [batch_size], higher = worse.
        max_advantage_penalty: Magnitude bounding the per-rollout penalty.

    Returns:
        Per-rollout penalty values of shape [batch_size], <= 0.
    """
    batch_size = signal.shape[0]
    # Estimators operate on float32 rewards
    pseudo_rewards = -signal.to(torch.float32)
    normalized = normalizer.compute_advantage(
        prompt_ids=prompt_ids,  # pyrefly: ignore[bad-argument-type]  Group-normalizing schemes (e.g. GRPO) require prompt_ids; batch-level schemes ignore it
        rewards=pseudo_rewards,
        mask=torch.ones(batch_size, 1, dtype=torch.float32, device=pseudo_rewards.device),
    )
    if normalizer.is_group_normalization:
        if prompt_ids is None:
            raise ValueError(
                f"Group-normalizing penalty normalizer {normalizer.name!r} requires prompt_ids for the per-group no-bonus max"
            )
        # Group by unique prompt in the same way as
        # calculate_baseline_and_std_per_prompt in nemo_rl/algorithms/utils.py.
        no_bonus_max = torch.empty_like(normalized)
        for prompt in torch.unique(prompt_ids, dim=0):
            is_matching_prompt = (prompt_ids == prompt).all(1)
            no_bonus_max[is_matching_prompt] = normalized[is_matching_prompt].max()
    else:
        no_bonus_max = normalized.max()
    penalties = (normalized - no_bonus_max).div(2)
    penalties = penalties.clamp(min=-abs(max_advantage_penalty))
    return penalties.squeeze(-1)


def apply_advantage_penalties(
    advantages: Tensor,
    batch: BatchedDataDict,
    penalties: Sequence[BaseAdvantagePenalty[Any, Any]],
    prompt_ids: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Apply all enabled advantage penalties in-place to ``advantages``.

    For each penalty: extract severity signals from the batch, normalize them through the penalty's normalizer, scale
    them by the penalty's weight, and let the penalty place the weighted penalties into the offending messages' token
    slices.

    Args:
        advantages: Token-level advantages of shape [batch_size, seq_len]. Modified in-place.
        batch: The rollout batch containing the message logs.
        penalties: Penalties to apply.
        prompt_ids: Tensor of shape [batch_size, ...] identifying which prompt each rollout belongs to. Only used by
            group-normalizing schemes.

    Returns:
        Mapping of penalty name to the normalized and weighted per-rollout penalty values (exactly as applied) for
        logging.
    """
    penalties_log: dict[str, Tensor] = {}
    for penalty in penalties:
        signals = penalty.compute_signal(batch)
        weighted = {
            spec.name: spec.settings.weight
            * normalize_penalty_signal(
                spec.settings.normalizer, prompt_ids, signals[spec.name], spec.settings.max_advantage_penalty
            )
            for spec in penalty.penalties
        }
        penalty.apply(advantages, batch, weighted)
        penalties_log.update(weighted)
    return penalties_log
