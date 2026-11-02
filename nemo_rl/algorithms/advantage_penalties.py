"""Advantage penalties for RL algorithms.

GDPO-style advantage-level penalties. Each penalty is a self-contained advantage-level reward that:

1. Extracts a per-rollout severity signal (higher = worse) from rollout metadata (e.g. message flags).
2. Normalizes it through a shared advantage estimator: the negated severity is fed as a pseudo-reward and the GDPO
   "no-bonus" shift ``(a - a.max()) / 2`` maps the best rollouts in the normalization group to exactly 0.
3. Subtracts a weighted share from the offending messages' token advantages.

The normalizer is any estimator from `ADVANTAGE_ESTIMATOR_MAPPING`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NotRequired, Optional, TypedDict

import torch

from nemo_rl.algorithms.advantage_estimator import ADVANTAGE_ESTIMATOR_MAPPING, BaseAdvantageEstimator

if TYPE_CHECKING:
    from torch import Tensor

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


DEFAULT_ADVANTAGE_PENALTY_NORMALIZER_NAME = "grpo"
DEFAULT_ADVANTAGE_PENALTY_WEIGHT = 0.1
DEFAULT_MAX_ADVANTAGE_PENALTY = 6767


class FlagAdvantagePenaltyConfig(TypedDict):
    """Config of ONE flag-based advantage penalty in the grpo config."""

    enabled: bool
    # Scale factor for the penalty magnitude (falls back to ``advantage_penalty.default_weight``).
    weight: NotRequired[float]
    # Per-penalty normalizer override (falls back to
    # ``advantage_penalty.default_advantage_penalty_normalizer``).
    advantage_penalty_normalizer: NotRequired[str]
    # Per-penalty max override (falls back to ``advantage_penalty.default_max_advantage_penalty``).
    max_advantage_penalty: NotRequired[float]


class NGramAdvantagePenaltyConfig(TypedDict):
    """Config of the N-gram repetition advantage penalty in the grpo config."""

    enabled: bool
    # size: the N-gram size
    size: int
    # threshold: occurrence count at which an N-gram is counted as a repeated phrase
    threshold: int
    # Scale factor for the penalty magnitude.
    weight: NotRequired[float]
    max_advantage_penalty: NotRequired[float]


class AdvantagePenaltyConfig(TypedDict):
    """Nested advantage-penalty configuration under ``grpo.advantage_penalty``.

    Each penalty sub-block enables one penalty; per-penalty settings fall back to the ``default_*`` keys.
    """

    # 1. Invalid tool call tokens
    invalid_tool_call: NotRequired[FlagAdvantagePenaltyConfig]
    # 2. Messages with malformed thinking
    malformed_thinking: NotRequired[FlagAdvantagePenaltyConfig]
    # 3. From the first EOS token in a generation to the final token (assistant turns only)
    eos_token: NotRequired[FlagAdvantagePenaltyConfig]
    # 4. Final assistant message when the final answer is empty
    empty_final_answer: NotRequired[FlagAdvantagePenaltyConfig]
    # 5. N-gram repetition, subtracted from token advantages.
    n_gram_repetition: NotRequired[NGramAdvantagePenaltyConfig]
    # Weight used by any enabled penalty that does not set its own weight.
    default_weight: NotRequired[float]
    # Advantage estimator used to normalize the penalty channels' pseudo-rewards.
    # Batch-level by default ("reinforce_plus_plus", the original GDPO-style behavior);
    # "grpo" gives per-prompt-group normalization, "reinforce" no normalization.
    default_advantage_penalty_normalizer: NotRequired[str]
    # Value bounding the per-token advantage reduction for penalties that do not set their own.
    default_max_advantage_penalty: NotRequired[float]


class BaseAdvantagePenalty(ABC):
    """Base class for advantage penalties.

    A penalty is a pseudo-reward: it builds a per-rollout severity signal from the batch, has it normalized by the
    shared advantage-estimator normalizer, and places the weighted penalty into the offending messages' token
    advantages.
    """

    def __init__(self, penalties: BaseAdvantagePenaltySpec | list[BaseAdvantagePenaltySpec]):
        self.penalties = penalties if isinstance(penalties, list) else [penalties]

    @abstractmethod
    def compute_signal(self, batch: BatchedDataDict) -> dict[str, Tensor]:
        """Extract the per-rollout severity signal for every penalty.

        Args:
            batch: The rollout batch containing the message logs.

        Returns:
            Mapping of penalty name to a tensor of shape [batch_size] with non-negative severities (higher = worse).
        """
        ...

    @abstractmethod
    def apply(self, advantages: Tensor, batch: BatchedDataDict, penalties: dict[str, Tensor]) -> None:
        """Add weighted per-rollout penalties to the offending messages' tokens.

        Args:
            advantages: Token-level advantages of shape [batch_size, seq_len]. Modified in-place.
            batch: The rollout batch containing the message logs.
            penalties: Mapping of a penalty name to the normalized per-rollout penalty values (pre-weight) of shape
                [batch_size].
        """
        ...


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


@dataclass(frozen=True, kw_only=True)
class BaseAdvantagePenaltySpec:
    name: str  # Penalty name used in logging
    config_key: str  # ``advantage_penalty`` sub-block enabling and configuring the penalty
    weight: float = DEFAULT_ADVANTAGE_PENALTY_WEIGHT  # Scale factor for the penalty magnitude
    normalizer: Optional[BaseAdvantageEstimator] = None  # Advantage normalizer for the penalty signal
    max_advantage_penalty: float = DEFAULT_MAX_ADVANTAGE_PENALTY  # Magnitude bounding the per-rollout penalty


@dataclass(frozen=True, kw_only=True)
class FlagAdvantagePenaltySpec(BaseAdvantagePenaltySpec):
    """Static spec of one flag-based penalty in the grpo config.

    Doubles as the runtime penalty config consumed by `FlagAdvantagePenalty`.
    """

    message_key: str  # Message-level boolean key carrying the flag
    assistant_only: bool  # Only consider assistant-generated messages


FLAG_PENALTY_SPECS: tuple[FlagAdvantagePenaltySpec, ...] = (
    FlagAdvantagePenaltySpec(name="invalid_tool_call", config_key="invalid_tool_call", message_key="is_invalid_tool_call", assistant_only=False),
    FlagAdvantagePenaltySpec(name="malformed_thinking", config_key="malformed_thinking", message_key="has_malformed_thinking", assistant_only=True),
    FlagAdvantagePenaltySpec(name="generate_eos_token", config_key="eos_token", message_key="has_eos_token", assistant_only=True),
    FlagAdvantagePenaltySpec(name="empty_final_answer", config_key="empty_final_answer", message_key="has_empty_final_answer", assistant_only=True),
)  # fmt: skip


class FlagAdvantagePenalty(BaseAdvantagePenalty):
    """Penalties driven by message-level boolean flags.

    Each penalty yields a binary per-rollout severity: 1 if any message in the rollout's message log carries the flag
    (restricted to assistant-generated messages when assistant_only is set). apply() adds weight * penalty to every
    offending message's token advantages.

    All flag-based penalties can be grouped and applied together in a single loop.
    """

    def __init__(self, penalties: list[FlagAdvantagePenaltySpec]):
        super().__init__(penalties)

    def compute_signal(self, batch: BatchedDataDict) -> dict[str, Tensor]:
        message_logs = batch["message_log"]
        signals = {penalty.name: torch.zeros(len(message_logs), dtype=torch.float64) for penalty in self.penalties}
        for i, message_log in enumerate(message_logs):
            for message in message_log:
                is_assistant = message["role"] == "assistant" and "generation_logprobs" in message
                for penalty in self.penalties:
                    if message.get(penalty.message_key, False) and (is_assistant or not penalty.assistant_only):
                        signals[penalty.name][i] = 1.0
        return signals

    def apply(self, advantages: Tensor, batch: BatchedDataDict, penalties: dict[str, Tensor]) -> None:
        message_logs = batch["message_log"]
        for i, message_log in enumerate(message_logs):
            token_offset = 0
            for j, message in enumerate(message_log):
                msg_len = len(message["token_ids"])
                is_assistant = message["role"] == "assistant" and "generation_logprobs" in message
                for penalty in self.penalties:
                    penalty_value = penalty.weight * penalties[penalty.name][i].item()
                    if (
                        penalty_value < 0
                        and message.get(penalty.message_key, False)
                        and (is_assistant or not penalty.assistant_only)
                    ):
                        print(
                            f"Setting negative advantage ({penalty_value:.3f}) for {penalty.name} in assistant message {i} turn {j}",
                            flush=True,
                        )
                        advantages[i, token_offset : token_offset + msg_len].add_(penalty_value)
                token_offset += msg_len


def apply_advantage_penalties(
    advantages: Tensor,
    batch: BatchedDataDict,
    penalties: list[FlagAdvantagePenalty],
    prompt_ids: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Apply all enabled advantage penalties in-place to ``advantages``.

    For each penalty: extract severity signals from the batch, normalize them through the penalty's normalizer, and let
    the penalty place weighted penalties into the offending messages' token slices.

    Args:
        advantages: Token-level advantages of shape [batch_size, seq_len]. Modified in-place.
        batch: The rollout batch containing the message logs.
        penalties: Penalties to apply.
        prompt_ids: Tensor of shape [batch_size, ...] identifying which prompt each rollout belongs to. Only used by
            group-normalizing schemes.

    Returns:
        Mapping of penalty name to the normalized per-rollout penalty values (pre-weight), for logging.
    """
    penalties_log: dict[str, Tensor] = {}
    for penalty in penalties:
        signals = penalty.compute_signal(batch)
        normalized = {
            penalty_spec.name: normalize_penalty_signal(
                penalty_spec.normalizer, prompt_ids, signals[penalty_spec.name], penalty_spec.max_advantage_penalty
            )
            for penalty_spec in penalty.penalties
        }
        penalty.apply(advantages, batch, normalized)
        penalties_log.update(normalized)
    return penalties_log


_PENALTY_NORMALIZER_DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "reinforce_plus_plus": {"minus_baseline": False},
    "reinforceplusplus": {"minus_baseline": False},
    "reinforce++": {"minus_baseline": False},
    "grpo": {"normalize_rewards": True, "use_leave_one_out_baseline": False},
    "reinforce": {},
}


def create_advantage_penalties(grpo_config: dict[str, Any], loss_config: dict) -> list[FlagAdvantagePenalty]:
    """Build the enabled advantage penalties from the nested ``grpo.advantage_penalty`` config.

    Each penalty sub-block can override ``weight``, ``advantage_penalty_normalizer``, and ``max_advantage_penalty``.
    Unset values fall back to the ``default_*`` keys, which themselves fall back to the module-level
    ``DEFAULT_ADVANTAGE_PENALTY_WEIGHT``, ``DEFAULT_ADVANTAGE_PENALTY_NORMALIZER_NAME``, and
    ``DEFAULT_MAX_ADVANTAGE_PENALTY``.

    Args:
        grpo_config: The ``grpo`` section of the master config.
        loss_config: The ``loss_fn`` section of the master config, required by some normalizer constructors.

    Returns:
        List of enabled penalties (empty if none are configured).
    """
    advantage_penalty_cfg = grpo_config.get("advantage_penalty", {})

    penalties: list[FlagAdvantagePenaltySpec] = []
    for penalty_spec in FLAG_PENALTY_SPECS:
        penalty_block = advantage_penalty_cfg.get(penalty_spec.config_key, {})
        if not penalty_block.get("enabled", False):
            continue
        normalizer_name = penalty_block.get(
            "advantage_penalty_normalizer",
            advantage_penalty_cfg.get(
                "default_advantage_penalty_normalizer", DEFAULT_ADVANTAGE_PENALTY_NORMALIZER_NAME
            ),
        )
        if normalizer_name not in ADVANTAGE_ESTIMATOR_MAPPING:
            raise ValueError(
                f"Invalid advantage_penalty_normalizer: {normalizer_name!r} for penalty {penalty_spec.name!r}. "
                f"Supported estimators are: {sorted(ADVANTAGE_ESTIMATOR_MAPPING.keys())}"
            )
        normalizer_config = {**_PENALTY_NORMALIZER_DEFAULT_CONFIGS[normalizer_name], "name": normalizer_name}
        normalizer = ADVANTAGE_ESTIMATOR_MAPPING[normalizer_name](normalizer_config, loss_config)
        penalties.append(
            replace(
                penalty_spec,
                weight=penalty_block.get(
                    "weight", advantage_penalty_cfg.get("default_weight", DEFAULT_ADVANTAGE_PENALTY_WEIGHT)
                ),
                normalizer=normalizer,
                max_advantage_penalty=penalty_block.get(
                    "max_advantage_penalty",
                    advantage_penalty_cfg.get("default_max_advantage_penalty", DEFAULT_MAX_ADVANTAGE_PENALTY),
                ),
            )
        )

    return [FlagAdvantagePenalty(penalties=penalties)] if penalties else []


ADVANTAGE_PENALTY_MAPPING: dict[str, type[BaseAdvantagePenalty]] = {
    "flag": FlagAdvantagePenalty
    # Future penalty types (n-gram repetition, overlong, stop-properly) register here.
    # Right now this thing is useless.
}
