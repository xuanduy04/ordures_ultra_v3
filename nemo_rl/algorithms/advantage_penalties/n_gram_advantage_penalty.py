from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, NotRequired, Self, TypedDict

import torch

from nemo_rl.algorithms.advantage_penalties.base_advantage_penalty import (
    BaseAdvantagePenalty,
    BaseAdvantagePenaltyDefinition,
    BaseAdvantagePenaltySettings,
    _resolve_common_settings,
)

if TYPE_CHECKING:
    from torch import Tensor

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class NGramAdvantagePenaltyConfig(TypedDict):
    """Config of the N-gram repetition advantage penalty in the grpo config."""

    enabled: bool

    size: int  # the N-gram size
    threshold: int  # occurrence count at which an N-gram is counted as a repeated phrase

    # Scale factor for the penalty magnitude (sign is ignored, falls back to ``advantage_penalty.default_weight``).
    weight: NotRequired[float]
    # Per-penalty normalizer override (falls back to ``advantage_penalty.default_advantage_penalty_normalizer``).
    advantage_penalty_normalizer: NotRequired[str]
    # Per-penalty max override (sign is ignored, falls back to ``advantage_penalty.default_max_advantage_penalty``).
    max_advantage_penalty: NotRequired[float]


@dataclass(frozen=True, kw_only=True)
class NGramAdvantagePenaltySettings(BaseAdvantagePenaltySettings):
    size: int  # the N-gram size
    threshold: int  # occurrence count at which an N-gram is counted as a repeated phrase

    @classmethod
    def from_config(cls, name: str, block: Mapping[str, Any], cfg: Mapping[str, Any], loss_config: dict) -> Self:
        """Resolve settings for penalty ``name`` from its sub-block, falling back to ``cfg``'s default_* keys."""
        return cls(
            size=block["size"], threshold=block["threshold"], **_resolve_common_settings(name, block, cfg, loss_config)
        )


class NGramAdvantagePenalty(BaseAdvantagePenalty[BaseAdvantagePenaltyDefinition, NGramAdvantagePenaltySettings]):
    """Penalty driven by the per-token N-gram repetition masks attached during rollouts.

    compute_signal() sums each rollout's n_gram_repetition_mask values (per-token overlap counts of repeated N-gram
    windows) into a per-rollout severity: the more repeated phrases cover a rollout's tokens, the worse it is. apply()
    spreads the weighted penalty over each offending assistant message's repeated tokens in proportion to their overlap
    count, so the most repeated tokens lose the most advantage.
    """

    DEFINITIONS = (
        BaseAdvantagePenaltyDefinition(name="n_gram_repetition", config_key="n_gram_repetition"),
    )  # fmt: skip

    def compute_signal(self, batch: BatchedDataDict) -> dict[str, Tensor]:
        message_logs = batch["message_log"]
        signals = {penalty.name: torch.zeros(len(message_logs), dtype=torch.float64) for penalty in self.penalties}
        for i, message_log in enumerate(message_logs):
            for message in message_log:
                repeat_mask = message.get("n_gram_repetition_mask")
                if repeat_mask is not None:
                    signals[self.penalties[0].name][i] += repeat_mask.sum().to(torch.float64)
        return signals

    def apply(self, advantages: Tensor, batch: BatchedDataDict, penalties: dict[str, Tensor]) -> None:
        message_logs = batch["message_log"]
        for i, message_log in enumerate(message_logs):
            token_offset = 0
            for j, message in enumerate(message_log):
                msg_len = len(message["token_ids"])
                repeat_mask = message.get("n_gram_repetition_mask")
                if repeat_mask is not None:
                    repeat_mask = repeat_mask.to(device=advantages.device, dtype=advantages.dtype)
                    if repeat_mask.max() > 0:
                        for penalty in self.penalties:
                            penalty_value = penalty.settings.weight * penalties[penalty.name][i].item()
                            if penalty_value < 0:
                                print(
                                    f"Setting negative advantage (max_penalty={penalty_value:.3f}) for {penalty.settings.size}-gram repetition in assistant message {i} turn {j}",
                                    flush=True,
                                )
                                advantages[i, token_offset : token_offset + msg_len].add_(
                                    repeat_mask / repeat_mask.max() * penalty_value
                                )
                token_offset += msg_len
