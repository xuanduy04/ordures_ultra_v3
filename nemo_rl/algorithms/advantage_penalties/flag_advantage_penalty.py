from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict

import torch

from nemo_rl.algorithms.advantage_penalties.base_advantage_penalty import (
    BaseAdvantagePenalty,
    BaseAdvantagePenaltyDefinition,
    BaseAdvantagePenaltySettings,
)

if TYPE_CHECKING:
    from torch import Tensor

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class FlagAdvantagePenaltyConfig(TypedDict):
    """Config of ONE flag-based advantage penalty in the grpo config."""

    enabled: bool

    # Scale factor for the penalty magnitude (sign is ignored, falls back to ``advantage_penalty.default_weight``).
    weight: NotRequired[float]
    # Per-penalty normalizer override (falls back to ``advantage_penalty.default_advantage_penalty_normalizer``).
    advantage_penalty_normalizer: NotRequired[str]
    # Per-penalty max override (sign is ignored, falls back to ``advantage_penalty.default_max_advantage_penalty``).
    max_advantage_penalty: NotRequired[float]


@dataclass(frozen=True, kw_only=True)
class FlagAdvantagePenaltyDefinition(BaseAdvantagePenaltyDefinition):
    message_key: str  # Message-level boolean key carrying the flag
    assistant_only: bool  # Only consider messages with "role" = "assistant"


class FlagAdvantagePenalty(BaseAdvantagePenalty[FlagAdvantagePenaltyDefinition, BaseAdvantagePenaltySettings]):
    """Penalties driven by message-level boolean flags.

    Each penalty yields a binary per-rollout severity: 1 if any message in the rollout's message log carries the flag
    (restricted to assistant-generated messages when assistant_only is set). apply() adds weight * penalty to every
    offending message's token advantages.

    All flag-based penalties can be grouped and applied together in a single loop.
    """

    DEFINITIONS = (
        FlagAdvantagePenaltyDefinition(name="invalid_tool_call", config_key="invalid_tool_call", message_key="is_invalid_tool_call", assistant_only=False),
        FlagAdvantagePenaltyDefinition(name="malformed_thinking", config_key="malformed_thinking", message_key="has_malformed_thinking", assistant_only=True),
        FlagAdvantagePenaltyDefinition(name="generate_eos_token", config_key="eos_token", message_key="has_eos_token", assistant_only=True),
        FlagAdvantagePenaltyDefinition(name="empty_final_answer", config_key="empty_final_answer", message_key="has_empty_final_answer", assistant_only=True),
    )  # fmt: skip

    def compute_signal(self, batch: BatchedDataDict) -> dict[str, Tensor]:
        message_logs = batch["message_log"]
        signals = {penalty.name: torch.zeros(len(message_logs), dtype=torch.float64) for penalty in self.penalties}
        for i, message_log in enumerate(message_logs):
            for message in message_log:
                is_assistant = message["role"] == "assistant" and "generation_logprobs" in message
                for penalty in self.penalties:
                    if message.get(penalty.definition.message_key, False) and (
                        is_assistant or not penalty.definition.assistant_only
                    ):
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
                    penalty_value = penalties[penalty.name][i].item()
                    if (
                        penalty_value < 0
                        and message.get(penalty.definition.message_key, False)
                        and (is_assistant or not penalty.definition.assistant_only)
                    ):
                        print(
                            f"Setting negative advantage ({penalty_value:.3f}) for {penalty.name} in assistant message {i} turn {j}",
                            flush=True,
                        )
                        advantages[i, token_offset : token_offset + msg_len].add_(penalty_value)
                token_offset += msg_len
