from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# A rule receives the raw NeMo-Gym dataset row (including ``agent_ref``) and returns True when the rollout's
# environment + row state expects the model to answer with a (structured) tool call.
@dataclass(frozen=True, kw_only=True)
class ToolCallExpectationRule:
    agent_ref_name: str
    answer_is_tool_call_field: Optional[str]


TOOL_CALL_EXPECTATION_RULES: tuple[ToolCallExpectationRule, ...] = (
    ToolCallExpectationRule(agent_ref_name="single_step_tool_use_with_argument_comparison", answer_is_tool_call_field=None),
    ToolCallExpectationRule(agent_ref_name="general_qa", answer_is_tool_call_field="answer_is_tool_call"),
)  # fmt: skip


def _matches_tool_call_expectation_rule(nemo_gym_row: dict[str, Any], rule: ToolCallExpectationRule) -> bool:
    return nemo_gym_row["agent_ref"]["name"].strip().lower().startswith(rule.agent_ref_name) and (
        nemo_gym_row.get(rule.answer_is_tool_call_field, False) if rule.answer_is_tool_call_field is not None else True
    )


def expects_tool_call(nemo_gym_row: dict[str, Any]) -> bool:
    """Return whether the rollout of this NeMo-Gym row is expected to answer with a tool call.

    Args:
        nemo_gym_row: Raw NeMo-Gym dataset row (including ``agent_ref`` and other metadata keys).

    Returns:
        True if the row matches any rule in TOOL_CALL_EXPECTATION_RULES.
    """
    return any(_matches_tool_call_expectation_rule(nemo_gym_row, rule) for rule in TOOL_CALL_EXPECTATION_RULES)
