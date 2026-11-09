from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# A rule receives the raw NeMo-Gym dataset row (including ``agent_ref``) and returns True when the rollout's
# environment + row state expects the model to answer with a (structured) tool call.
@dataclass(frozen=True, kw_only=True)
class ToolCallExpectationRule:
    agent_ref_name: str
    # Path of keys into the NeMo-Gym row; None = every row of this env expects a tool call.
    field_path: Optional[tuple[str, ...]]
    # Value the row field at ``field_path`` must equal for the row to expect a tool call.
    expected_value: Any


TOOL_CALL_EXPECTATION_RULES: tuple[ToolCallExpectationRule, ...] = (
    ToolCallExpectationRule(agent_ref_name="single_step_tool_use_with_argument_comparison", field_path=("expected_action", "type"), expected_value="function_call"),
    ToolCallExpectationRule(agent_ref_name="swe_pivot_single_step_tool_use_with_argument_comparison", field_path=("expected_action", "type"), expected_value="function_call"),
    ToolCallExpectationRule(agent_ref_name="toolcall_schema_single_step_tool_use_with_argument_comparison", field_path=("expected_action", "type"), expected_value="function_call"),
    ToolCallExpectationRule(agent_ref_name="general_qa", field_path=("answer_is_tool_call",), expected_value=True),
)  # fmt: skip


def _matches_tool_call_expectation_rule(nemo_gym_row: dict[str, Any], rule: ToolCallExpectationRule) -> bool:
    if not nemo_gym_row["agent_ref"]["name"].strip().lower().startswith(rule.agent_ref_name):
        return False
    if rule.field_path is None:
        return True
    value: Any = nemo_gym_row
    for key in rule.field_path:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return value == rule.expected_value


def expects_tool_call(nemo_gym_row: dict[str, Any]) -> bool:
    """Return whether the rollout of this NeMo-Gym row is expected to answer with a tool call.

    Args:
        nemo_gym_row: Raw NeMo-Gym dataset row (including ``agent_ref`` and other metadata keys).

    Returns:
        True if the row matches any rule in TOOL_CALL_EXPECTATION_RULES.
    """
    return any(_matches_tool_call_expectation_rule(nemo_gym_row, rule) for rule in TOOL_CALL_EXPECTATION_RULES)
