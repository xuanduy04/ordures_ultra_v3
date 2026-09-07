# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""On-policy distillation (OPD) helpers for async GRPO.

Teacher routing, config helpers, and remote teacher serve resolution.
Advantage computation lives in advantage_estimator.OPDAdvantageEstimator.
IS truncation lives in loss_functions.ClippedPGLoss (ICE-POP mode).
"""

from __future__ import annotations

from typing import Any, NotRequired, Optional, TypedDict

from nemo_rl.algorithms.vllm_teacher_client import (
    normalize_teacher_serve,
    probe_teacher_serve,
)


# ---------------------------------------------------------------------------
# Config TypedDicts
# ---------------------------------------------------------------------------


class GrpoBlendConfig(TypedDict):
    normalize_rewards: NotRequired[bool]
    use_leave_one_out_baseline: NotRequired[bool]


class OnPolicyDistillationConfig(TypedDict):
    enabled: bool
    # Values are `_teachers.<alias>` entries: {url, model} serve specs.
    teacher_model_by_agent_name: NotRequired[dict[str, Any]]
    default_teacher_alias: NotRequired[Optional[str]]
    strict_agent_name_match: NotRequired[bool]
    opd_advantage_weight: NotRequired[float]
    grpo_advantage_weight: NotRequired[float]
    opd_advantage_clip_low: NotRequired[float]
    opd_advantage_clip_high: NotRequired[float]
    grpo_advantage_clip_low: NotRequired[float]
    grpo_advantage_clip_high: NotRequired[float]
    zero_out_of_bounds_advantages: NotRequired[bool]
    grpo: NotRequired[GrpoBlendConfig]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def is_opd_enabled(master_config: dict[str, Any]) -> bool:
    return bool(master_config.get("on_policy_distillation", {}).get("enabled", False))


# ---------------------------------------------------------------------------
# Teacher routing
# ---------------------------------------------------------------------------


def resolve_reference_aliases(
    agent_refs: list[dict],
    teacher_model_by_agent_name: dict[str, Any],
    default_teacher_alias: Optional[str] = None,
    strict_agent_name_match: bool = False,
) -> list[str]:
    aliases: list[str] = []
    for ref in agent_refs:
        name = ref["name"]
        if name in teacher_model_by_agent_name:
            aliases.append(name)
        elif strict_agent_name_match:
            raise ValueError(
                f"No teacher model mapping for agent '{name}'. "
                f"Available: {sorted(teacher_model_by_agent_name.keys())}"
            )
        elif default_teacher_alias:
            print(f"[OPD] Agent '{name}' not in teacher mapping, falling back to '{default_teacher_alias}'")
            aliases.append(default_teacher_alias)
        else:
            raise ValueError(
                f"No teacher model mapping for agent '{name}' and no default_teacher_alias set."
            )
    return aliases


def resolve_teacher_specs(
    aliases: list[str],
    teacher_model_by_agent_name: dict[str, Any],
) -> list[dict[str, str]]:
    """Resolve agent aliases to normalized ``{url, model}`` serve specs."""
    specs: list[dict[str, str]] = []
    for alias in aliases:
        if alias not in teacher_model_by_agent_name:
            raise KeyError(
                f"Agent alias '{alias}' has no teacher serve mapping. "
                f"Available: {sorted(teacher_model_by_agent_name.keys())}"
            )
        specs.append(normalize_teacher_serve(teacher_model_by_agent_name[alias]))
    return specs


def validate_teacher_serves(master_config: dict[str, Any]) -> None:
    """Fail fast if any OPD teacher mapping is not a ``{url, model}`` serve.

    Also probes each unique serve URL (``GET {url}/models``) so a bad URL or
    model name fails at setup, before any rollout is collected.
    """
    opd_cfg = master_config["on_policy_distillation"]
    teacher_model_by_agent_name = opd_cfg["teacher_model_by_agent_name"]
    models_by_url: dict[str, set[str]] = {}
    for raw in teacher_model_by_agent_name.values():
        spec = normalize_teacher_serve(raw)
        models_by_url.setdefault(spec["url"], set()).add(spec["model"])
    for url, models in models_by_url.items():
        probe_teacher_serve(url, models)
