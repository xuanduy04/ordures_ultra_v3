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

import asyncio

import pytest
import torch

from nemo_rl.algorithms.vllm_teacher_client import TeacherContextLengthError
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


# ---------------------------------------------------------------------------
# Mock remote teacher client for _compute_teacher_logprobs tests
# ---------------------------------------------------------------------------


class _FakeTeacherClient:
    """Returns a constant-filled logprobs tensor; records the served teacher."""

    def __init__(self, fill_value=2.0):
        self._fill_value = fill_value
        self.last_teacher = None
        self.last_input_lengths = None

    async def score_group(self, teacher, input_ids, input_lengths=None):
        self.last_teacher = teacher
        self.last_input_lengths = input_lengths
        B, S = input_ids.shape
        return torch.full((B, S), self._fill_value)


class _ContextLengthFailingTeacherClient:
    """Raises TeacherContextLengthError on every score_group call."""

    async def score_group(self, teacher, input_ids, input_lengths=None):
        raise TeacherContextLengthError(
            teacher["url"], "This model's maximum context length is 4096 tokens"
        )


def _make_collector(**overrides):
    """Build a bare AsyncTrajectoryCollector (bypass Ray) for unit testing."""
    from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector

    collector_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    defaults = {
        "on_policy_distillation_cfg": {},
        "_has_opd_teachers": False,
        "_teacher_logprob_client": None,
    }
    defaults.update(overrides)
    obj = object.__new__(collector_cls)
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


_TEACHER_SERVES = {
    "math_agent": {"url": "http://t-math:8000/v1", "model": "teacher-math"},
    "code_agent": {"url": "http://t-code:8000/v1", "model": "teacher-code"},
}


# ---------------------------------------------------------------------------
# _compute_teacher_logprobs (remote vLLM serve path)
# ---------------------------------------------------------------------------


def test_compute_teacher_logprobs_remote():
    """A prompt group is scored on the teacher serve for its agent."""
    client = _FakeTeacherClient(fill_value=2.0)
    collector = _make_collector(
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": _TEACHER_SERVES,
        },
        _has_opd_teachers=True,
        _teacher_logprob_client=client,
    )

    B, S = 4, 16
    input_ids = torch.randint(0, 100, (B, S))
    agent_refs = [{"name": "math_agent"}] * B

    result, total_time = asyncio.run(
        collector._compute_teacher_logprobs(input_ids, agent_refs)
    )

    assert result.shape == (B, S)
    assert torch.allclose(result, torch.tensor(2.0))
    assert client.last_teacher == _TEACHER_SERVES["math_agent"]
    assert total_time >= 0.0


def test_compute_teacher_logprobs_passes_lengths():
    """Per-sample lengths are forwarded to the client."""
    client = _FakeTeacherClient(fill_value=1.0)
    collector = _make_collector(
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": _TEACHER_SERVES,
        },
        _has_opd_teachers=True,
        _teacher_logprob_client=client,
    )

    B, S = 2, 8
    input_ids = torch.randint(0, 100, (B, S))
    lengths = torch.tensor([5, 8])
    agent_refs = [{"name": "code_agent"}] * B

    asyncio.run(
        collector._compute_teacher_logprobs(input_ids, agent_refs, lengths)
    )

    assert client.last_teacher == _TEACHER_SERVES["code_agent"]
    assert torch.equal(client.last_input_lengths, lengths)


def test_compute_teacher_logprobs_default_alias_fallback():
    """Unmapped agents fall back to the default_teacher_alias serve."""
    client = _FakeTeacherClient(fill_value=3.0)
    collector = _make_collector(
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": _TEACHER_SERVES,
            "default_teacher_alias": "code_agent",
        },
        _has_opd_teachers=True,
        _teacher_logprob_client=client,
    )

    input_ids = torch.randint(0, 100, (1, 4))
    agent_refs = [{"name": "unknown_agent"}]

    asyncio.run(collector._compute_teacher_logprobs(input_ids, agent_refs))

    assert client.last_teacher == _TEACHER_SERVES["code_agent"]


def test_compute_teacher_logprobs_unmapped_alias_raises():
    """A fallback alias with no mapping entry raises a clear KeyError."""
    collector = _make_collector(
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": _TEACHER_SERVES,
            "default_teacher_alias": "missing_alias",
        },
        _has_opd_teachers=True,
        _teacher_logprob_client=_FakeTeacherClient(),
    )

    input_ids = torch.randint(0, 100, (1, 4))
    agent_refs = [{"name": "unknown_agent"}]

    with pytest.raises(KeyError, match="no teacher serve mapping"):
        asyncio.run(collector._compute_teacher_logprobs(input_ids, agent_refs))


def test_compute_teacher_logprobs_context_length_returns_nan():
    """Context-length overflow degrades to an all-NaN tensor instead of raising."""
    collector = _make_collector(
        on_policy_distillation_cfg={
            "teacher_model_by_agent_name": _TEACHER_SERVES,
        },
        _has_opd_teachers=True,
        _teacher_logprob_client=_ContextLengthFailingTeacherClient(),
    )

    B, S = 2, 8
    input_ids = torch.randint(0, 100, (B, S))
    agent_refs = [{"name": "math_agent"}] * B

    result, total_time = asyncio.run(
        collector._compute_teacher_logprobs(input_ids, agent_refs)
    )

    assert result.shape == (B, S)
    assert torch.isnan(result).all()
    assert total_time >= 0.0


# ---------------------------------------------------------------------------
# Serve spec resolution / validation
# ---------------------------------------------------------------------------


def test_resolve_teacher_specs_routes_and_normalizes():
    from nemo_rl.algorithms.opd import resolve_teacher_specs
    specs = resolve_teacher_specs(["math_agent", "code_agent"], _TEACHER_SERVES)
    assert specs == [
        _TEACHER_SERVES["math_agent"],
        _TEACHER_SERVES["code_agent"],
    ]

    stripped = resolve_teacher_specs(
        ["math_agent"], {"math_agent": {"url": " http://x/v1 ", "model": " m "}}
    )
    assert stripped == [{"url": "http://x/v1", "model": "m"}]


def test_resolve_teacher_specs_unknown_alias():
    from nemo_rl.algorithms.opd import resolve_teacher_specs
    with pytest.raises(KeyError, match="no teacher serve mapping"):
        resolve_teacher_specs(["nope"], _TEACHER_SERVES)


def test_validate_teacher_serves_valid(monkeypatch):
    from nemo_rl.algorithms.opd import validate_teacher_serves

    monkeypatch.setattr(
        "nemo_rl.algorithms.opd.probe_teacher_serve", lambda url, models: None
    )
    validate_teacher_serves(
        {"on_policy_distillation": {"teacher_model_by_agent_name": _TEACHER_SERVES}}
    )


def test_validate_teacher_serves_probe_failure(monkeypatch):
    """A failing serve probe surfaces as RuntimeError at setup time."""
    from nemo_rl.algorithms.opd import validate_teacher_serves

    def _boom(url, models):
        raise RuntimeError(f"probe failed for {url}")

    monkeypatch.setattr("nemo_rl.algorithms.opd.probe_teacher_serve", _boom)
    with pytest.raises(RuntimeError, match="probe failed"):
        validate_teacher_serves(
            {"on_policy_distillation": {"teacher_model_by_agent_name": _TEACHER_SERVES}}
        )


def test_validate_teacher_serves_missing_fields():
    from nemo_rl.algorithms.opd import validate_teacher_serves
    with pytest.raises(ValueError, match="'url' and 'model'"):
        validate_teacher_serves(
            {
                "on_policy_distillation": {
                    "teacher_model_by_agent_name": {"math": "/ckpt/math"}
                }
            }
        )


def test_validate_teacher_serves_non_string_fields():
    from nemo_rl.algorithms.opd import validate_teacher_serves
    with pytest.raises(ValueError, match="'url'"):
        validate_teacher_serves(
            {
                "on_policy_distillation": {
                    "teacher_model_by_agent_name": {
                        "math": {"url": None, "model": "m"}
                    }
                }
            }
        )


# ---------------------------------------------------------------------------
# Unsort / reorder_data regression tests
# ---------------------------------------------------------------------------


def test_reorder_data_vs_direct_gather():
    """Verify reorder_data inverts the permutation, while direct gather does not.

    shard_by_batch_size returns a forward permutation (sorted_pos → orig_idx).
    To restore original order we need the *inverse* (argsort), which
    reorder_data computes.  A direct gather ``result[indices]`` applies
    the forward permutation and silently produces wrong results.
    """
    # Simulate: 4 samples reordered by sequence packing as [3, 0, 2, 1]
    forward_perm = [3, 0, 2, 1]
    # After inference, results are in sorted order:
    #   position 0 = result for orig sample 3
    #   position 1 = result for orig sample 0  etc.
    sorted_results = BatchedDataDict(
        {"logprobs": torch.tensor([[30.0], [0.0], [20.0], [10.0]])}
    )
    # label: sorted_results[i] holds the value for original sample forward_perm[i]
    #   sorted_results[0]=30 → orig 3,  sorted_results[1]=0 → orig 0, etc.

    # --- WRONG: direct gather (the old bug) ---
    wrong = sorted_results["logprobs"][forward_perm]
    # wrong[0] = sorted_results[3] = 10  (should be 0 for orig 0)
    assert not torch.equal(wrong, torch.tensor([[0.0], [10.0], [20.0], [30.0]])), \
        "Direct gather should NOT produce the correct original order"

    # --- CORRECT: reorder_data (inverse permutation) ---
    correct = BatchedDataDict({"logprobs": sorted_results["logprobs"].clone()})
    correct.reorder_data(forward_perm)
    assert torch.equal(correct["logprobs"], torch.tensor([[0.0], [10.0], [20.0], [30.0]])), \
        "reorder_data should restore the original sample order"


def test_reorder_data_inverse_permutation_various():
    """reorder_data correctly inverts arbitrary permutations, including identity."""
    # Identity permutation
    bdd = BatchedDataDict({"x": torch.tensor([[0.0], [1.0], [2.0]])})
    bdd.reorder_data([0, 1, 2])
    assert torch.equal(bdd["x"], torch.tensor([[0.0], [1.0], [2.0]]))

    # Reversal
    bdd = BatchedDataDict({"x": torch.tensor([[0.0], [1.0], [2.0]])})
    bdd.reorder_data([2, 1, 0])
    # batch_sorted_indices=[2,1,0] means sorted[0] came from orig 2, etc.
    # Inverse: orig[2]=sorted[0]=0.0, orig[1]=sorted[1]=1.0, orig[0]=sorted[2]=2.0
    assert torch.equal(bdd["x"], torch.tensor([[2.0], [1.0], [0.0]]))

    # Non-trivial: simulate 4 samples reordered as [2, 3, 0, 1]
    bdd = BatchedDataDict({"x": torch.tensor([[20.0], [30.0], [0.0], [10.0]])})
    bdd.reorder_data([2, 3, 0, 1])
    assert torch.equal(bdd["x"], torch.tensor([[0.0], [10.0], [20.0], [30.0]])),  \
        "After reorder_data, row i should hold the result for original sample i"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def test_is_opd_enabled():
    from nemo_rl.algorithms.opd import is_opd_enabled
    assert is_opd_enabled({"on_policy_distillation": {"enabled": True}})
    assert not is_opd_enabled({"on_policy_distillation": {"enabled": False}})
    assert not is_opd_enabled({})


def test_resolve_reference_aliases_bad_agent_ref():
    from nemo_rl.algorithms.opd import resolve_reference_aliases
    with pytest.raises(KeyError):
        resolve_reference_aliases(
            [{"not_name": "oops"}], {"math": _TEACHER_SERVES["math_agent"]}
        )


def test_resolve_reference_aliases_fallback():
    from nemo_rl.algorithms.opd import resolve_reference_aliases
    aliases = resolve_reference_aliases(
        [{"name": "math_agent"}, {"name": "unknown"}, {"name": "code_agent"}],
        _TEACHER_SERVES,
        default_teacher_alias="math_agent",
    )
    assert aliases == ["math_agent", "math_agent", "code_agent"]


def test_resolve_reference_aliases_strict_raises():
    from nemo_rl.algorithms.opd import resolve_reference_aliases
    with pytest.raises(ValueError, match="No teacher model mapping"):
        resolve_reference_aliases(
            [{"name": "unknown"}], _TEACHER_SERVES, strict_agent_name_match=True
        )
