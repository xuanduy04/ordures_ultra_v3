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

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    BaseAdvantageEstimator,
    OPDAdvantageEstimator,
    ReinforceAdvantageEstimator,
)


def _make_estimator(
    opd_advantage_weight=1.0,
    orm_advantage_weight=0.0,
    opd_advantage_clip_low=None,
    opd_advantage_clip_high=None,
    orm_advantage_clip_low=None,
    orm_advantage_clip_high=None,
    zero_out_of_bounds_advantages=False,
    orm_estimator_name=None,
    orm_estimator_kwargs=None,
):
    estimator_config = {
        "opd_advantage_weight": opd_advantage_weight,
        "orm_advantage_weight": orm_advantage_weight,
    }
    if opd_advantage_clip_low is not None:
        estimator_config["opd_advantage_clip_low"] = opd_advantage_clip_low
    if opd_advantage_clip_high is not None:
        estimator_config["opd_advantage_clip_high"] = opd_advantage_clip_high
    if orm_advantage_clip_low is not None:
        estimator_config["orm_advantage_clip_low"] = orm_advantage_clip_low
    if orm_advantage_clip_high is not None:
        estimator_config["orm_advantage_clip_high"] = orm_advantage_clip_high
    if zero_out_of_bounds_advantages:
        estimator_config["zero_out_of_bounds_advantages"] = zero_out_of_bounds_advantages
    if orm_estimator_name is not None:
        estimator_config["orm_advantage_estimator"] = {
            "orm_estimator_name": orm_estimator_name,
            **(orm_estimator_kwargs or {}),
        }
    loss_config = {}
    return OPDAdvantageEstimator(estimator_config, loss_config)


def test_opd_basic_positive_distill_advantage():
    """teacher_lp > student_lp => positive advantages."""
    estimator = _make_estimator()
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)  # log(1) = 0
    student_lp = torch.full((B, S), -1.0)  # lower logprob
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    assert adv.shape == (B, S)
    assert (adv > 0).all(), "teacher_lp > student_lp should yield positive advantages"


def test_opd_teacher_equals_student():
    """Same logprobs => zero advantages."""
    estimator = _make_estimator()
    B, S = 2, 4
    logprobs = torch.randn(B, S)
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=logprobs, prev_logprobs=logprobs
    )

    torch.testing.assert_close(adv, torch.zeros(B, S))


def test_opd_with_orm_advantage():
    """ORM blending with weight=0.5, ORM computed internally (reinforce)."""
    estimator = _make_estimator(
        orm_advantage_weight=0.5, orm_estimator_name="reinforce"
    )
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), -2.0)
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.full((B,), 4.0)  # reinforce: adv = reward = 4.0

    adv = estimator.compute_advantage(
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=teacher_lp,
        prev_logprobs=student_lp,
    )

    # distill = 0 - (-2) = 2.0; orm contribution = 0.5 * 4.0 = 2.0; total = 4.0
    expected = torch.full((B, S), 4.0)
    torch.testing.assert_close(adv, expected)


def test_opd_mask_applied():
    """Masked tokens should have zero advantage."""
    estimator = _make_estimator()
    B, S = 1, 6
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), -1.0)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    # Masked positions must be zero
    assert (adv[:, 3:] == 0).all(), "Masked positions should be zero"
    # Unmasked positions should be positive (teacher > student)
    assert (adv[:, :3] > 0).all(), "Unmasked positions should be positive"


def test_opd_metrics_returned():
    """self.last_metrics should be populated after compute_advantage."""
    estimator = _make_estimator()
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), -1.0)
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    assert "on_policy_distillation/teacher_student_logprob_gap_mean" in estimator.last_metrics
    assert "on_policy_distillation/adv_mean" in estimator.last_metrics
    assert "on_policy_distillation/adv_std" in estimator.last_metrics
    # teacher - student = 0 - (-1) = 1.0
    assert abs(estimator.last_metrics["on_policy_distillation/teacher_student_logprob_gap_mean"] - 1.0) < 1e-5
    assert abs(estimator.last_metrics["on_policy_distillation/adv_mean"] - 1.0) < 1e-5
    assert abs(estimator.last_metrics["on_policy_distillation/adv_std"]) < 1e-5


def test_opd_with_orm_grpo_blending():
    """opd_weight=1.0, orm_weight=1.0 (grpo) => blended advantages."""
    orm_kwargs = {"normalize_rewards": False, "use_leave_one_out_baseline": False}
    estimator = _make_estimator(
        orm_advantage_weight=1.0,
        orm_estimator_name="grpo",
        orm_estimator_kwargs=orm_kwargs,
    )
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), -2.0)
    mask = torch.ones(B, S)
    prompt_ids = torch.tensor([[0], [0]])
    rewards = torch.tensor([1.0, 3.0])

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    # distill = 0 - (-2) = 2.0 per token
    # grpo: baseline = mean([1, 3]) = 2.0, so grpo_adv = [1-2, 3-2] = [-1, 1]
    # expanded to [B, S] gives [[-1,-1,-1,-1], [1,1,1,1]]
    # combined = 1.0 * 2.0 + 1.0 * grpo_adv = [[1,1,1,1], [3,3,3,3]]
    distill = 2.0
    grpo_adv_0 = -1.0
    grpo_adv_1 = 1.0
    expected = torch.tensor([
        [distill + grpo_adv_0] * S,
        [distill + grpo_adv_1] * S,
    ], dtype=torch.float32)
    torch.testing.assert_close(adv, expected)

    assert "on_policy_distillation/grpo_adv_mean" in estimator.last_metrics
    assert "on_policy_distillation/grpo_adv_std" in estimator.last_metrics


def test_opd_orm_weight_zero():
    """orm_advantage_weight=0 gives identical results to pure OPD."""
    orm_kwargs = {"normalize_rewards": False, "use_leave_one_out_baseline": False}
    estimator_blend = _make_estimator(
        orm_advantage_weight=0.0,
        orm_estimator_name="grpo",
        orm_estimator_kwargs=orm_kwargs,
    )
    estimator_pure = _make_estimator(orm_advantage_weight=0.0)

    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), -1.0)
    mask = torch.ones(B, S)
    prompt_ids = torch.tensor([[0], [1]])
    rewards = torch.tensor([0.5, 1.5])

    adv_blend = estimator_blend.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )
    adv_pure = estimator_pure.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    torch.testing.assert_close(adv_blend, adv_pure)


def test_opd_pure_orm_grpo():
    """opd_weight=0, orm_weight=1.0 (grpo) recovers plain GRPO from rewards."""
    orm_kwargs = {"normalize_rewards": False, "use_leave_one_out_baseline": False}
    estimator = _make_estimator(
        opd_advantage_weight=0.0,
        orm_advantage_weight=1.0,
        orm_estimator_name="grpo",
        orm_estimator_kwargs=orm_kwargs,
    )

    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), 0.0)
    mask = torch.ones(B, S)
    prompt_ids = torch.tensor([[0], [0]])
    rewards = torch.tensor([1.0, 5.0])

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    # baseline = mean([1, 5]) = 3, grpo_adv = [-2, 2] expanded to [B, S]
    expected = torch.tensor([
        [-2.0] * S,
        [2.0] * S,
    ], dtype=torch.float32)
    torch.testing.assert_close(adv, expected)


def test_opd_distill_advantage_clip():
    """OPD distill advantages are clamped to opd_advantage_clip bounds."""
    estimator = _make_estimator(opd_advantage_clip_low=-2.0, opd_advantage_clip_high=2.0)
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.full((B, S), -5.0)  # gap = 5.0, outside [low, high]
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    expected = torch.full((B, S), 2.0)
    torch.testing.assert_close(adv, expected)


def test_opd_distill_advantage_clip_low():
    """Negative OPD distill advantages are clamped at the low bound."""
    estimator = _make_estimator(opd_advantage_clip_low=-2.0, opd_advantage_clip_high=2.0)
    B, S = 1, 4
    teacher_lp = torch.full((B, S), -5.0)
    student_lp = torch.zeros(B, S)  # gap = -5.0
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    expected = torch.full((B, S), -2.0)
    torch.testing.assert_close(adv, expected)


def test_opd_orm_blend_advantage_clip():
    """ORM blend advantages are clamped to orm_advantage_clip bounds."""
    orm_kwargs = {"normalize_rewards": False, "use_leave_one_out_baseline": False}
    estimator = _make_estimator(
        opd_advantage_weight=0.0,
        orm_advantage_weight=1.0,
        orm_estimator_name="grpo",
        orm_estimator_kwargs=orm_kwargs,
        orm_advantage_clip_low=-1.0,
        orm_advantage_clip_high=1.0,
    )
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.zeros(B, S)
    mask = torch.ones(B, S)
    prompt_ids = torch.tensor([[0], [0]])
    rewards = torch.tensor([1.0, 5.0])

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    # baseline = mean([1, 5]) = 3, grpo_adv = [-2, 2] -> clamped to [-1, 1]
    expected = torch.tensor([
        [-1.0] * S,
        [1.0] * S,
    ], dtype=torch.float32)
    torch.testing.assert_close(adv, expected)


def test_opd_default_clip_values_are_noop():
    """Default clip bounds (-6767, 6767) leave large advantages unclipped."""
    estimator = _make_estimator()
    B, S = 1, 4
    teacher_lp = torch.full((B, S), 1000.0)
    student_lp = torch.zeros(B, S)  # gap = 1000.0, within default bounds
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    expected = torch.full((B, S), 1000.0)
    torch.testing.assert_close(adv, expected)


def test_opd_nan_teacher_rows_preserve_orm():
    """NaN teacher rows zero the distill contribution; ORM is preserved."""
    orm_kwargs = {"normalize_rewards": False, "use_leave_one_out_baseline": False}
    estimator = _make_estimator(
        orm_advantage_weight=1.0,
        opd_advantage_weight=1.0,
        orm_estimator_name="grpo",
        orm_estimator_kwargs=orm_kwargs,
    )
    B, S = 4, 4
    teacher_lp = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [float("nan"), float("nan"), float("nan"), float("nan")],
            [2.0, 2.0, 2.0, 2.0],
        ]
    )
    student_lp = torch.full((B, S), -2.0)
    mask = torch.ones(B, S)
    prompt_ids = torch.tensor([[0], [0], [0], [0]])
    rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])

    adv = estimator.compute_advantage(
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=teacher_lp,
        prev_logprobs=student_lp,
    )

    # GRPO baseline = mean([1, 3, 5, 7]) = 4.0
    grpo_adv = rewards - 4.0  # [-3, -1, 1, 3]
    # distill = teacher - student = [2, 3, NaN->0, 4]
    distill = torch.tensor([2.0, 3.0, 0.0, 4.0])
    expected = torch.stack(
        [torch.full((S,), float(d) + float(g)) for d, g in zip(distill, grpo_adv)]
    )

    torch.testing.assert_close(adv, expected)
    assert not torch.isnan(adv).any()

    # ORM contribution fully preserved on the NaN row:
    # adv = 0 (distill) + grpo_adv = 1.0
    assert torch.allclose(adv[2], torch.full((S,), 1.0))

    # Finite rows match the all-finite baseline exactly.
    baseline = estimator.compute_advantage(
        prompt_ids,
        rewards,
        mask,
        teacher_logprobs=torch.nan_to_num(teacher_lp, nan=0.0),
        prev_logprobs=student_lp,
    )
    finite_rows = torch.tensor([True, True, False, True])
    torch.testing.assert_close(adv[finite_rows], baseline[finite_rows])


def test_opd_zero_out_of_bounds_distill_advantages():
    """zero_out_of_bounds_advantages: OPD distill values outside bounds become 0."""
    estimator = _make_estimator(
        opd_advantage_clip_low=-2.0,
        opd_advantage_clip_high=2.0,
        zero_out_of_bounds_advantages=True,
    )
    B, S = 1, 6
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0, 2.0]])  # gaps: 3,1,0,-1,-3,-2
    mask = torch.ones(B, S)
    prompt_ids = torch.arange(B)
    rewards = torch.zeros(B)

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    # out-of-bounds gaps (3, -3) are zeroed; in-bounds values kept as-is
    expected = torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0, -2.0]])
    torch.testing.assert_close(adv, expected)


def test_opd_zero_out_of_bounds_orm_advantages():
    """zero_out_of_bounds_advantages: ORM blend values outside bounds become 0."""
    orm_kwargs = {"normalize_rewards": False, "use_leave_one_out_baseline": False}
    estimator = _make_estimator(
        opd_advantage_weight=0.0,
        orm_advantage_weight=1.0,
        orm_estimator_name="grpo",
        orm_estimator_kwargs=orm_kwargs,
        orm_advantage_clip_low=-1.0,
        orm_advantage_clip_high=1.0,
        zero_out_of_bounds_advantages=True,
    )
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.zeros(B, S)
    mask = torch.ones(B, S)
    prompt_ids = torch.tensor([[0], [0]])
    rewards = torch.tensor([1.0, 5.0])

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    # baseline = mean([1, 5]) = 3, grpo_adv = [-2, 2] -> both zeroed
    expected = torch.zeros(B, S)
    torch.testing.assert_close(adv, expected)


def test_opd_zero_out_of_bounds_orm_keeps_in_bounds():
    """zero_out_of_bounds_advantages: in-bounds ORM blend values are kept."""
    orm_kwargs = {"normalize_rewards": False, "use_leave_one_out_baseline": False}
    estimator = _make_estimator(
        opd_advantage_weight=0.0,
        orm_advantage_weight=1.0,
        orm_estimator_name="grpo",
        orm_estimator_kwargs=orm_kwargs,
        orm_advantage_clip_low=-1.0,
        orm_advantage_clip_high=1.0,
        zero_out_of_bounds_advantages=True,
    )
    B, S = 2, 4
    teacher_lp = torch.zeros(B, S)
    student_lp = torch.zeros(B, S)
    mask = torch.ones(B, S)
    prompt_ids = torch.tensor([[0], [0]])
    rewards = torch.tensor([2.0, 3.0])

    adv = estimator.compute_advantage(
        prompt_ids, rewards, mask, teacher_logprobs=teacher_lp, prev_logprobs=student_lp
    )

    # baseline = mean([2, 3]) = 2.5, grpo_adv = [-0.5, 0.5] -> within bounds, kept
    expected = torch.tensor([
        [-0.5] * S,
        [0.5] * S,
    ], dtype=torch.float32)
    torch.testing.assert_close(adv, expected)


def test_opd_is_base_advantage_estimator():
    """OPDAdvantageEstimator must be a BaseAdvantageEstimator subclass."""
    estimator = _make_estimator()
    assert isinstance(estimator, BaseAdvantageEstimator)


def test_opd_requires_teacher_logprobs():
    """OPD raises when teacher_logprobs is missing."""
    estimator = _make_estimator()
    with pytest.raises(ValueError, match="teacher_logprobs"):
        estimator.compute_advantage(
            torch.arange(2), torch.zeros(2), torch.ones(2, 4), prev_logprobs=torch.zeros(2, 4)
        )


def test_opd_requires_prev_logprobs():
    """OPD raises when prev_logprobs is missing."""
    estimator = _make_estimator()
    with pytest.raises(ValueError, match="prev_logprobs"):
        estimator.compute_advantage(
            torch.arange(2), torch.zeros(2), torch.ones(2, 4), teacher_logprobs=torch.zeros(2, 4)
        )


def test_opd_unsupported_orm_estimator_raises():
    """Unsupported ORM estimator names raise ValueError at construction."""
    with pytest.raises(ValueError, match="Unsupported ORM advantage estimator"):
        _make_estimator(orm_advantage_weight=1.0, orm_estimator_name="unknown")


def test_opd_orm_weight_without_estimator_config_raises():
    """orm_advantage_weight > 0 without orm_advantage_estimator raises ValueError."""
    with pytest.raises(ValueError, match="orm_advantage_estimator"):
        _make_estimator(orm_advantage_weight=1.0, orm_estimator_name=None)


def _make_reinforce_estimator(minus_baseline=False):
    return ReinforceAdvantageEstimator({"minus_baseline": minus_baseline})


def test_reinforce_no_baseline():
    """REINFORCE with minus_baseline=False returns raw rewards per token."""
    estimator = _make_reinforce_estimator(minus_baseline=False)
    B, S = 2, 4
    prompt_ids = torch.tensor([[0], [1]])
    rewards = torch.tensor([1.0, 3.0])
    mask = torch.ones(B, S)

    adv = estimator.compute_advantage(prompt_ids, rewards, mask)

    expected = torch.tensor([
        [1.0] * S,
        [3.0] * S,
    ], dtype=torch.float32)
    torch.testing.assert_close(adv, expected)


def test_reinforce_minus_baseline():
    """REINFORCE with minus_baseline=True subtracts the per-prompt mean."""
    estimator = _make_reinforce_estimator(minus_baseline=True)
    B, S = 2, 4
    prompt_ids = torch.tensor([[0], [0]])
    rewards = torch.tensor([1.0, 3.0])
    mask = torch.ones(B, S)

    adv = estimator.compute_advantage(prompt_ids, rewards, mask)

    # baseline = mean([1, 3]) = 2.0 -> adv = [-1, 1]
    expected = torch.tensor([
        [-1.0] * S,
        [1.0] * S,
    ], dtype=torch.float32)
    torch.testing.assert_close(adv, expected)


def test_reinforce_mask_applied():
    """REINFORCE advantages are zero on masked tokens."""
    estimator = _make_reinforce_estimator(minus_baseline=False)
    B, S = 1, 4
    prompt_ids = torch.arange(B)
    rewards = torch.tensor([2.0])
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)

    adv = estimator.compute_advantage(prompt_ids, rewards, mask)

    assert (adv[:, 2:] == 0).all(), "Masked positions should be zero"
    assert (adv[:, :2] == 2.0).all(), "Unmasked positions should equal the reward"
