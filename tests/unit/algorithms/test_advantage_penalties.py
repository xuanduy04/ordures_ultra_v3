import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import (
    ADVANTAGE_ESTIMATOR_MAPPING,
    GRPOAdvantageEstimator,
    ReinforcePlusPlusAdvantageEstimator,
)
from nemo_rl.algorithms.advantage_penalties import (
    FlagAdvantagePenalty,
    apply_advantage_penalties,
    create_advantage_penalties,
    normalize_penalty_signal,
)
from nemo_rl.algorithms.advantage_penalties.base_advantage_penalty import (
    DEFAULT_ADVANTAGE_PENALTY_WEIGHT,
    AdvantagePenaltySpec,
    BaseAdvantagePenaltySettings,
)
from nemo_rl.algorithms.advantage_penalties.flag_advantage_penalty import FlagAdvantagePenaltyDefinition


def _make_normalizer(name: str):
    """Build a normalizer estimator matching the penalty-normalizer defaults."""
    configs = {
        "reinforce_plus_plus": {"name": "reinforce_plus_plus", "minus_baseline": False},
        "grpo": {"name": "grpo", "normalize_rewards": True, "use_leave_one_out_baseline": False},
        "reinforce": {"name": "reinforce"},
    }
    loss_config = {"use_kl_in_reward": False, "reference_policy_kl_penalty": 0.0, "reference_policy_kl_type": "k3"}
    return ADVANTAGE_ESTIMATOR_MAPPING[name](configs[name], loss_config)


def _make_message(role, n_tokens, flag_keys, generated):
    message = {"role": role, "token_ids": torch.arange(n_tokens, dtype=torch.long)}
    if generated:
        message["generation_logprobs"] = torch.zeros(n_tokens)
    for key in flag_keys:
        message[key] = True
    return message


def _make_flag_spec(
    name, config_key, message_key, assistant_only, weight=1.0, normalizer=None, max_advantage_penalty=6767
):
    return AdvantagePenaltySpec(
        definition=FlagAdvantagePenaltyDefinition(
            name=name, config_key=config_key, message_key=message_key, assistant_only=assistant_only
        ),
        settings=BaseAdvantagePenaltySettings(
            weight=weight, normalizer=normalizer, max_advantage_penalty=max_advantage_penalty
        ),
    )


def test_normalize_penalty_signal_matches_old_batch_formula():
    """Mixed flags: flagged -> -1/(2*std), clean -> 0 (the original GDPO math)."""
    normalizer = _make_normalizer("reinforce_plus_plus")
    flags = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)

    penalties = normalize_penalty_signal(normalizer, None, flags, 6767)

    std = flags.std(unbiased=False)
    expected = (-flags / (2 * std)).clamp(min=-6767)
    torch.testing.assert_close(penalties, expected.float())


def test_normalize_penalty_signal_degenerate_signals_are_zero():
    """Nobody flagged or everybody flagged: no signal to learn from."""
    normalizer = _make_normalizer("reinforce_plus_plus")
    for flags in (torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)):
        penalties = normalize_penalty_signal(normalizer, None, flags, 6767)
        torch.testing.assert_close(penalties, torch.zeros(4))


def test_normalize_penalty_signal_floor_clamp():
    normalizer = _make_normalizer("reinforce_plus_plus")
    flags = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)

    penalties = normalize_penalty_signal(normalizer, None, flags, 0.25)

    assert (penalties >= -0.25).all()
    torch.testing.assert_close(penalties[flags == 1.0], torch.full((3,), -0.25))
    torch.testing.assert_close(penalties[flags == 0.0], torch.zeros(1))


def test_normalize_penalty_signal_grpo_group_norm_with_per_group_no_bonus():
    """GRPO normalizes per prompt group; the no-bonus max is per-group."""
    normalizer = _make_normalizer("grpo")
    prompt_ids = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2]])
    flags = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)

    penalties = normalize_penalty_signal(normalizer, prompt_ids, flags, 6767)

    r = -flags
    a = torch.zeros(4)
    group1 = r[:2]
    a[:2] = (group1 - group1.mean()) / group1.std(unbiased=True)
    group_max = torch.zeros(4)
    group_max[:2] = a[:2].max()
    expected = (a - group_max) / 2
    torch.testing.assert_close(penalties, expected.float())


def test_normalize_penalty_signal_grpo_per_group_max_is_not_batch_max():
    """Each group's best rollout gets 0, even when another group peaks higher."""
    normalizer = _make_normalizer("grpo")
    prompt_ids = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2]])
    flags = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float64)

    penalties = normalize_penalty_signal(normalizer, prompt_ids, flags, 6767)

    r = -flags
    a = torch.zeros(6)
    group1 = r[:2]
    a[:2] = (group1 - group1.mean()) / group1.std(unbiased=True)
    group2 = r[2:]
    a[2:] = (group2 - group2.mean()) / group2.std(unbiased=True)
    group_max = torch.zeros(6)
    group_max[:2] = a[:2].max()
    group_max[2:] = a[2:].max()
    expected = (a - group_max) / 2
    torch.testing.assert_close(penalties, expected.float())
    # The old batch-wide max would have pushed group 1's best rollout negative.
    batch_max_expected = (a - a.max()) / 2
    assert abs(batch_max_expected[1].item()) > 0.1


def test_normalize_penalty_signal_batch_normalizer_ignores_prompt_ids():
    """Batch-level schemes keep the batch-wide max even when prompt_ids is given."""
    normalizer = _make_normalizer("reinforce_plus_plus")
    prompt_ids = torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2]])
    flags = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)

    with_ids = normalize_penalty_signal(normalizer, prompt_ids, flags, 6767)
    without_ids = normalize_penalty_signal(normalizer, None, flags, 6767)
    torch.testing.assert_close(with_ids, without_ids)


def test_normalizer_group_normalization_flag():
    assert _make_normalizer("grpo").is_group_normalization
    assert not _make_normalizer("reinforce_plus_plus").is_group_normalization
    assert not _make_normalizer("reinforce").is_group_normalization


def test_reinforce_group_normalization_flag_reflects_baseline_mode():
    loss_config = {"use_kl_in_reward": False, "reference_policy_kl_penalty": 0.0, "reference_policy_kl_type": "k3"}
    per_prompt = ADVANTAGE_ESTIMATOR_MAPPING["reinforce"](
        {"name": "reinforce", "minus_baseline": True, "batch_level_baseline": False}, loss_config
    )
    assert per_prompt.is_group_normalization
    batch_level = ADVANTAGE_ESTIMATOR_MAPPING["reinforce"](
        {"name": "reinforce", "minus_baseline": True, "batch_level_baseline": True}, loss_config
    )
    assert not batch_level.is_group_normalization


def test_flag_compute_signal_assistant_only_and_any_role():
    penalty = FlagAdvantagePenalty(
        penalties=[
            _make_flag_spec("any", "any", "flag_any", assistant_only=False),
            _make_flag_spec("asst", "asst", "flag_asst", assistant_only=True),
        ]
    )
    batch = {
        "message_log": [
            [
                _make_message("user", 3, ["flag_any", "flag_asst"], generated=False),
                _make_message("assistant", 4, ["flag_asst"], generated=True),
            ],
            [
                _make_message("user", 3, ["flag_any"], generated=False),
                _make_message("assistant", 4, [], generated=True),
            ],
        ]
    }

    signals = penalty.compute_signal(batch)

    torch.testing.assert_close(signals["any"], torch.tensor([1.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(signals["asst"], torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_flag_apply_adds_penalty_to_offending_message_only():
    configs = [_make_flag_spec("c", "c", "flag", assistant_only=False, weight=2.0)]
    penalty = FlagAdvantagePenalty(penalties=configs)
    batch = {
        "message_log": [
            [_make_message("user", 3, [], generated=False), _make_message("assistant", 4, ["flag"], generated=True)],
            [_make_message("user", 3, [], generated=False), _make_message("assistant", 4, [], generated=True)],
        ]
    }
    advantages = torch.zeros(2, 7)
    penalties = {"c": torch.tensor([-0.5, 0.0], dtype=torch.float64)}

    penalty.apply(advantages, batch, penalties)

    # apply() receives already-weighted penalties, so settings.weight is not applied again.
    expected = torch.zeros(2, 7)
    expected[0, 3:7] = -0.5
    torch.testing.assert_close(advantages, expected)


def test_flag_apply_skips_zero_penalties():
    configs = [_make_flag_spec("c", "c", "flag", assistant_only=False, weight=1.0)]
    penalty = FlagAdvantagePenalty(penalties=configs)
    batch = {"message_log": [[_make_message("assistant", 4, ["flag"], generated=True)]]}
    advantages = torch.zeros(1, 4)
    penalties = {"c": torch.tensor([0.0], dtype=torch.float64)}

    penalty.apply(advantages, batch, penalties)

    torch.testing.assert_close(advantages, torch.zeros(1, 4))


def test_apply_advantage_penalties_matches_old_inline_code():
    flags = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    batch = {
        "message_log": [
            [
                _make_message("user", 2, [], generated=False),
                _make_message("assistant", 3, ["flag"] if flags[i] else [], generated=True),
            ]
            for i in range(4)
        ]
    }
    penalties = [
        FlagAdvantagePenalty(
            penalties=[
                _make_flag_spec(
                    "c",
                    "c",
                    "flag",
                    assistant_only=False,
                    weight=0.5,
                    normalizer=_make_normalizer("reinforce_plus_plus"),
                    max_advantage_penalty=6767,
                )
            ]
        )
    ]
    advantages = torch.zeros(4, 5)

    adv_penalties = apply_advantage_penalties(advantages, batch, penalties)

    std = flags.std(unbiased=False)
    old = 0.5 * (-flags / (2 * std)).clamp(min=-6767)
    # The logged penalties equal exactly what is added to the advantages (post-weight).
    torch.testing.assert_close(adv_penalties["c"], old.float())
    for i in range(4):
        expected_row = torch.zeros(5)
        if flags[i]:
            expected_row[2:5] = old[i]
        torch.testing.assert_close(advantages[i], expected_row)


def test_create_advantage_penalties_empty_when_disabled():
    loss_config = {"use_kl_in_reward": False, "reference_policy_kl_penalty": 0.0, "reference_policy_kl_type": "k3"}
    assert create_advantage_penalties({}, loss_config) == []


def test_create_advantage_penalties_flag_configs():
    loss_config = {"use_kl_in_reward": False, "reference_policy_kl_penalty": 0.0, "reference_policy_kl_type": "k3"}
    grpo_config = {
        "advantage_penalty": {
            "invalid_tool_call": {"enabled": True, "weight": 0.3},
            "eos_token": {"enabled": True},
            "empty_final_answer": {"enabled": True},
        }
    }

    penalties = create_advantage_penalties(grpo_config, loss_config)

    assert len(penalties) == 1
    flag = penalties[0]
    assert isinstance(flag, FlagAdvantagePenalty)
    configs = {c.name: c for c in flag.penalties}
    assert set(configs) == {"invalid_tool_call", "generate_eos_token", "empty_final_answer"}
    assert configs["invalid_tool_call"].settings.weight == 0.3
    assert configs["generate_eos_token"].settings.weight == DEFAULT_ADVANTAGE_PENALTY_WEIGHT
    assert configs["empty_final_answer"].settings.weight == DEFAULT_ADVANTAGE_PENALTY_WEIGHT


def test_create_advantage_penalties_default_weight_fallback():
    loss_config = {"use_kl_in_reward": False, "reference_policy_kl_penalty": 0.0, "reference_policy_kl_type": "k3"}
    grpo_config = {"advantage_penalty": {"empty_final_answer": {"enabled": True}, "default_weight": 0.1}}

    penalties = create_advantage_penalties(grpo_config, loss_config)

    assert penalties[0].penalties[0].settings.weight == 0.1


def test_create_advantage_penalties_grpo_normalizer():
    loss_config = {"use_kl_in_reward": False, "reference_policy_kl_penalty": 0.0, "reference_policy_kl_type": "k3"}
    grpo_config = {
        "advantage_penalty": {"invalid_tool_call": {"enabled": True}, "default_advantage_penalty_normalizer": "grpo"}
    }

    penalties = create_advantage_penalties(grpo_config, loss_config)

    assert isinstance(penalties[0].penalties[0].settings.normalizer, GRPOAdvantageEstimator)


def test_create_advantage_penalties_per_penalty_normalizer_override():
    loss_config = {"use_kl_in_reward": False, "reference_policy_kl_penalty": 0.0, "reference_policy_kl_type": "k3"}
    grpo_config = {
        "advantage_penalty": {
            "invalid_tool_call": {"enabled": True, "advantage_penalty_normalizer": "reinforce_plus_plus"},
            "malformed_thinking": {"enabled": True},
            "default_advantage_penalty_normalizer": "grpo",
        }
    }

    penalties = create_advantage_penalties(grpo_config, loss_config)

    assert len(penalties) == 1
    normalizers_by_penalty = {spec.name: type(spec.settings.normalizer) for spec in penalties[0].penalties}
    assert normalizers_by_penalty["invalid_tool_call"] is ReinforcePlusPlusAdvantageEstimator
    assert normalizers_by_penalty["malformed_thinking"] is GRPOAdvantageEstimator


def test_create_advantage_penalties_per_penalty_max_override():
    loss_config = {"use_kl_in_reward": False, "reference_policy_kl_penalty": 0.0, "reference_policy_kl_type": "k3"}
    grpo_config = {
        "advantage_penalty": {
            "invalid_tool_call": {"enabled": True, "max_advantage_penalty": 200},
            "empty_final_answer": {"enabled": True},
            "default_max_advantage_penalty": -1.67,
        }
    }

    penalties = create_advantage_penalties(grpo_config, loss_config)

    assert len(penalties) == 1
    max_by_penalty = {spec.name: spec.settings.max_advantage_penalty for spec in penalties[0].penalties}
    assert max_by_penalty["invalid_tool_call"] == 200
    assert max_by_penalty["empty_final_answer"] == 1.67


def test_create_advantage_penalties_rejects_unknown_normalizer():
    loss_config = {"use_kl_in_reward": False, "reference_policy_kl_penalty": 0.0, "reference_policy_kl_type": "k3"}
    grpo_config = {
        "advantage_penalty": {"invalid_tool_call": {"enabled": True}, "default_advantage_penalty_normalizer": "opd"}
    }

    with pytest.raises(ValueError):
        create_advantage_penalties(grpo_config, loss_config)
