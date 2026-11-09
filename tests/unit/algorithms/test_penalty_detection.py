import pytest
import torch

from nemo_rl.algorithms.advantage_penalties.detectors import apply_penalty_detection
from nemo_rl.algorithms.advantage_penalties.tool_call_expectations import expects_tool_call


# ---- Helpers to build minimal result dicts ----


def _make_result(reward=1.0, output_items=None, message_log=None):
    """Build a minimal result dict matching the structure from nemo_gym."""
    return {
        "full_result": {"reward": reward, "response": {"output": output_items or []}},
        "message_log": message_log or [],
    }


def _reasoning_item(text, generation_str=None):
    item = {"type": "reasoning", "summary": [{"text": text, "type": "summary_text"}]}
    if generation_str is not None:
        item["generation_str"] = generation_str
    return item


def _message_item(text, generation_str=None):
    item = {"type": "message", "content": [{"text": text, "type": "output_text"}], "role": "assistant"}
    if generation_str is not None:
        item["generation_str"] = generation_str
    return item


def _function_call_item(name="tool", generation_str=None):
    item = {"type": "function_call", "name": name, "arguments": "{}", "call_id": "c1"}
    if generation_str is not None:
        item["generation_str"] = generation_str
    return item


def _msg(role, token_ids):
    return {"role": role, "token_ids": torch.tensor(token_ids)}


# =====================================================================
# Empty final answer (grpo.advantage_penalty.empty_final_answer)
# =====================================================================


class TestEmptyFinalAnswer:
    CFG = {"grpo": {"advantage_penalty": {"empty_final_answer": {"enabled": True}}}}

    def test_marks_last_assistant_message(self):
        result = _make_result(
            reward=1.0,
            output_items=[_reasoning_item("thinking"), _message_item("")],
            message_log=[_msg("user", [100]), _msg("assistant", [300, 400])],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert counts["empty_final_answer"] == 1
        assert "has_empty_final_answer" not in result["message_log"][0]
        assert result["message_log"][1]["has_empty_final_answer"] is True

    def test_multi_turn_marks_only_last_assistant(self):
        result = _make_result(
            reward=1.0,
            output_items=[_reasoning_item("thinking"), _message_item("")],
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 400]),
                _msg("user", [500]),
                _msg("assistant", [600, 700]),
            ],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert "has_empty_final_answer" not in result["message_log"][1]
        assert result["message_log"][3]["has_empty_final_answer"] is True

    def test_nonempty_answer_not_flagged(self):
        result = _make_result(
            reward=1.0,
            output_items=[_reasoning_item("thinking"), _message_item("The answer is 42")],
            message_log=[_msg("user", [100]), _msg("assistant", [300, 400])],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert "has_empty_final_answer" not in result["message_log"][1]

    def test_function_call_skip_not_flagged(self):
        result = _make_result(
            reward=1.0,
            output_items=[_reasoning_item("thinking"), _function_call_item()],
            message_log=[_msg("user", [100]), _msg("assistant", [300, 400])],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert "has_empty_final_answer" not in result["message_log"][1]

    def test_empty_message_log_no_crash(self):
        result = _make_result(reward=1.0, output_items=[])
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert counts["empty_final_answer"] == 1

    def test_grpo_flag_triggers_roles_guard(self):
        result = _make_result(reward=1.0, message_log=[_msg("user", [100]), _msg("tool", [300])])
        try:
            apply_penalty_detection([result], self.CFG)
        except AssertionError:
            pass
        else:
            raise AssertionError("expected gym-roles AssertionError")


# =====================================================================
# EOS token in generation (grpo.advantage_penalty.eos_token)
# =====================================================================


class TestEosToken:
    CFG = {"grpo": {"advantage_penalty": {"eos_token": {"enabled": True}}}}

    def test_marks_from_first_eos_to_final_message_assistant_only(self):
        """Punish from the literal first EOS token until the final token,
        assistant turns only.
        """
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100]),
                _msg("assistant", [300, 400]),  # clean turn before first EOS
                _msg("user", [500]),
                _msg("assistant", [600, 2]),  # first EOS here
                _msg("user", [700]),
                _msg("assistant", [800, 900]),  # after EOS — still flagged
            ],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert counts["eos_token"] == 1
        ml = result["message_log"]
        assert "has_eos_token" not in ml[0]  # user
        assert "has_eos_token" not in ml[1]  # assistant before first EOS
        assert "has_eos_token" not in ml[2]  # user
        assert ml[3]["has_eos_token"] is True
        assert "has_eos_token" not in ml[4]  # user after EOS — never flagged
        assert ml[5]["has_eos_token"] is True

    def test_no_eos_no_flags(self):
        result = _make_result(reward=1.0, message_log=[_msg("user", [100]), _msg("assistant", [300, 400])])
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert "has_eos_token" not in result["message_log"][1]

    def test_eos_in_user_only_no_flags(self):
        result = _make_result(reward=1.0, message_log=[_msg("user", [100, 2, 200]), _msg("assistant", [300, 400])])
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert "has_eos_token" not in result["message_log"][0]
        assert "has_eos_token" not in result["message_log"][1]

    def test_custom_eos_token_id(self):
        cfg = {"grpo": {"advantage_penalty": {"eos_token": {"enabled": True}}}, "token_ids": {"eos": 99}}
        result = _make_result(reward=1.0, message_log=[_msg("user", [100]), _msg("assistant", [300, 99, 400])])
        counts = apply_penalty_detection([result], cfg)
        assert result["full_result"]["reward"] == 1.0
        assert result["message_log"][1]["has_eos_token"] is True

    def test_custom_multi_eos_token_ids(self):
        cfg = {"grpo": {"advantage_penalty": {"eos_token": {"enabled": True}}}, "token_ids": {"eos": "2,1 ,6,,7,  "}}
        result = _make_result(reward=1.0, message_log=[_msg("user", [100]), _msg("assistant", [300, 6, 400])])
        counts = apply_penalty_detection([result], cfg)
        assert result["full_result"]["reward"] == 1.0
        assert counts["eos_token"] == 1
        assert result["message_log"][1]["has_eos_token"] is True

    def test_grpo_flag_triggers_roles_guard(self):
        result = _make_result(reward=1.0, message_log=[_msg("user", [100]), _msg("tool", [300])])
        try:
            apply_penalty_detection([result], self.CFG)
        except AssertionError:
            pass
        else:
            raise AssertionError("expected gym-roles AssertionError")


# =====================================================================
# Malformed think tags (grpo.advantage_penalty.malformed_thinking)
# =====================================================================


class TestMalformedThinkTag:
    CFG = {"grpo": {"advantage_penalty": {"malformed_thinking": {"enabled": True}}}}

    def test_token_violation_marks_only_violating_turns(self):
        """Turn 1 valid, turn 2 has double close — only turn 2 marked."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400]),
                _msg("user", [500, 12]),
                _msg("assistant", [600, 13, 13, 700]),
            ],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert counts["malformed_thinking"] == 1
        ml = result["message_log"]
        assert "has_malformed_thinking" not in ml[0]
        assert "has_malformed_thinking" not in ml[1]
        assert ml[3]["has_malformed_thinking"] is True

    def test_multiple_violating_turns_all_marked(self):
        """Both turns malformed — every violating assistant message marked."""
        result = _make_result(
            reward=1.0,
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 13]),
                _msg("user", [500, 12]),
                _msg("assistant", [600, 13, 13]),
            ],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        ml = result["message_log"]
        assert ml[1]["has_malformed_thinking"] is True
        assert ml[3]["has_malformed_thinking"] is True

    def test_no_violation_no_flags(self):
        result = _make_result(reward=1.0, message_log=[_msg("user", [100, 12]), _msg("assistant", [300, 13, 400])])
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert "has_malformed_thinking" not in result["message_log"][1]

    def test_string_violation_marks_corresponding_assistant(self):
        """Piecemeal thinking in generation_str marks that turn's assistant."""
        result = _make_result(
            reward=1.0,
            output_items=[_message_item("answer", generation_str="some <think> text </think> answer")],
            message_log=[
                _msg("user", [100, 12]),
                _msg("assistant", [300, 13, 400]),  # token IDs are fine
            ],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert result["full_result"]["reward"] == 1.0
        assert result["message_log"][1]["has_malformed_thinking"] is True


# =====================================================================
# N-gram repetition (grpo.advantage_penalty.n_gram_repetition)
# =====================================================================


class TestNGramRepetition:
    CFG = {"grpo": {"advantage_penalty": {"n_gram_repetition": {"enabled": True}}}}

    def test_attaches_mask_and_counts_repeated_samples(self):
        result = _make_result(
            reward=1.0,
            message_log=[_msg("user", [100]), _msg("assistant", [1, 2, 3, 1, 2, 3, 1, 2, 3])],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert counts["n_gram_repetition"] == 1
        assert "n_gram_repetition_mask" in result["message_log"][1]

    def test_mask_values_explicit_size(self):
        cfg = {"grpo": {"advantage_penalty": {"n_gram_repetition": {"enabled": True, "size": 3, "threshold": 2}}}}
        result = _make_result(
            reward=1.0,
            message_log=[_msg("user", [100]), _msg("assistant", [1, 2, 3, 1, 2, 3, 1, 2, 3])],
        )
        apply_penalty_detection([result], cfg)
        mask = result["message_log"][1]["n_gram_repetition_mask"]
        torch.testing.assert_close(mask, torch.tensor([0.0, 0, 0, 1, 2, 3, 3, 2, 1]))

    def test_no_repetition_no_count(self):
        result = _make_result(reward=1.0, message_log=[_msg("user", [100]), _msg("assistant", [1, 2, 3, 4])])
        counts = apply_penalty_detection([result], self.CFG)
        assert counts["n_gram_repetition"] == 0
        assert "n_gram_repetition_mask" in result["message_log"][1]  # mask still attached
        assert not result["message_log"][1]["n_gram_repetition_mask"].any()

    def test_custom_size_and_threshold(self):
        cfg = {"grpo": {"advantage_penalty": {"n_gram_repetition": {"enabled": True, "size": 2, "threshold": 2}}}}
        result = _make_result(reward=1.0, message_log=[_msg("user", [100]), _msg("assistant", [1, 2, 1, 2])])
        counts = apply_penalty_detection([result], cfg)
        assert counts["n_gram_repetition"] == 1
        mask = result["message_log"][1]["n_gram_repetition_mask"]
        torch.testing.assert_close(mask, torch.tensor([0.0, 0, 1, 1]))

    def test_high_threshold_no_count(self):
        cfg = {"grpo": {"advantage_penalty": {"n_gram_repetition": {"enabled": True, "size": 3, "threshold": 3}}}}
        result = _make_result(reward=1.0, message_log=[_msg("user", [100]), _msg("assistant", [1, 2, 3, 1, 2, 3])])
        counts = apply_penalty_detection([result], cfg)
        assert counts["n_gram_repetition"] == 0

    def test_user_tokens_ignored(self):
        result = _make_result(
            reward=1.0,
            message_log=[_msg("user", [1, 2, 3, 1, 2, 3, 1, 2, 3]), _msg("assistant", [9, 9, 9])],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert counts["n_gram_repetition"] == 0
        assert "n_gram_repetition_mask" not in result["message_log"][0]


# =====================================================================
# Invalid tool call (grpo.advantage_penalty.invalid_tool_call)
# =====================================================================


def _single_turn(output_items):
    return _make_result(output_items=output_items, message_log=[_msg("user", [100]), _msg("assistant", [300])])


def _row(agent_name="some_agent", **extra):
    return {"agent_ref": {"name": agent_name}, **extra}


class TestInvalidToolCall:
    CFG = {"grpo": {"advantage_penalty": {"invalid_tool_call": {"enabled": True}}}}

    def test_pattern_in_message_text_flagged(self):
        result = _single_turn([_message_item('<tool_call>{"name": "x"}</tool_call>', generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG)
        assert counts["invalid_tool_call"] == 1
        assert result["message_log"][1]["is_invalid_tool_call"] is True

    def test_pattern_in_reasoning_summary_flagged(self):
        result = _single_turn([_reasoning_item("thinking <function_call> x", generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG)
        assert counts["invalid_tool_call"] == 1
        assert result["message_log"][1]["is_invalid_tool_call"] is True

    def test_clean_text_not_flagged(self):
        result = _single_turn([_message_item("The answer is 42", generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG)
        assert counts["invalid_tool_call"] == 0
        assert "is_invalid_tool_call" not in result["message_log"][1]

    def test_structured_function_call_not_flagged(self):
        result = _single_turn([_reasoning_item("thinking"), _function_call_item(generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG)
        assert counts["invalid_tool_call"] == 0
        assert "is_invalid_tool_call" not in result["message_log"][1]

    def test_custom_patterns(self):
        cfg = {
            "grpo": {
                "advantage_penalty": {"invalid_tool_call": {"enabled": True, "invalid_tool_call_patterns": ["[CALL]"]}}
            }
        }
        default_hit = _single_turn([_message_item("<tool_call>", generation_str="g")])
        custom_hit = _single_turn([_message_item("[CALL] x", generation_str="g")])
        counts = apply_penalty_detection([default_hit, custom_hit], cfg)
        assert counts["invalid_tool_call"] == 1
        assert "is_invalid_tool_call" not in default_hit["message_log"][1]
        assert custom_hit["message_log"][1]["is_invalid_tool_call"] is True

    def test_multi_turn_marks_only_offending_turn(self):
        result = _make_result(
            output_items=[
                _function_call_item(generation_str="g1"),
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                _message_item("<tool_call>", generation_str="g2"),
            ],
            message_log=[_msg("user", [100]), _msg("assistant", [300]), _msg("user", [400]), _msg("assistant", [500])],
        )
        counts = apply_penalty_detection([result], self.CFG)
        assert counts["invalid_tool_call"] == 1
        assert "is_invalid_tool_call" not in result["message_log"][1]
        assert result["message_log"][3]["is_invalid_tool_call"] is True

    def test_disabled_no_flag(self):
        cfg = {"grpo": {"advantage_penalty": {"invalid_tool_call": {"enabled": False}}}}
        result = _single_turn([_message_item("<tool_call>", generation_str="g")])
        counts = apply_penalty_detection([result], cfg)
        assert counts["invalid_tool_call"] == 0
        assert "is_invalid_tool_call" not in result["message_log"][1]

    def test_turn_count_mismatch_asserts(self):
        result = _single_turn([_message_item("answer")])  # no generation_str
        with pytest.raises(AssertionError):
            apply_penalty_detection([result], self.CFG)


# =====================================================================
# Missing tool call (grpo.advantage_penalty.missing_tool_call)
# =====================================================================


class TestMissingToolCall:
    CFG = {"grpo": {"advantage_penalty": {"missing_tool_call": {"enabled": True}}}}
    EXPECTING_ROW = _row("single_step_tool_use_with_argument_comparison_simple_agent")

    def test_expected_message_turn_flagged(self):
        result = _single_turn([_reasoning_item("thinking"), _message_item("answer", generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG, [self.EXPECTING_ROW])
        assert counts["missing_tool_call"] == 1
        assert result["message_log"][1]["is_missing_tool_call"] is True

    def test_expected_function_call_turn_not_flagged(self):
        result = _single_turn([_reasoning_item("thinking"), _function_call_item(generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG, [self.EXPECTING_ROW])
        assert counts["missing_tool_call"] == 0
        assert "is_missing_tool_call" not in result["message_log"][1]

    def test_function_call_before_final_item_not_flagged(self):
        """A function_call that is not the token-bearing item still counts for its turn."""
        result = _single_turn([_function_call_item(), _function_call_item(name="other", generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG, [self.EXPECTING_ROW])
        assert counts["missing_tool_call"] == 0

    @pytest.mark.parametrize(
        "row",
        [
            _row("some_other_agent"),
            _row("general_qa_simple_agent"),
            _row("general_qa_simple_agent", answer_is_tool_call=False),
        ],
    )
    def test_not_expected_not_flagged(self, row):
        result = _single_turn([_message_item("answer", generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG, [row])
        assert counts["missing_tool_call"] == 0
        assert "is_missing_tool_call" not in result["message_log"][1]

    def test_general_qa_answer_is_tool_call_flagged(self):
        result = _single_turn([_message_item("answer", generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG, [_row("general_qa_simple_agent", answer_is_tool_call=True)])
        assert counts["missing_tool_call"] == 1
        assert result["message_log"][1]["is_missing_tool_call"] is True

    def test_invalid_takes_precedence_both_enabled(self):
        cfg = {
            "grpo": {"advantage_penalty": {"invalid_tool_call": {"enabled": True}, "missing_tool_call": {"enabled": True}}}
        }
        result = _single_turn([_message_item("<tool_call>", generation_str="g")])
        counts = apply_penalty_detection([result], cfg, [self.EXPECTING_ROW])
        assert counts["invalid_tool_call"] == 1
        assert counts["missing_tool_call"] == 0
        assert result["message_log"][1]["is_invalid_tool_call"] is True
        assert "is_missing_tool_call" not in result["message_log"][1]

    def test_invalid_takes_precedence_only_missing_enabled(self):
        result = _single_turn([_message_item("<tool_call>", generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG, [self.EXPECTING_ROW])
        assert counts["missing_tool_call"] == 0
        assert "is_missing_tool_call" not in result["message_log"][1]
        assert "is_invalid_tool_call" not in result["message_log"][1]

    def test_truncated_reasoning_only_turn_flagged(self):
        result = _single_turn([_reasoning_item("thinking that got cut o", generation_str="g")])
        counts = apply_penalty_detection([result], self.CFG, [self.EXPECTING_ROW])
        assert counts["missing_tool_call"] == 1
        assert result["message_log"][1]["is_missing_tool_call"] is True

    def test_multi_turn_flags_only_turn_without_tool_call(self):
        result = _make_result(
            output_items=[
                _reasoning_item("thinking"),
                _function_call_item(generation_str="g1"),
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                _message_item("final answer", generation_str="g2"),
            ],
            message_log=[_msg("user", [100]), _msg("assistant", [300]), _msg("user", [400]), _msg("assistant", [500])],
        )
        counts = apply_penalty_detection([result], self.CFG, [self.EXPECTING_ROW])
        assert counts["missing_tool_call"] == 1
        assert "is_missing_tool_call" not in result["message_log"][1]
        assert result["message_log"][3]["is_missing_tool_call"] is True

    def test_masked_dummy_result_not_flagged(self):
        result = _make_result(output_items=[], message_log=[_msg("user", [0])])
        counts = apply_penalty_detection([result], self.CFG, [self.EXPECTING_ROW])
        assert counts["missing_tool_call"] == 0

    def test_enabled_without_rows_asserts(self):
        result = _single_turn([_message_item("answer", generation_str="g")])
        with pytest.raises(AssertionError):
            apply_penalty_detection([result], self.CFG)

    def test_disabled_no_flag(self):
        cfg = {"grpo": {"advantage_penalty": {"missing_tool_call": {"enabled": False}}}}
        result = _single_turn([_message_item("answer", generation_str="g")])
        counts = apply_penalty_detection([result], cfg, [self.EXPECTING_ROW])
        assert counts["missing_tool_call"] == 0
        assert "is_missing_tool_call" not in result["message_log"][1]


class TestExpectsToolCall:
    @pytest.mark.parametrize(
        ("row", "expected"),
        [
            (_row("single_step_tool_use_with_argument_comparison_simple_agent"), True),
            (_row("general_qa_simple_agent", answer_is_tool_call=True), True),
            (_row("general_qa_simple_agent", answer_is_tool_call=False), False),
            (_row("general_qa_simple_agent"), False),
            (_row("math_simple_agent", answer_is_tool_call=True), False),
        ],
    )
    def test_rules(self, row, expected):
        assert expects_tool_call(row) is expected


# =====================================================================
# Cross-cutting: multiple detections, config gating, batch behavior
# =====================================================================


class TestCrossCutting:
    def test_no_config_no_detections(self):
        result = _make_result(reward=1.0)
        counts = apply_penalty_detection([result], None)
        assert result["full_result"]["reward"] == 1.0
        assert counts == {
            "empty_final_answer": 0,
            "eos_token": 0,
            "malformed_thinking": 0,
            "n_gram_repetition": 0,
            "invalid_tool_call": 0,
            "missing_tool_call": 0,
        }

    def test_empty_config_no_detections(self):
        result = _make_result(reward=1.0)
        counts = apply_penalty_detection([result], {})
        assert result["full_result"]["reward"] == 1.0
        assert all(v == 0 for v in counts.values())

    def test_empty_results_no_crash(self):
        counts = apply_penalty_detection([], {"grpo": {"advantage_penalty": {"eos_token": {"enabled": True}}}})
        assert all(v == 0 for v in counts.values())

    def test_multiple_detections_stack(self):
        """A result that triggers both the EOS and empty-final-answer detections."""
        cfg = {
            "grpo": {
                "advantage_penalty": {"eos_token": {"enabled": True}, "empty_final_answer": {"enabled": True}}
            }
        }
        result = _make_result(
            reward=1.0,
            output_items=[_reasoning_item("thinking"), _message_item("")],
            message_log=[_msg("user", [100]), _msg("assistant", [300, 2])],
        )
        counts = apply_penalty_detection([result], cfg)
        assert counts["eos_token"] == 1
        assert counts["empty_final_answer"] == 1
        assert result["message_log"][1]["has_eos_token"] is True
        assert result["message_log"][1]["has_empty_final_answer"] is True

    def test_batch_of_results_mixed(self):
        """Two results: first is fine, second has EOS."""
        cfg = {"grpo": {"advantage_penalty": {"eos_token": {"enabled": True}}}}
        r1 = _make_result(reward=1.0, message_log=[_msg("user", [100]), _msg("assistant", [300, 400])])
        r2 = _make_result(reward=1.0, message_log=[_msg("user", [100]), _msg("assistant", [300, 2])])
        counts = apply_penalty_detection([r1, r2], cfg)
        assert counts["eos_token"] == 1
        assert "has_eos_token" not in r1["message_log"][1]
        assert r2["message_log"][1]["has_eos_token"] is True

    def test_disabled_detection_leaves_no_flags(self):
        cfg = {"grpo": {"advantage_penalty": {"malformed_thinking": {"enabled": False}}}}
        result = _make_result(reward=1.0, message_log=[_msg("user", [100, 13, 13, 12]), _msg("assistant", [300, 400])])
        counts = apply_penalty_detection([result], cfg)
        assert counts["malformed_thinking"] == 0
        assert "has_malformed_thinking" not in result["message_log"][1]

    def test_enabled_detection_does_not_leak_other_flags(self):
        """Only eos_token enabled: no empty_final_answer flag even when the answer is empty."""
        cfg = {"grpo": {"advantage_penalty": {"eos_token": {"enabled": True}}}}
        result = _make_result(
            reward=1.0,
            output_items=[_reasoning_item("thinking"), _message_item("")],
            message_log=[_msg("user", [100]), _msg("assistant", [300, 2])],
        )
        counts = apply_penalty_detection([result], cfg)
        assert counts["eos_token"] == 1
        assert counts["empty_final_answer"] == 0
        assert "has_empty_final_answer" not in result["message_log"][1]
