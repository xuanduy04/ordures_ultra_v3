import os
import re

import pytest
import torch

# apply_empty_generation_gate only needs torch, but rollouts.py imports ray/vllm/zmq.
# Extract and exec just the function to avoid the heavy dependency chain.
try:
    from nemo_rl.experience.rollouts import apply_empty_generation_gate
except (ImportError, ModuleNotFoundError):
    _rollouts_path = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "nemo_rl", "experience", "rollouts.py"
        )
    )
    _src = open(_rollouts_path).read()
    _match = re.search(r"^(def apply_empty_generation_gate\(.+?)(?=\ndef )", _src, re.DOTALL | re.MULTILINE)
    assert _match, "Could not find apply_empty_generation_gate in rollouts.py"
    exec(_match.group(1))


def _make_result(reward=1.0, output_items=None, message_log=None):
    """Build a minimal result dict matching the structure from nemo_gym."""
    return {
        "full_result": {"reward": reward, "response": {"output": output_items or []}},
        "message_log": message_log or [],
    }


def _msg(role, token_ids):
    return {"role": role, "token_ids": torch.tensor(token_ids)}


# =====================================================================
# Empty generation gate
# =====================================================================


class TestEmptyGenerationGate:
    CFG = {"token_ids": {"eos": "2, 11"}}

    def _empty_assistant(self):
        return {
            "role": "assistant",
            "token_ids": torch.tensor([], dtype=torch.int64),
            "generation_logprobs": torch.tensor([], dtype=torch.float32),
        }

    def test_empty_assistant_gated_to_smallest_eos(self):
        result = _make_result(message_log=[_msg("user", [1, 2, 3]), self._empty_assistant()])
        count = apply_empty_generation_gate([result], self.CFG)
        assert count == 1
        msg = result["message_log"][1]
        assert msg["token_ids"].tolist() == [2]
        assert msg["generation_logprobs"].tolist() == [0.0]
        assert msg["has_empty_generation"] is True

    def test_non_empty_assistant_untouched(self):
        result = _make_result(
            message_log=[
                _msg("user", [1]),
                {
                    "role": "assistant",
                    "token_ids": torch.tensor([5, 6]),
                    "generation_logprobs": torch.tensor([-0.1, -0.2]),
                },
            ]
        )
        count = apply_empty_generation_gate([result], self.CFG)
        assert count == 0
        assert result["message_log"][1]["token_ids"].tolist() == [5, 6]
        assert "has_empty_generation" not in result["message_log"][1]

    def test_int_eos_config(self):
        result = _make_result(message_log=[self._empty_assistant()])
        apply_empty_generation_gate([result], {"token_ids": {"eos": 11}})
        assert result["message_log"][0]["token_ids"].tolist() == [11]

    def test_none_config_noop(self):
        result = _make_result(message_log=[self._empty_assistant()])
        assert apply_empty_generation_gate([result], None) == 0
        assert result["message_log"][0]["token_ids"].numel() == 0
        assert "has_empty_generation" not in result["message_log"][0]

    def test_user_messages_untouched(self):
        result = _make_result(message_log=[_msg("user", [1, 2]), self._empty_assistant()])
        apply_empty_generation_gate([result], self.CFG)
        assert result["message_log"][0]["token_ids"].tolist() == [1, 2]

    def test_dtype_preserved(self):
        result = _make_result(message_log=[self._empty_assistant()])
        apply_empty_generation_gate([result], self.CFG)
        msg = result["message_log"][0]
        assert msg["token_ids"].dtype == torch.int64
        assert msg["generation_logprobs"].dtype == torch.float32

    def test_multiple_empty_messages_counted(self):
        result = _make_result(
            message_log=[_msg("user", [1]), self._empty_assistant(), _msg("user", [2]), self._empty_assistant()]
        )
        assert apply_empty_generation_gate([result], self.CFG) == 2

    def test_invalid_eos_config_raises(self):
        result = _make_result(message_log=[self._empty_assistant()])
        with pytest.raises(ValueError):
            apply_empty_generation_gate([result], {"token_ids": {"eos": "abc"}})
