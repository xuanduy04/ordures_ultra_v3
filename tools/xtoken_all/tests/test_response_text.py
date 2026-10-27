"""Exercise real collation control flow with small NumPy-backed tensor doubles."""

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


text_module = load("response_text_test", "nemo_rl/data/xtoken_response_text.py")


class Tensor(np.ndarray):
    def cpu(self):
        return self

    def clone(self):
        return self.copy()

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)


def tensor(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Tensor)


class Tokenizer:
    pad_token_id = 0
    all_special_ids = [0, 1]
    is_fast = True

    def decode(self, ids, **kwargs):
        return "\ufffd" if 3 in ids else "ok"

    def apply_chat_template(self, *args, **kwargs):
        return "T:"

    def __call__(self, text, **kwargs):
        assert "\ufffd" not in text
        count = len(text) - 2
        return {"input_ids": [10] + [20] * count,
                "offset_mapping": [(0, 2)] + [(2 + i, 3 + i) for i in range(count)]}


@dataclass
class Alignment:
    student_chunk_id: object
    teacher_chunk_id: object
    student_exact_partition_mask: object
    teacher_exact_partition_mask: object
    pair_valid: object
    pair_is_correct: object


class Aligner:
    def align(self, student, teacher, student_attention_mask, teacher_attention_mask):
        valid = (student_attention_mask.sum(1) > 0) & (teacher_attention_mask.sum(1) > 0)
        return Alignment(tensor(np.where(student_attention_mask, 0, -1)), tensor(np.where(teacher_attention_mask, 0, -1)),
            student_attention_mask.astype(bool), teacher_attention_mask.astype(bool), valid[:, None], valid[:, None].copy())


@pytest.fixture
def collator(monkeypatch):
    torch = SimpleNamespace(tensor=tensor, long=np.int64, bool=np.bool_,
        full=lambda shape, value, dtype=None: tensor(np.full(shape, value, dtype=dtype)),
        zeros_like=lambda array, dtype=None: tensor(np.zeros_like(array, dtype=dtype)),
        full_like=lambda array, value: tensor(np.full_like(array, value)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "nemo_rl.distributed.batched_data_dict", SimpleNamespace(BatchedDataDict=dict))
    monkeypatch.setitem(sys.modules, "nemo_rl.data.xtoken_response_text", text_module)
    module = load("response_collator_numpy_test", "nemo_rl/data/xtoken_rollout_collate.py")
    return module.RolloutCrossTokenizerCollator(Tokenizer(), [Tokenizer()], [Aligner()], 16, [16], 1, [1], [1])


def batch():
    return {"prompts": [[{"role": "user", "content": "q"}]] * 2, "sample_mask": tensor([1., 1.]), "idx": [7, 8]}


def generation(first):
    return {"output_ids": tensor([[9, first, 1], [9, 2, 1]]), "unpadded_sequence_lengths": tensor([3, 3])}


def test_bad_sample_is_masked_good_sample_survives_and_ids_never_change(collator, capsys):
    output = collator(batch(), generation(3), [[9], [9]], 66, 67)
    assert output["input_ids"].tolist() == [[9, 3, 1], [9, 2, 1]]
    assert output["sample_mask"].tolist() == [0., 1.]
    assert output["token_mask"].tolist() == [[0, 0, 0], [0, 1, 0]]
    assert not output["teacher_0_token_mask"][0].any()
    assert not output["alignment_0_pair_valid"][0].any()
    report = json.loads(capsys.readouterr().out.split("XTOKEN_RESPONSE_MASK=", 1)[1])
    assert report["masked_count"] == 1 and report["samples"][0]["source_index"] == 7
    assert report["sampled_ids_preserved"] and report["policy_version"] == 66


def test_same_tokenizer_needs_no_text_decode(collator, capsys):
    collator.aligners = [None]
    collator.student_tokenizer.decode = lambda *args, **kwargs: pytest.fail("same-tokenizer decoded text")
    output = collator(batch(), generation(3), [[9], [9]], 0, 1)
    assert output["sample_mask"].tolist() == [1., 1.]
    assert output["token_mask"].tolist() == [[0, 1, 0], [0, 1, 0]]
    assert not capsys.readouterr().out


def test_all_bad_samples_make_no_empty_optimizer_update(collator):
    bad = generation(3)
    bad["output_ids"][1, 1] = 3
    with pytest.raises(ValueError, match="no optimizer update"):
        collator(batch(), bad, [[9], [9]], 0, 1)


def test_structural_errors_still_fail_instead_of_being_masked(collator):
    bad = generation(0)
    with pytest.raises(ValueError, match="control"):
        collator(batch(), bad, [[9], [9]], 0, 1)
    with pytest.raises(ValueError, match="changed"):
        collator(batch(), generation(2), [[8], [9]], 0, 1)
