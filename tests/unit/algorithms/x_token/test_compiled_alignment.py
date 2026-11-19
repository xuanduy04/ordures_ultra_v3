import unittest
from dataclasses import fields
from importlib.util import find_spec
from random import Random

import torch

from nemo_rl.algorithms.x_token.compiled_alignment import align_dp_numba, pack_alignment_numpy
from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch, AlignmentPair, TokenAligner


class _Tokenizer:
    def __init__(self, tokens):
        self.tokens = tokens

    def convert_ids_to_tokens(self, ids):
        return [self.tokens[index] for index in ids]


class TestCompiledAlignment(unittest.TestCase):
    def assert_alignment_equal(self, actual, expected):
        for field in fields(AlignmentBatch):
            torch.testing.assert_close(
                getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0, msg=field.name
            )

    @unittest.skipUnless(find_spec("numba"), "Numba backend is optional")
    def test_dp_scores_and_traceback(self):
        rng = Random(1729)
        tokens = ["a", "b", "ab", "Ġa", "é", "<pad>", ""]
        for case in range(320):
            student = rng.choices(tokens, k=rng.randrange(20))
            teacher = rng.choices(tokens, k=rng.randrange(20))
            options = {
                "exact_match_score": (3.0, 0.0, 1.1)[case % 3],
                "combination_score_multiplier": (1.5, 0.0, 0.37)[case % 3],
                "gap_penalty": (-1.5, 0.0, -0.21)[case % 3],
                "max_combination_len": case % 7,
                "ignore_leading_char_diff": bool(case % 2),
            }
            self.assertEqual(
                align_dp_numba(student, teacher, **options), TokenAligner._align_dp(student, teacher, **options), case
            )

    def test_numpy_packing_preserves_overlapping_spans(self):
        pairs = [
            [
                AlignmentPair([], [], 0, 3, 1, 2, True),
                AlignmentPair([], [], 1, 2, 1, 2, True),
                AlignmentPair([], [], -1, -1, 3, 5, False),
                AlignmentPair([], [], 7, 8, 0, 1, True),
            ],
            [],
        ]
        shape = {"b": 2, "t_s": 4, "t_t": 5}
        self.assert_alignment_equal(pack_alignment_numpy(pairs, **shape), TokenAligner._pairs_to_batch(pairs, **shape))

    @unittest.skipUnless(find_spec("numba"), "Numba backend is optional")
    def test_spawned_compiled_backend_matches_python(self):
        student = _Tokenizer(["<pad>", "Hello", "Ġworld", "ĠâĪ", "ĳ", "!", "Ġa", "b"])
        teacher = _Tokenizer(["<pad>", "Hello", "Ġworld", "Ġ∑", "!", "Ġab"])
        student_ids = torch.tensor([[1, 2, 3, 4, 5], [0, 6, 7, 0, 0], [0, 0, 0, 0, 0]])
        teacher_ids = torch.tensor([[1, 2, 3, 4], [5, 0, 0, 0], [0, 0, 0, 0]])
        masks = {"student_attention_mask": student_ids.ne(0), "teacher_attention_mask": teacher_ids.ne(0)}
        expected = TokenAligner(student, teacher, "", 4).align(student_ids, teacher_ids, **masks)
        optimized = TokenAligner(student, teacher, "", 4, num_workers=2, backend="numba")
        self.assert_alignment_equal(optimized.align(student_ids, teacher_ids, **masks), expected)


if __name__ == "__main__":
    unittest.main()
