import unittest
from dataclasses import fields

import torch

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch, TokenAligner


class TokenStrings:
    def __init__(self, tokens):
        self.tokens = tokens

    def convert_ids_to_tokens(self, ids):
        return [self.tokens[token] for token in ids]


class TestParallelAlignment(unittest.TestCase):
    def test_parallel_keeps_order_and_exact_fields(self):
        student = TokenStrings(["<pad>", "Hello", "Ġworld", "ĠâĪ", "ĳ", "!", "Ġa", "b"])
        teacher = TokenStrings(["<pad>", "Hello", "Ġworld", "Ġ∑", "!", "Ġab"])
        student_ids = torch.tensor([[1, 2, 3, 4, 5], [0, 6, 7, 0, 0], [0, 0, 0, 0, 0]])
        teacher_ids = torch.tensor([[1, 2, 3, 4], [5, 0, 0, 0], [0, 0, 0, 0]])
        masks = {"student_attention_mask": student_ids.ne(0), "teacher_attention_mask": teacher_ids.ne(0)}

        expected = TokenAligner(student, teacher, "", 4).align(student_ids, teacher_ids, **masks)
        actual = TokenAligner(student, teacher, "", 4, num_workers=2).align(student_ids, teacher_ids, **masks)

        for field in fields(AlignmentBatch):
            torch.testing.assert_close(
                getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0, msg=field.name
            )


if __name__ == "__main__":
    unittest.main()
