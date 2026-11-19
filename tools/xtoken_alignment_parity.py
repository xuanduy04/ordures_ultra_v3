import json
from dataclasses import fields
from time import perf_counter
from unittest.mock import patch

import torch

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch, TokenAligner
from nemo_rl.utils.timer import Timer


def main():
    """Run the standard Gym entry point and check both aligners on each real training batch."""
    from examples.nemo_gym.run_grpo_nemo_gym import main as train

    align = TokenAligner.align
    align_dp = TokenAligner._align_dp

    def compare(self, student_ids, teacher_ids, **masks):
        if self.num_workers < 2:
            raise ValueError("The parity run requires alignment_workers >= 2")

        dp_s = 0.0
        dp_cells = 0

        def measure_dp(student_tokens, teacher_tokens, **kwargs):
            nonlocal dp_s, dp_cells
            started = perf_counter()
            result = align_dp(student_tokens, teacher_tokens, **kwargs)
            dp_s += perf_counter() - started
            dp_cells += len(student_tokens) * len(teacher_tokens)
            return result

        serial_timer = Timer(context={"worker": "alignment_serial"})
        parallel_timer = Timer(context={"worker": "alignment_parallel"})
        serial = TokenAligner(
            self.student_tokenizer, self.teacher_tokenizer, self.projection_matrix_path, self.max_combination_len
        )
        with patch.object(TokenAligner, "_align_dp", staticmethod(measure_dp)):
            with serial_timer.time("total"):
                expected = align(serial, student_ids, teacher_ids, timer=serial_timer, **masks)
        with parallel_timer.time("total"):
            actual = align(self, student_ids, teacher_ids, timer=parallel_timer, **masks)

        names = [field.name for field in fields(AlignmentBatch)]
        for name in names:
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0, msg=name)
        print(
            "XTOKEN_ALIGNMENT_PARITY "
            + json.dumps(
                {
                    "samples": student_ids.shape[0],
                    "workers": self.num_workers,
                    "equal_fields": names,
                    "serial_s": serial_timer.get_timing_metrics(),
                    "serial_dp_s": dp_s,
                    "serial_dp_cells": dp_cells,
                    "parallel_s": parallel_timer.get_timing_metrics(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return actual

    with patch.object(TokenAligner, "align", compare):
        train()


if __name__ == "__main__":
    main()
