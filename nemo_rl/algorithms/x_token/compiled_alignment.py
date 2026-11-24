from functools import cache
from typing import Callable, cast

import numpy as np
import torch

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch, AlignmentPair, canonical_token

# Match the installed NumPy version's scalar addition to a Python float.
_DP_DTYPE = np.asarray(np.float32(0) + 0.0).dtype
_as_dp_operand = cast(Callable[[float], np.floating], _DP_DTYPE.type)


def _fill_dp(student, teacher, match, mismatch, gap, combination, limit):
    """Score one anchored segment and store the original strict tie order."""
    n, m = len(student) - 1, len(teacher) - 1
    score = np.zeros((n + 1, m + 1), dtype=np.float32)
    trace = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] + gap
        trace[i, 0] = 2
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] + gap
        trace[0, j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = student[i, 1] == teacher[j, 1]
            best = score[i - 1, j - 1] + (match if same else mismatch)
            move = 1
            up = score[i - 1, j] + gap
            left = score[i, j - 1] + gap
            if up > best:
                best, move = up, 2
            if left > best:
                best, move = left, 3
            for k in range(2, min(j, limit) + 1):
                if student[i, 1] == teacher[j, k]:
                    candidate = score[i - 1, j - k] + combination[k]
                    if candidate > best:
                        best, move = candidate, 3 + k
            for k in range(2, min(i, limit) + 1):
                if teacher[j, 1] == student[i, k]:
                    candidate = score[i - k, j - 1] + combination[k]
                    if candidate > best:
                        best, move = candidate, 3 + limit + k
            score[i, j] = best
            trace[i, j] = move
    return trace, score[n, m]


@cache
def _numba_kernel():
    from numba import njit

    return njit(nogil=True, fastmath=False, cache=True)(_fill_dp)


def align_dp_numba(
    student_tokens,
    teacher_tokens,
    *,
    exact_match_score,
    combination_score_multiplier,
    gap_penalty,
    max_combination_len,
    ignore_leading_char_diff,
):
    """Run the reference recurrence and traceback with an integer-coded compiled kernel."""
    kernel = _numba_kernel()
    symbols = {}
    limit = min(max(1, max_combination_len), max(1, len(student_tokens), len(teacher_tokens)))

    def encode(tokens):
        ids = np.zeros((len(tokens) + 1, limit + 1), dtype=np.int32)
        for end in range(1, len(tokens) + 1):
            for size in range(1, min(limit, end) + 1):
                value = tokens[end - 1] if size == 1 else "".join(tokens[end - size : end])
                if ignore_leading_char_diff:
                    value = canonical_token(value)
                ids[end, size] = symbols.setdefault(value, len(symbols) + 1)
        return ids

    student, teacher = encode(student_tokens), encode(teacher_tokens)
    combinations = np.asarray([combination_score_multiplier * k for k in range(limit + 1)], dtype=_DP_DTYPE)
    trace, score = kernel(
        student,
        teacher,
        _as_dp_operand(exact_match_score),
        _as_dp_operand(-exact_match_score),
        _as_dp_operand(gap_penalty),
        combinations,
        limit,
    )
    pairs = []
    i, j = len(student_tokens), len(teacher_tokens)
    while i or j:
        move = int(trace[i, j])
        if move == 1:
            ds, dt = 1, 1
        elif move == 2:
            ds, dt = 1, 0
        elif move == 3:
            ds, dt = 0, 1
        elif move <= 3 + limit:
            ds, dt = 1, move - 3
        else:
            ds, dt = move - 3 - limit, 1
        pairs.append(
            AlignmentPair(
                student_tokens[i - ds : i] if ds else [],
                teacher_tokens[j - dt : j] if dt else [],
                i - ds if ds else -1,
                i if ds else -1,
                j - dt if dt else -1,
                j if dt else -1,
            )
        )
        i -= ds
        j -= dt
    pairs.reverse()
    return pairs, float(score)


def pack_alignment_numpy(per_sample_pairs, *, b, t_s, t_t):
    """Pack all spans in a sample at once and preserve last-write-wins."""
    max_pairs = max(1, max(map(len, per_sample_pairs), default=0))
    valid = np.zeros((b, max_pairs), dtype=np.bool_)
    correct = np.zeros_like(valid)
    student_exact = np.zeros((b, t_s), dtype=np.bool_)
    teacher_exact = np.zeros((b, t_t), dtype=np.bool_)
    student_chunk = np.full((b, t_s), -1, dtype=np.int64)
    teacher_chunk = np.full((b, t_t), -1, dtype=np.int64)
    counts = np.asarray(list(map(len, per_sample_pairs)), dtype=np.int64)

    def write_side(starts, ends, width, pair_ids, exact, chunk, partition):
        keep = (starts >= 0) & (starts < width) & (ends > starts) & (ends <= width)
        start = starts[keep]
        lengths = ends[keep] - start
        offsets = lengths.cumsum() - lengths
        positions = np.arange(lengths.sum()) + np.repeat(start - offsets, lengths)
        # Pair indices are increasing, so maximum reproduces the last overlapping span.
        np.maximum.at(chunk, positions, np.repeat(pair_ids[keep], lengths))
        partition[starts[exact & (starts >= 0) & (starts < width)]] = True

    for row, pairs in enumerate(per_sample_pairs):
        n = len(pairs)
        if n == 0:
            continue
        s0, s1, t0, t1, same = np.asarray(
            [(p.s_start, p.s_end, p.t_start, p.t_end, bool(p.is_correct)) for p in pairs], dtype=np.int64
        ).T
        ids = np.arange(n, dtype=np.int64)
        valid[row, :n] = True
        correct[row, :n] = same
        exact = (same != 0) & (s0 != -1) & (t0 != -1) & (s1 - s0 == 1) & (t1 - t0 == 1)
        write_side(s0, s1, t_s, ids, exact, student_chunk[row], student_exact[row])
        write_side(t0, t1, t_t, ids, exact, teacher_chunk[row], teacher_exact[row])

    return AlignmentBatch(
        pair_valid=torch.from_numpy(valid),
        pair_is_correct=torch.from_numpy(correct),
        student_exact_partition_mask=torch.from_numpy(student_exact),
        teacher_exact_partition_mask=torch.from_numpy(teacher_exact),
        student_chunk_id=torch.from_numpy(student_chunk),
        teacher_chunk_id=torch.from_numpy(teacher_chunk),
        num_chunks=torch.from_numpy(counts),
    )
