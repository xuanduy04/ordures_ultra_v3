import json
import os
import subprocess
from collections import defaultdict
from dataclasses import fields
from itertools import zip_longest
from threading import Event, Thread
from unittest.mock import patch

import psutil
import torch

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch, TokenAligner
from nemo_rl.utils.timer import Timer


class BatchCaptured(BaseException):
    """Exit the training loop after capture and run its usual worker cleanup."""


def physical_cpus(affinity):
    """Select one allowed logical CPU per physical core, alternating sockets."""
    topology = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,ONLINE"], text=True)
    by_socket = defaultdict(list)
    seen = set()
    for line in topology.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket, online = line.split(",")
        key = (socket, core)
        if online == "Y" and int(cpu) in affinity and key not in seen:
            by_socket[socket].append(int(cpu))
            seen.add(key)
    return [cpu for group in zip_longest(*by_socket.values()) for cpu in group if cpu is not None]


def measure_alignment(aligner, student_ids, teacher_ids, masks, cpus):
    """Measure one batch with inherited worker affinity and process-tree RSS."""
    root = psutil.Process()
    peak_rss = 0
    done = Event()

    def sample_rss():
        nonlocal peak_rss
        while True:
            total = 0
            for process in [root, *root.children(recursive=True)]:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak_rss = max(peak_rss, total)
            if done.wait(1):
                return

    original_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    sampler = Thread(target=sample_rss, daemon=True)
    sampler.start()
    timer = Timer(context={"worker": "alignment_scale", "batch": len(student_ids), "cores": len(cpus)})
    try:
        with timer.time("total"):
            result = aligner.align(student_ids, teacher_ids, timer=timer, **masks)
    finally:
        done.set()
        sampler.join()
        os.sched_setaffinity(0, original_affinity)
    return result, timer.get_timing_metrics(), peak_rss


def main():
    """Reuse one real rollout batch to sweep alignment at fixed token lengths."""
    import ray

    from examples.nemo_gym.run_grpo_nemo_gym import main as run_rollouts

    available = physical_cpus(os.sched_getaffinity(0))
    if len(available) < 96:
        raise ValueError("The scaling test requires 96 allowed physical CPU cores")
    capture = None

    def capture_inputs(self, student_ids, teacher_ids, **masks):
        nonlocal capture
        capture = (self, student_ids, teacher_ids, masks)
        print(f"XTOKEN_SCALING_CAPTURE samples={len(student_ids)}", flush=True)
        raise BatchCaptured

    with patch.object(TokenAligner, "align", capture_inputs):
        try:
            run_rollouts()
        except BatchCaptured:
            pass
        finally:
            ray.shutdown()

    if capture is None:
        raise RuntimeError("No alignment inputs captured")
    source, student_ids, teacher_ids, masks = capture
    if len(student_ids) != 128:
        raise ValueError(f"The test expects 128 source rollouts, got {len(student_ids)}")

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)

    print(
        "XTOKEN_SCALING_INPUT "
        + json.dumps(
            {
                "source_samples": len(student_ids),
                "student_width": student_ids.shape[1],
                "teacher_width": teacher_ids.shape[1],
                "student_mean_valid": masks["student_attention_mask"].sum(1).float().mean().item(),
                "teacher_mean_valid": masks["teacher_attention_mask"].sum(1).float().mean().item(),
                "allowed_physical_cores": len(available),
            }
        ),
        flush=True,
    )
    reference = None
    for copies, workers in ((1, 12), (2, 24), (4, 48), (8, 96)):
        aligner = TokenAligner(
            source.student_tokenizer,
            source.teacher_tokenizer,
            source.projection_matrix_path,
            source.max_combination_len,
            num_workers=workers,
        )
        result, timings, peak_rss = measure_alignment(
            aligner,
            student_ids.repeat(copies, 1),
            teacher_ids.repeat(copies, 1),
            {name: mask.repeat(copies, 1) for name, mask in masks.items()},
            available[:workers],
        )
        if reference is None:
            reference = result
        for field in fields(AlignmentBatch):
            data = getattr(reference, field.name)
            expected = data.repeat((copies,) + (1,) * (data.ndim - 1))
            torch.testing.assert_close(getattr(result, field.name), expected, rtol=0, atol=0, msg=field.name)
        print(
            "XTOKEN_SCALING_RESULT "
            + json.dumps(
                {
                    "samples": len(student_ids) * copies,
                    "physical_cores": workers,
                    "timings_s": timings,
                    "peak_tree_rss_bytes": peak_rss,
                    "equal_fields": len(fields(AlignmentBatch)),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
