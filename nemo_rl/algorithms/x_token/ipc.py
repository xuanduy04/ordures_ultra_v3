"""Persistent teacher-logit IPC buffer and deterministic sample ordering."""
from typing import Any, Optional
import torch
from nemo_rl.models.policy.utils import get_handle_from_tensor

def get_handle_from_tensor(tensor: torch.Tensor) -> tuple[Any]:
    """Get IPC handle from a tensor."""
    from torch.multiprocessing.reductions import reduce_tensor

    # skip serializing the function for better refit performance
    return reduce_tensor(tensor.detach())[1:]

def ensure_teacher_ipc_buffer(
    storage: Optional[torch.Tensor],
    handle: Optional[tuple[Any, ...]],
    num_microbatches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[Any, ...]]:
    """Lazy-alloc / grow ``[N_mb, B, T, V]`` teacher-logits IPC storage.

    Returns the (possibly reallocated) ``(storage, handle)``. Reallocates and
    re-exports the IPC handle whenever any dim of the requested shape exceeds
    the current storage, or dtype/device changed; otherwise the existing
    storage and cached handle are returned unchanged.
    """
    needs_realloc = (
        storage is None
        or storage.shape[0] < num_microbatches
        or storage.shape[1] < batch_size
        or storage.shape[2] < seq_len
        or storage.shape[3] < vocab_size
        or storage.dtype != dtype
        or storage.device != device
    )
    if needs_realloc:
        storage = torch.empty(
            (num_microbatches, batch_size, seq_len, vocab_size),
            dtype=dtype,
            device=device,
        )
        handle = get_handle_from_tensor(storage)
    assert storage is not None and handle is not None
    return storage, handle

def aggregate_per_sample_handles(
    worker_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten teacher per-sample IPC handles into a global-batch-ordered list.

    Each worker returns ``{"dp_rank": int, "per_sample_handles": list}`` where
    the handle list is in local sample order; the several workers sharing a
    ``dp_rank`` are TP/CP replicas that each contribute one shard per sample.
    Concatenating samples in ``sorted(dp_rank)`` order reproduces the original
    global sample order (rank 0 holds the first ``gbs/dp`` samples, rank 1 the
    next, ...), so the result is a length-``gbs`` list independent of the
    teacher's DP degree. Element ``i`` is ``{"teacher_shards": [shard, ...]}``
    holding all TP×CP shards of global sample ``i``.
    """
    handles_by_dp_rank: dict[int, list[list[dict[str, Any]]]] = {}
    for worker_result in worker_results:
        dp_rank = worker_result["dp_rank"]
        handles_by_dp_rank.setdefault(dp_rank, []).append(
            worker_result["per_sample_handles"]
        )
    aggregated: list[dict[str, Any]] = []
    for dp_rank in sorted(handles_by_dp_rank.keys()):
        worker_handles_in_dp = handles_by_dp_rank[dp_rank]
        num_samples = len(worker_handles_in_dp[0])
        for worker_handles in worker_handles_in_dp:
            assert len(worker_handles) == num_samples, (
                f"dp={dp_rank}: per_sample_handles length mismatch "
                f"{[len(h) for h in worker_handles_in_dp]}"
            )
        for sample_idx in range(num_samples):
            aggregated.append(
                {
                    "teacher_shards": [
                        worker_handles[sample_idx]
                        for worker_handles in worker_handles_in_dp
                    ]
                }
            )
    return aggregated
