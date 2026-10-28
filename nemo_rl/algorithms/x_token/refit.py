"""X-Token weight refit, isolated from the company GRPO training module."""
from __future__ import annotations

from contextlib import nullcontext
import gc
import os
from typing import TYPE_CHECKING, Optional

import ray

if TYPE_CHECKING:
    from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
    from nemo_rl.models.generation.interfaces import GenerationInterface
    from nemo_rl.utils.timer import Timer


def _malloc_trim() -> None:
    """Return freed glibc heap arenas to the OS to reduce RSS.

    CPython frees objects back to glibc, but glibc often retains the arena for
    reuse, so RSS stays at its high-water mark. This is a no-op if malloc_trim
    is unavailable.
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _log_rss(tag: str) -> None:
    """Print current and peak resident set size for memory diagnostics."""
    try:
        import resource

        with open("/proc/self/statm") as statm_file:
            resident_pages = int(statm_file.read().split()[1])
        page_size_bytes = os.sysconf("SC_PAGE_SIZE")
        current_gib = resident_pages * page_size_bytes / (1024**3)
        peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        print(
            f"  [rss] {tag}: current={current_gib:.2f} GiB, peak={peak_gib:.2f} GiB",
            flush=True,
        )
    except Exception:
        pass


def refit_policy_generation(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    colocated_inference: bool,
    _refit_buffer_size_gb: Optional[int] = None,
    timer: Optional[Timer] = None,
    kv_scales: Optional[dict[str, float]] = None,
) -> None:
    """Refit the policy generation interface with the latest policy weights.

    Args:
        policy: The policy to provide weights to the inference engine.
        policy_generation: The inference engine to refit.
        _refit_buffer_size_gb: The size of the buffer to use for refitting.
            If it is None, the buffer size will be computed by the remaining memory.
            This parameter is primarily used for testing.
        timer: Optional Timer used to time the prepare/transfer/update phase
        kv_scales: Optional dictionary of KV cache scales for FP8 quantization.
    """
    gc.collect()
    _malloc_trim()
    if colocated_inference:
        policy.offload_before_refit()
        policy_generation.prepare_for_generation(tags=["weights"])

    # Create a context manager that does nothing when timer is None
    timer_context = (
        timer.time("prepare_for_generation/transfer_and_update_weights")
        if timer is not None
        else nullcontext()
    )
    with timer_context:
        # update weights
        update_success = False
        if colocated_inference:
            # get model param keys, which is grouped by size
            if _refit_buffer_size_gb is not None:
                buffer_size_bytes = _refit_buffer_size_gb * (1024**3)
            else:
                # Empirically sets ratio as 30% to maximize efficiency.
                # The remaining 70% is a necessary buffer reserved for the parameter all-gathering across the expert-parallelism dimension.
                memory_ratio = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.3")
                buffer_size_bytes = int(
                    policy.get_free_memory_bytes() * float(memory_ratio)
                )

            # Original ZMQ IPC path for vLLM
            futures_train = policy.stream_weights_via_ipc_zmq(
                buffer_size_bytes=buffer_size_bytes
            )
            futures_inference = policy_generation.update_weights_via_ipc_zmq()
            # wait for all futures to complete
            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)
        else:
            # update weights through nccl
            policy.offload_before_refit()
            futures_train = policy.broadcast_weights_for_collective(kv_scales=kv_scales)
            futures_inference = policy_generation.update_weights_from_collective()
            # wait for all futures to complete
            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)
            policy.prepare_for_training()

        del futures_train, futures_inference, results

        # check if update is successful
        if not update_success:
            error_tag = "cuda-ipc" if colocated_inference else "nccl"
            error_message = (
                "❌ Error: Updating weights for the generation policy failed during refit.\n"
                f"This often indicates an issue with {error_tag} or "
                "a problem within the generation backend (e.g., vLLM worker).\n"
            )
            raise RuntimeError(error_message)
    gc.collect()
    if colocated_inference:
        policy.offload_after_refit()
        policy_generation.prepare_for_generation(tags=["kv_cache"])
    gc.collect()
    _malloc_trim()
    _log_rss("refit/exit")
