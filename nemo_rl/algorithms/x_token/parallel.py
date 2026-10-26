"""Differentiable X-Token parallel helpers ported from NeMo RL v0.7.0."""
from typing import Any, Optional
import torch
import torch.distributed.nn.functional
from torch.distributed.tensor import DTensor
from nemo_rl.distributed.model_utils import get_logprobs_from_vocab_parallel_logits

def _compute_distributed_log_softmax_with_grad(
    vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Differentiable stable distributed log_softmax across tensor parallel workers.

    Args:
        vocab_parallel_logits (torch.Tensor): Logits with shape
            [batch_size, seq_length, vocab_size // TP], where TP is the tensor
            parallel size.
        group (torch.distributed.ProcessGroup): Process group for the all-reduces.

    Returns:
        torch.Tensor: Log probabilities, same shape as the input, normalized across
            the full vocabulary dimension. Differentiable (gradients flow through
            the all-reduces).
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True).detach()
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=True).float()

    sum_exp_logits = torch.distributed.nn.functional.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )

    return vocab_parallel_logits - sum_exp_logits.log().to(vocab_parallel_logits.dtype)

@torch.no_grad()
def _compute_distributed_log_softmax(
    vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Non-differentiable variant of :func:`_compute_distributed_log_softmax_with_grad`.

    Same math and shape contract, wrapped in ``torch.no_grad()`` for the
    inference / logprob paths that don't need gradients.
    """
    return _compute_distributed_log_softmax_with_grad(vocab_parallel_logits, group)

@torch.no_grad()
def _compute_distributed_softmax(
    vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Compute a stable distributed softmax across tensor parallel workers.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L239

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_length, vocab_size//TP]
            where TP is the tensor parallel size.
        group (torch.distributed.ProcessGroup): Process group for the all-reduce operations.

    Returns:
        torch.Tensor: Softmax output with the same shape as input, normalized across the full vocabulary.
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    vocab_parallel_logits = vocab_parallel_logits - logits_max

    exp_logits = vocab_parallel_logits.exp_()

    sum_exp_logits = exp_logits.sum(-1, keepdim=True)
    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )
    exp_logits.div_(sum_exp_logits)

    return exp_logits

class DistributedCrossEntropy(torch.autograd.Function):
    """Compute soft-target cross entropy across TP-sharded vocab.

    This returns H(p_target, q_student), which matches forward KL up to the
    target entropy constant. Backward propagates only through student logits.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        student_logits: torch.Tensor,
        target_logits: torch.Tensor,
        group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        if student_logits.shape != target_logits.shape:
            raise ValueError(
                "student_logits and target_logits must have the same shape, "
                f"got {student_logits.shape} and {target_logits.shape}."
            )

        target_probs = _compute_distributed_softmax(
            target_logits.to(dtype=torch.float32),
            group=group,
        )
        student_log_probs = _compute_distributed_log_softmax(
            student_logits.to(dtype=torch.float32), group=group
        )
        # Reuse the log-softmax buffers to avoid extra full-vocab allocations.
        local_cross_entropy = torch.einsum(
            "...v,...v->...", target_probs, student_log_probs
        ).neg_()
        torch.distributed.all_reduce(
            local_cross_entropy,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        if not inference_only:
            student_probs = student_log_probs.exp_()
            ctx.save_for_backward(target_probs, student_probs)

        return local_cross_entropy.contiguous()

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        grad_output = grad_outputs[0]
        target_probs, student_probs = ctx.saved_tensors

        # d(H(p, q))/d(z_v) = q_v - p_v
        grad_student = (student_probs - target_probs) * grad_output.unsqueeze(-1)
        return grad_student, None, None, None

def allgather_cp_sharded_tensor(
    tensor, cp_group, seq_dim=1
):  # , unpadded_seqlen=None):
    return AllGatherCPTensor.apply(tensor, cp_group, seq_dim)  # , unpadded_seqlen)

class _AllReduceSum(torch.autograd.Function):
    """Autograd-aware SUM all-reduce primitive; forward clones + reduces, backward is identity.

    Math: ``y = Σ_r x_r`` ⇒ ``dy/dx_r = 1``. Every rank holds the same
    ``y`` after forward, so the upstream grad is identical across ranks
    — passing it back as ``grad_x_r`` lets the rank-local chain rule
    route gradients to each rank's own input shard with no cross-rank
    coupling.

    Use the functional wrappers :func:`group_all_reduce_sum_with_grad` (keeps
    the gradient) and :func:`group_all_reduce_sum` (no gradient) rather than
    calling ``.apply`` directly. A ``group`` of ``None`` (or world size <= 1,
    or an uninitialized process group) is a no-op so single-process / CPU paths
    work unchanged.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        x: torch.Tensor,
        group: Optional[torch.distributed.ProcessGroup],
    ) -> torch.Tensor:
        ctx.group = group
        if (
            group is None
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size(group) <= 1
        ):
            return x
        out = x.clone()
        torch.distributed.all_reduce(
            out, op=torch.distributed.ReduceOp.SUM, group=group
        )
        return out

    @staticmethod
    def backward(  # pyrefly: ignore[bad-override]
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return grad_out, None

def group_all_reduce_sum_with_grad(
    x: torch.Tensor,
    group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    """Differentiable SUM all-reduce of ``x`` over ``group``.

    Gradients flow back through the reduce (backward is identity), so this is
    the variant for forward-path reductions whose result feeds the loss — e.g.
    CP chunk-sum aggregation and TP projection-partial combination. ``group``
    of ``None`` (or world size <= 1, or dist uninitialized) is a no-op.
    """
    return _AllReduceSum.apply(x, group)

def group_all_reduce_sum(
    x: torch.Tensor,
    group: Optional[torch.distributed.ProcessGroup],
) -> torch.Tensor:
    """Non-differentiable variant of :func:`group_all_reduce_sum_with_grad`.

    Same reduction, but no gradient flows through it — use for normalizers /
    denominators that must be treated as constants (e.g. the global
    valid-token or valid-chunk counts). Pass an explicit ``group`` (e.g.
    ``torch.distributed.group.WORLD`` to reduce over the full DP×CP×TP mesh, or
    a narrower process group); ``None`` (or world size <= 1, or dist
    uninitialized) is a no-op that returns the local value.
    """
    with torch.no_grad():
        return _AllReduceSum.apply(x, group)

class AllGatherCPTensor(torch.autograd.Function):
    def forward(
        ctx, tensor, cp_group: torch.distributed.ProcessGroup, seq_dim=1
    ):  # , unpadded_seqlen: Optional[int] = None):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank_chunks = []
        for _ in range(cp_size):
            cp_rank_chunks.append(torch.empty_like(tensor))

        torch.distributed.all_gather(
            tensor_list=cp_rank_chunks, tensor=tensor, group=cp_group
        )

        # undo the CP load balancing chunking
        tensor_chunks = []
        for logit_chunk in cp_rank_chunks:
            tensor_chunks.extend(torch.chunk(logit_chunk, chunks=2, dim=seq_dim))

        chunk_indices = []
        for cp_rank in range(cp_size):
            chunk_indices.append(cp_rank)
            chunk_indices.append(2 * cp_size - cp_rank - 1)

        chunks_and_indices = list(zip(tensor_chunks, chunk_indices))
        chunks_and_indices = sorted(chunks_and_indices, key=lambda tup: tup[1])
        ret_tensor = [chunk for chunk, _ in chunks_and_indices]
        ret_tensor = torch.cat(ret_tensor, dim=seq_dim)

        ctx.seq_dim = seq_dim
        ctx.cp_group = cp_group
        # ctx.unpadded_seqlen = unpadded_seqlen

        return ret_tensor

    def backward(ctx, grad_output):
        cp_size = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        torch.distributed.all_reduce(grad_output, group=ctx.cp_group)

        # chunk the seqdim in 2*cp chunks, and select with a CP load balanced indexing
        seq_dim = ctx.seq_dim
        # if ctx.unpadded_seqlen is not None:
        # # Zero out grad_output along the seq_dim after unpadded_seqlen
        # slicer = [slice(None)] * grad_output.dim()
        # slicer[seq_dim] = slice(ctx.unpadded_seqlen, None)
        #     grad_output[tuple(slicer)] = 0

        grad_output = grad_output.view(
            *grad_output.shape[0:seq_dim],
            2 * cp_size,
            grad_output.shape[seq_dim] // (2 * cp_size),
            *grad_output.shape[(seq_dim + 1) :],
        )

        index = torch.tensor(
            [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
        ).cuda(non_blocking=True)

        grad_input = grad_output.index_select(seq_dim, index)
        grad_input = grad_input.view(
            *grad_input.shape[0:seq_dim], -1, *grad_input.shape[(seq_dim + 2) :]
        )

        return grad_input, None, None  # , None

def cp_load_balanced_to_contiguous(
    x: torch.Tensor,
    *,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Re-layout a tensor from load-balanced CP order to this rank's contiguous window.

    PyTorch ``context_parallel`` shards the sequence in a load-balanced
    (``2*cp`` interleaved) order. :func:`allgather_cp_sharded_tensor` undoes the
    chunking to the full contiguous sequence; this re-slices to this CP rank's
    contiguous ``[cp_rank*L, (cp_rank+1)*L)`` window. No-op when CP world <= 1.
    Uses the grad-preserving ``DTensor.to_local()`` so the gradient is kept.
    """
    if cp_group is None or torch.distributed.get_world_size(cp_group) <= 1:
        return x
    local = x.to_local() if isinstance(x, DTensor) else x
    full = allgather_cp_sharded_tensor(local, cp_group, seq_dim=seq_dim)
    cp_size = torch.distributed.get_world_size(cp_group)
    cp_rank = torch.distributed.get_rank(cp_group)
    local_len = full.shape[seq_dim] // cp_size
    return full.narrow(seq_dim, cp_rank * local_len, local_len).contiguous()

def cp_shift_next(
    x: torch.Tensor,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    *,
    fill: float,
) -> torch.Tensor:
    """Next-token (left-by-1) shift along seq dim 1, CP-aware for contiguous sharding.

    Returns ``out[:, t] = x[:, t + 1]`` over the *global* sequence: interior
    positions roll locally, each rank's last position takes the next CP rank's
    first row (contiguous CP sharding), and the global-last position is set to
    ``fill`` (no next token). With no / size-1 ``cp_group`` this is a plain local
    left-roll with the last position set to ``fill``.
    """
    out = torch.roll(x, shifts=-1, dims=1)
    cp_size = torch.distributed.get_world_size(cp_group) if cp_group is not None else 1
    if cp_size <= 1:
        out[:, -1] = fill
        return out
    cp_rank = torch.distributed.get_rank(cp_group)
    first = x[:, 0].contiguous()
    gathered = [torch.empty_like(first) for _ in range(cp_size)]
    torch.distributed.all_gather(gathered, first, group=cp_group)
    out[:, -1] = gathered[cp_rank + 1] if cp_rank < cp_size - 1 else fill
    return out

def vocab_parallel_log_softmax(
    logits: torch.Tensor,
    temperature: float,
    *,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """TP-aware ``log_softmax(logits / temperature)`` keeping the vocab shard.

    With ``tp_group`` world > 1 the logits are vocab-sharded, so the softmax
    normalization is reduced across the TP group (kept differentiable) while the
    result stays sharded on the same vocab axis. Otherwise this is a plain local
    ``log_softmax``. DTensor inputs are unwrapped to their local shard.
    """
    if isinstance(logits, DTensor):
        logits = logits.to_local()
    scaled = logits.float() / temperature
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        return _compute_distributed_log_softmax_with_grad(scaled, group=tp_group)
    return torch.log_softmax(scaled, dim=-1)

def vocab_parallel_full_log_softmax(
    logits: torch.Tensor,
    temperature: float,
    *,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """TP-aware ``log_softmax`` gathered to the full vocab ``[B, T, V]``.

    Unlike :func:`vocab_parallel_log_softmax` (which keeps the result
    vocab-sharded), callers that slice arbitrary vocab indices need the full
    vocab axis. With ``tp_group`` world > 1 the sharded log-probs are all-gathered
    via Megatron's autograd-aware gather; otherwise a plain local ``log_softmax``.
    """
    if isinstance(logits, DTensor):
        logits = logits.to_local()
    scaled = logits.float() / temperature
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        # Local import keeps the optional Megatron dependency boundary intact.
        from megatron.core.tensor_parallel import (
            gather_from_tensor_model_parallel_region,
        )

        sharded_log_probs = _compute_distributed_log_softmax_with_grad(
            scaled, group=tp_group
        )
        return gather_from_tensor_model_parallel_region(
            sharded_log_probs, group=tp_group
        )
    return torch.log_softmax(scaled, dim=-1)

def vocab_parallel_argmax(
    logits: torch.Tensor,
    *,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Per-position global argmax token id over the (possibly TP-sharded) vocab.

    Returns ``[B, T]`` global token ids. With ``tp_group`` world > 1 the vocab is
    sharded, so a distributed top-1 recovers the global argmax; otherwise a plain
    local ``argmax``. No gradient.
    """
    local_logits = logits.to_local() if isinstance(logits, DTensor) else logits
    tp_world = torch.distributed.get_world_size(tp_group) if tp_group is not None else 1
    if tp_world > 1:
        tp_rank = torch.distributed.get_rank(tp_group)
        local_vocab_size = int(local_logits.shape[-1])
        _, topk_global_idx = distributed_vocab_topk(
            local_logits,
            k=1,
            tp_group=tp_group,
            vocab_start_index=tp_rank * local_vocab_size,
            vocab_end_index=(tp_rank + 1) * local_vocab_size,
        )
        return topk_global_idx.squeeze(-1)
    return local_logits.argmax(dim=-1)

@torch.no_grad()
def distributed_vocab_topk(
    vocab_parallel_logits: torch.Tensor,
    k: int,
    tp_group: torch.distributed.ProcessGroup,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute global top-k over TP-sharded vocabulary logits.

    Args:
        vocab_parallel_logits: [B, S, V_local]
        k: number of top tokens to select globally
        tp_group: tensor-parallel process group
        vocab_start_index: global vocab start for this rank (inclusive)
        vocab_end_index: global vocab end for this rank (exclusive)
        chunk_size: optional chunk along sequence dim to bound memory

    Returns:
        topk_vals: [B, S, k]
        topk_global_indices: [B, S, k] (global token ids)
    """
    assert vocab_end_index > vocab_start_index
    world_size = torch.distributed.get_world_size(tp_group)

    B, S, V_local = vocab_parallel_logits.shape
    V_total = V_local * world_size
    K_eff = int(min(k, max(1, V_total)))

    if chunk_size is None:
        chunk_size = S

    vals_chunks: list[torch.Tensor] = []
    idx_chunks: list[torch.Tensor] = []

    for s0 in range(0, S, chunk_size):
        s1 = min(S, s0 + chunk_size)
        # Upcast only the current slice to bound the fp32 working set to
        # [B, chunk_size, V_local] (chunk_size driven by logprob_chunk_size).
        logits_chunk = vocab_parallel_logits[:, s0:s1, :].to(dtype=torch.float32)
        # local top-k on this TP rank
        local_vals, local_idx_local = torch.topk(logits_chunk, min(k, V_local), dim=-1)
        local_idx_global = local_idx_local + int(vocab_start_index)

        # gather candidates from all TP ranks
        gathered_vals = [torch.empty_like(local_vals) for _ in range(world_size)]
        gathered_idx = [torch.empty_like(local_idx_global) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_vals, local_vals, group=tp_group)
        torch.distributed.all_gather(gathered_idx, local_idx_global, group=tp_group)

        all_vals = torch.cat(gathered_vals, dim=-1)
        all_idx = torch.cat(gathered_idx, dim=-1)

        sel_vals, sel_pos = torch.topk(all_vals, K_eff, dim=-1)
        sel_idx = torch.gather(all_idx, dim=-1, index=sel_pos)

        vals_chunks.append(sel_vals)
        idx_chunks.append(sel_idx)

    topk_vals = (
        torch.cat(vals_chunks, dim=1) if len(vals_chunks) > 1 else vals_chunks[0]
    )
    topk_global_indices = (
        torch.cat(idx_chunks, dim=1) if len(idx_chunks) > 1 else idx_chunks[0]
    )

    return topk_vals, topk_global_indices


def to_local_if_dtensor(value):
    """Unwrap a DTensor without breaking its autograd graph."""
    return value.to_local() if isinstance(value, DTensor) else value
