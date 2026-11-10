"""Projected top-k plus OTHER-bucket KL; an approximation, not Gold/H-KL."""

import torch

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.algorithms.x_token.loss_utils import (
    get_sparse_projection_matrix,
    Fp32SparseMM,
    slice_sparse_projection_rows,
)
from nemo_rl.algorithms.x_token.parallel import (
    to_local_if_dtensor,
    vocab_parallel_log_softmax,
    group_all_reduce_sum,
    group_all_reduce_sum_with_grad,
)


def coarse_kl(student, teacher, *, reverse, epsilon):
    """KL on explicit buckets followed by their residual probability bucket."""
    student = student / student.sum(-1, keepdim=True).clamp_min(1.0)
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1.0)
    student = torch.cat([student, (1 - student.sum(-1, keepdim=True)).clamp_min(0)], -1)
    teacher = torch.cat([teacher, (1 - teacher.sum(-1, keepdim=True)).clamp_min(0)], -1)
    log_s, log_t = student.clamp_min(epsilon).log(), teacher.clamp_min(epsilon).log()
    return (student * (log_s - log_t)).sum(-1) if reverse else (teacher * (log_t - log_s)).sum(-1)


class ProjectedTopKTailLoss(LossFunction):
    """Average student probabilities per aligned chunk and score teacher rows.

    Each teacher position defines its own partition: returned, reachable token IDs and OTHER. Student probabilities are
    projected to the teacher vocabulary with row-normalized mapping weights. Missing student mappings and omitted
    teacher tokens remain in OTHER. Teacher rows are averaged per chunk; chunks are weighted by their number of student
    response tokens. This is a distinct objective from upstream chunk-averaged-log-probability P-KL/Gold/H-KL.
    """

    requires_raw_logits = True
    uses_topk_scores = True
    loss_type = LossType.TOKEN_LEVEL
    handles_sequence_packing = True
    cp_replicated_with_identity_grad = True

    def __init__(self, config):
        if config["temperature"] != 1:
            raise ValueError("Top-k tail KL requires temperature=1; hidden scores cannot be re-tempered")
        if config["gold_loss"] or config["xtoken_loss"]:
            raise ValueError("Top-k tail KL is a separate approximation, not Gold/H-KL")
        if config["kd_loss_mode"] != "sum" or config["dynamic_loss_scaling"] or config["ce_loss_scale"] != 0:
            raise ValueError("Top-k tail KL currently requires static sum aggregation and CE=0")
        if config.get("sum_weights_metric") is not None or config["normalize_teacher_by_vocab"]:
            raise ValueError("Top-k tail KL requires static teacher weights without vocabulary rescaling")
        if any(config[key][i] for key in ("teacher_gold_loss", "teacher_xtoken_loss") for i in range(len(config[key]))):
            raise ValueError("Per-teacher Gold/H-KL overrides are incompatible with top-k tail KL")
        self.cfg = config
        self.projection_matrix_paths = config["projection_matrix_paths"]
        self.num_teachers = len(self.projection_matrix_paths)
        self._projection_cache = {}
        self._projection_shard_cache = {}
        if not 0 < config["tail_epsilon"] < 1:
            raise ValueError("tail_epsilon must be in (0, 1)")
        if any(weight < 0 for weight in config["teacher_weights"]) or sum(config["teacher_weights"]) <= 0:
            raise ValueError("Teacher weights must be nonnegative with positive sum")

    def _projection(self, index, device):
        key = (index, device)
        if key not in self._projection_cache:
            matrix = get_sparse_projection_matrix(
                self.projection_matrix_paths[index],
                device,
                student_vocab_size=self.cfg["student_vocab_size"],
                teacher_vocab_size=self.cfg["teacher_vocab_sizes"][index],
            ).coalesce()
            if matrix.shape != (self.cfg["student_vocab_size"], self.cfg["teacher_vocab_sizes"][index]):
                raise ValueError("Projection IDs exceed the configured tokenizer vocabularies")
            ids, values = matrix.indices(), matrix.values()
            if not torch.isfinite(values).all() or (values < 0).any():
                raise ValueError("Projection weights must be finite and nonnegative")
            sums = torch.zeros(matrix.shape[0], device=device).scatter_add_(0, ids[0], values)
            if not torch.isfinite(sums).all():
                raise ValueError("Projection row sums must be finite")
            row_sums = sums[ids[0]]
            normalized = values / torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
            matrix = torch.sparse_coo_tensor(ids, normalized, matrix.shape, device=device).coalesce()
            reachable = torch.zeros(matrix.shape[1], dtype=torch.bool, device=device)
            reachable[ids[1][normalized > 0]] = True
            self._projection_cache[key] = matrix, reachable
        return self._projection_cache[key]

    @staticmethod
    def _cp_slice(tensor, cp_group):
        cp_size = torch.distributed.get_world_size(cp_group) if cp_group is not None else 1
        if cp_size == 1:
            return tensor
        if tensor.shape[-1] % (2 * cp_size):
            raise ValueError("Student sequence padding must be divisible by 2 * CP")
        rank = torch.distributed.get_rank(cp_group)
        blocks = tensor.reshape(*tensor.shape[:-1], 2 * cp_size, -1)
        return torch.cat([blocks[..., rank, :], blocks[..., 2 * cp_size - rank - 1, :]], dim=-1)

    def _student_layout(self, chunks, mask, data, logits, cp_group, packed_seq_params):
        """Map target metadata to native Megatron predictor order, before reduction."""
        cp_size = torch.distributed.get_world_size(cp_group) if cp_group is not None else 1
        chunk_count = chunks[1]
        chunk_ids = chunks[0]
        groups, valid = [], []
        if packed_seq_params is not None:
            cu = packed_seq_params.cu_seqlens_q_padded
            if cu is None:
                cu = packed_seq_params.cu_seqlens_q
            if len(cu) != len(chunk_ids) + 1 or logits.shape[0] != 1:
                raise ValueError("Packed student boundaries do not match the original batch")
            padded_lengths = (cu[1:] - cu[:-1]).tolist()
        else:
            if logits.shape[0] != len(chunk_ids):
                raise ValueError("Student logits batch differs from alignment batch")
            padded_lengths = [logits.shape[1] * cp_size] * len(chunk_ids)
        for b, padded in enumerate(padded_lengths):
            length = int(data["input_lengths"][b]) if packed_seq_params is not None else chunk_ids.shape[1]
            if not 1 <= length <= chunk_ids.shape[1] or padded < length:
                raise ValueError("Student logits/padding do not cover the sampled sequence")
            ids = torch.nn.functional.pad(chunk_ids[b, 1:length], (0, padded - length + 1), value=-1)
            weights = torch.nn.functional.pad(mask[b, 1:length], (0, padded - length + 1), value=0)
            ids = self._cp_slice(ids, cp_group)
            weights = self._cp_slice(weights, cp_group)
            if (ids >= chunk_count).any():
                raise ValueError("Invalid student chunk ID")
            groups.append(ids.clamp_min(0) + b * chunk_count)
            valid.append((ids >= 0) & (weights > 0))
        group_ids, keep = torch.cat(groups), torch.cat(valid)
        if group_ids.numel() != logits.shape[0] * logits.shape[1]:
            raise ValueError("Packed/CP metadata length differs from local student logits")
        return group_ids, keep

    def __call__(
        self,
        logits,
        data,
        global_valid_seqs,
        global_valid_toks,
        vocab_parallel_rank=None,
        vocab_parallel_group=None,
        context_parallel_group=None,
        packed_seq_params=None,
    ):
        local = to_local_if_dtensor(logits)
        with torch.autocast(device_type=local.device.type, enabled=False):
            return self._compute_float32(
                local,
                data,
                global_valid_seqs,
                global_valid_toks,
                vocab_parallel_rank,
                vocab_parallel_group,
                context_parallel_group,
                packed_seq_params,
            )

    def _compute_float32(
        self,
        logits,
        data,
        global_valid_seqs,
        global_valid_toks,
        vocab_parallel_rank=None,
        vocab_parallel_group=None,
        context_parallel_group=None,
        packed_seq_params=None,
    ):
        logits = to_local_if_dtensor(logits).float()
        device = logits.device
        tp_size = torch.distributed.get_world_size(vocab_parallel_group) if vocab_parallel_group is not None else 1
        local_vocab = logits.shape[-1]
        if local_vocab * tp_size < self.cfg["student_vocab_size"]:
            raise ValueError("Student logits do not cover the configured tokenizer vocabulary")
        if tp_size > 1:
            rank = torch.distributed.get_rank(vocab_parallel_group)
            if vocab_parallel_rank is not None and rank != vocab_parallel_rank:
                raise ValueError("Megatron vocabulary rank does not match its process group")
            vocab_start = rank * local_vocab
            global_ids = torch.arange(local_vocab, device=device) + vocab_start
            logits = logits.masked_fill(global_ids >= self.cfg["student_vocab_size"], -float("inf"))
            probabilities = vocab_parallel_log_softmax(logits, 1.0, tp_group=vocab_parallel_group).exp()
        else:
            vocab_start = 0
            probabilities = logits[..., : self.cfg["student_vocab_size"]].softmax(-1)
            local_vocab = probabilities.shape[-1]
        sample_mask = data["sample_mask"].to(device)
        mask = data["token_mask"].to(device)
        batch = len(sample_mask)
        denominator = global_valid_toks.to(device).clamp_min(1)
        total = probabilities.sum() * 0
        metrics = {}
        for index, path in enumerate(self.projection_matrix_paths):
            prefix = f"teacher_{index}_"
            ids = data[prefix + "topk_ids"].to(device)[:, :-1]
            logprobs = data[prefix + "topk_logprobs"].to(device)[:, :-1].detach()
            present = data[prefix + "topk_valid"].to(device)[:, :-1]
            if path is None:
                chunks = data["input_ids"].shape[1] - 1
                student_chunks = torch.arange(-1, chunks, device=device).expand(batch, -1)
                teacher_chunks = student_chunks[:, 1:]
                pair_valid = torch.ones(batch, chunks, dtype=torch.bool, device=device)
                teacher_mask = mask[:, 1:]
            else:
                student_chunks = data[f"alignment_{index}_student_chunk_id"].to(device)
                teacher_chunks = data[f"alignment_{index}_teacher_chunk_id"].to(device)[:, 1:]
                pair_valid = data[f"alignment_{index}_pair_valid"].to(device).bool()
                if self.cfg["exact_token_match_only"]:
                    pair_valid = pair_valid & data[f"alignment_{index}_pair_is_correct"].to(device).bool()
                chunks = pair_valid.shape[1]
                teacher_mask = data[prefix + "token_mask"].to(device)[:, 1:]
            if chunks == 0:
                continue
            s_index, s_valid = self._student_layout(
                (student_chunks, chunks), mask, data, logits, context_parallel_group, packed_seq_params
            )
            counts = torch.zeros(batch * chunks, device=device).scatter_add_(0, s_index, s_valid.float())
            counts = group_all_reduce_sum(counts, context_parallel_group)
            sums = torch.zeros(batch * chunks, local_vocab, device=device).index_add(
                0, s_index, probabilities.reshape(-1, local_vocab) * s_valid[:, None]
            )
            sums = group_all_reduce_sum_with_grad(sums, context_parallel_group)
            chunk_probabilities = sums / counts[:, None].clamp_min(1)
            if path is None and tp_size == 1:
                projected = chunk_probabilities
                reachable = torch.ones(local_vocab, dtype=torch.bool, device=device)
            else:
                if path is None:
                    width = self.cfg["student_vocab_size"]
                    diagonal = torch.arange(width, device=device)
                    matrix = torch.sparse_coo_tensor(
                        torch.stack([diagonal, diagonal]), torch.ones(width, device=device), (width, width)
                    ).coalesce()
                    reachable = torch.ones(width, dtype=torch.bool, device=device)
                else:
                    matrix, reachable = self._projection(index, device)
                if tp_size > 1:
                    key = (index, device, vocab_start, local_vocab)
                    if key not in self._projection_shard_cache:
                        self._projection_shard_cache[key] = slice_sparse_projection_rows(
                            matrix, vocab_start, vocab_start + local_vocab
                        )
                    matrix = self._projection_shard_cache[key]
                projected = Fp32SparseMM.apply(matrix, chunk_probabilities.t()).t()
                projected = group_all_reduce_sum_with_grad(projected, vocab_parallel_group)
            offsets = torch.arange(batch, device=device)[:, None] * chunks
            if (teacher_chunks >= chunks).any():
                raise ValueError("Invalid teacher chunk ID")
            t_index = (teacher_chunks.clamp_min(0) + offsets).clamp_max(batch * chunks - 1)
            t_valid = (teacher_chunks >= 0) & (teacher_chunks < chunks) & (teacher_mask > 0)
            t_valid &= pair_valid.flatten()[t_index] & (counts[t_index] > 0) & (sample_mask[:, None] > 0)
            rows = t_valid.nonzero(as_tuple=True)
            if rows[0].numel() == 0:
                continue
            group = t_index[rows]
            selected_ids = ids[rows]
            in_vocab = (selected_ids >= 0) & (selected_ids < projected.shape[-1])
            safe_ids = selected_ids.clamp(0, projected.shape[-1] - 1)
            keep = present[rows] & in_vocab & reachable[safe_ids]
            teacher_p = logprobs[rows].exp() * keep
            student_p = projected[group[:, None], safe_ids] * keep
            row_loss = coarse_kl(student_p, teacher_p, reverse=self.cfg["reverse_kl"], epsilon=self.cfg["tail_epsilon"])
            t_counts = torch.zeros(batch * chunks, device=device).scatter_add_(0, group, torch.ones_like(row_loss))
            weights = counts[group] / t_counts[group].clamp_min(1) * sample_mask[rows[0]]
            value = (row_loss * weights).sum() / denominator
            total = total + self.cfg["teacher_weights"][index] * value
            metrics[f"teacher_retained_mass_t{index}"] = teacher_p.sum(-1).mean().item()
            metrics[f"student_retained_mass_t{index}"] = student_p.sum(-1).mean().item()
            metrics[f"valid_teacher_positions_t{index}"] = rows[0].numel()
        total = self.cfg["kl_loss_weight"] * total
        metrics.update(
            {
                "loss": total.detach().item(),
                "kl_loss": total.detach().item(),
                "ce_loss": 0.0,
                "kl_loss_scale": 1.0,
                "num_valid_samples": batch,
            }
        )
        return total, metrics
