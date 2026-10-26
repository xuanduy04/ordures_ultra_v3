"""Projected top-k plus OTHER-bucket KL; an approximation, not Gold/H-KL."""

import torch

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.algorithms.x_token.loss_utils import get_sparse_projection_matrix, Fp32SparseMM
from nemo_rl.algorithms.x_token.parallel import to_local_if_dtensor


def coarse_kl(student, teacher, *, reverse, epsilon):
    """KL on explicit buckets followed by their residual probability bucket."""
    student = student / student.sum(-1, keepdim=True).clamp_min(1.)
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1.)
    student = torch.cat([student, (1 - student.sum(-1, keepdim=True)).clamp_min(0)], -1)
    teacher = torch.cat([teacher, (1 - teacher.sum(-1, keepdim=True)).clamp_min(0)], -1)
    log_s, log_t = student.clamp_min(epsilon).log(), teacher.clamp_min(epsilon).log()
    return (student * (log_s - log_t)).sum(-1) if reverse else (teacher * (log_t - log_s)).sum(-1)


class ProjectedTopKTailLoss(LossFunction):
    """Average student probabilities per aligned chunk and score teacher rows.

    Each teacher position defines its own partition: returned, reachable token
    IDs and OTHER. Student probabilities are projected to the teacher vocabulary
    with row-normalized mapping weights. Missing student mappings and omitted
    teacher tokens remain in OTHER. Teacher rows are averaged per chunk; chunks
    are weighted by their number of student response tokens. This is a distinct
    objective from upstream chunk-averaged-log-probability P-KL/Gold/H-KL.
    """

    requires_raw_logits = True
    uses_topk_scores = True
    loss_type = LossType.TOKEN_LEVEL

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
        if not 0 < config["tail_epsilon"] < 1:
            raise ValueError("tail_epsilon must be in (0, 1)")
        if any(weight < 0 for weight in config["teacher_weights"]) or sum(config["teacher_weights"]) <= 0:
            raise ValueError("Teacher weights must be nonnegative with positive sum")

    def _projection(self, index, device):
        key = (index, device)
        if key not in self._projection_cache:
            matrix = get_sparse_projection_matrix(
                self.projection_matrix_paths[index], device,
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

    def __call__(self, logits, data, global_valid_seqs, global_valid_toks):
        local = to_local_if_dtensor(logits)
        with torch.autocast(device_type=local.device.type, enabled=False):
            return self._compute_float32(local, data, global_valid_seqs, global_valid_toks)

    def _compute_float32(self, logits, data, global_valid_seqs, global_valid_toks):
        logits = to_local_if_dtensor(logits).float()[..., :self.cfg["student_vocab_size"]]
        device = logits.device
        probabilities = logits[:, :-1].softmax(-1)
        mask = data["token_mask"][:, 1:].to(device)
        sample_mask = data["sample_mask"].to(device)
        denominator = global_valid_toks.to(device).clamp_min(1)
        total = probabilities.sum() * 0
        metrics = {}
        for index, path in enumerate(self.projection_matrix_paths):
            prefix = f"teacher_{index}_"
            ids = data[prefix + "topk_ids"].to(device)[:, :-1]
            logprobs = data[prefix + "topk_logprobs"].to(device)[:, :-1].detach()
            present = data[prefix + "topk_valid"].to(device)[:, :-1]
            batch, student_length, vocab = probabilities.shape
            if path is None:
                chunks = student_length
                student_chunks = torch.arange(chunks, device=device).expand(batch, -1)
                teacher_chunks = student_chunks
                pair_valid = torch.ones(batch, chunks, dtype=torch.bool, device=device)
                teacher_mask = mask
            else:
                student_chunks = data[f"alignment_{index}_student_chunk_id"].to(device)[:, 1:]
                teacher_chunks = data[f"alignment_{index}_teacher_chunk_id"].to(device)[:, 1:]
                pair_valid = data[f"alignment_{index}_pair_valid"].to(device).bool()
                if self.cfg["exact_token_match_only"]:
                    pair_valid = pair_valid & data[f"alignment_{index}_pair_is_correct"].to(device).bool()
                chunks = pair_valid.shape[1]
                teacher_mask = data[prefix + "token_mask"].to(device)[:, 1:]
            if chunks == 0:
                continue
            offsets = torch.arange(batch, device=device)[:, None] * chunks
            s_index = (student_chunks.clamp_min(0) + offsets).flatten()
            s_valid = (student_chunks >= 0) & (student_chunks < chunks) & (mask > 0)
            if (student_chunks >= chunks).any() or (teacher_chunks >= chunks).any():
                raise ValueError("Invalid student chunk ID")
            s_index = s_index.clamp_max(batch * chunks - 1)
            counts = torch.zeros(batch * chunks, device=device).scatter_add_(0, s_index, s_valid.float().flatten())
            chunk_probabilities = torch.zeros(batch * chunks, vocab, device=device).index_add(
                0, s_index, (probabilities * s_valid[..., None]).reshape(-1, vocab))
            chunk_probabilities = chunk_probabilities / counts[:, None].clamp_min(1)
            if path is None:
                projected = chunk_probabilities
                reachable = torch.ones(vocab, dtype=torch.bool, device=device)
            else:
                matrix, reachable = self._projection(index, device)
                projected = Fp32SparseMM.apply(matrix, chunk_probabilities.t()).t()
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
            row_loss = coarse_kl(student_p, teacher_p,
                                 reverse=self.cfg["reverse_kl"], epsilon=self.cfg["tail_epsilon"])
            t_counts = torch.zeros(batch * chunks, device=device).scatter_add_(0, group, torch.ones_like(row_loss))
            weights = counts[group] / t_counts[group].clamp_min(1) * sample_mask[rows[0]]
            value = (row_loss * weights).sum() / denominator
            total = total + self.cfg["teacher_weights"][index] * value
            metrics[f"teacher_retained_mass_t{index}"] = teacher_p.sum(-1).mean().item()
            metrics[f"student_retained_mass_t{index}"] = student_p.sum(-1).mean().item()
            metrics[f"valid_teacher_positions_t{index}"] = rows[0].numel()
        total = self.cfg["kl_loss_weight"] * total
        metrics.update({"loss": total.detach().item(), "kl_loss": total.detach().item(),
                        "ce_loss": 0., "kl_loss_scale": 1., "num_valid_samples": logits.shape[0]})
        return total, metrics
