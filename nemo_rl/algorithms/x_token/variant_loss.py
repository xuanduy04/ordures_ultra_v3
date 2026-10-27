"""Remote-teacher algorithm adapters on the existing response-only Gym batch."""

import torch

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.algorithms.x_token.parallel import to_local_if_dtensor
from nemo_rl.algorithms.x_token.variant_kernels import (
    dpca_advantages, dpca_objective, grass_chunk_shrink, head_gram,
    selected_logprobs, selected_logit_rows, shared_reverse_kl, simct_objective,
)


MODES = ("dpca", "ctopd_full", "ctopd_topk", "simct_full", "simct_topk", "grass_chunk_fixed_sigma")


def configure_variant(config, teachers, student_tokenizer, teacher_tokenizers):
    """Fail before model loading when a selected mode lacks required scores."""
    variant = config["variant"]
    mode = variant["mode"]
    if mode not in MODES or len(teachers) != 1:
        raise ValueError("Variant modes require one teacher and a recognized explicit mode")
    if teachers[0].remote is None:
        raise ValueError("New variant adapters currently require a remote teacher; legacy local IPC modes remain available")
    score_mode = teachers[0].remote["score_mode"]
    if mode.endswith("_full") and score_mode != "full":
        raise ValueError("Exact shared/virtual vocabulary requires full teacher scores; use the separately named _topk approximation or a full-score endpoint")
    if mode.endswith("_topk") and score_mode != "topk":
        raise ValueError("The explicit top-k variant requires top-k transport")
    if config["temperature"] != 1 or config["ce_loss_scale"] != 0 or config["dynamic_loss_scaling"] or config["kd_loss_mode"] != "sum":
        raise ValueError("Initial variant adapters require T=1, CE=0, static single-teacher sum")
    if config["gold_loss"] or config["xtoken_loss"] or config["normalize_teacher_by_vocab"]:
        raise ValueError("Variant modes do not combine legacy Gold/H-KL or vocabulary rescaling flags")
    variant["score_mode"] = score_mode
    if mode.startswith(("ctopd", "simct")):
        from nemo_rl.algorithms.x_token.variant_alignment import shared_ids

        variant["student_shared_ids"], variant["teacher_shared_ids"] = shared_ids(student_tokenizer, teacher_tokenizers[0])
    return config


def segments_from_ids(student_chunks, teacher_chunks, valid):
    segments, atom_ids = [], []
    for atom in range(valid.numel()):
        if not bool(valid[atom]):
            continue
        s, t = (student_chunks == atom).nonzero().flatten(), (teacher_chunks == atom).nonzero().flatten()
        if not len(s) or not len(t):
            continue
        if len(s) != int(s[-1] - s[0] + 1) or len(t) != int(t[-1] - t[0] + 1):
            raise ValueError("Alignment atom is not contiguous")
        segments.append((int(t[0]), int(t[-1]) + 1, int(s[0]), int(s[-1]) + 1))
        atom_ids.append(atom)
    return segments, atom_ids


class VariantLoss(LossFunction):
    requires_raw_logits = True
    loss_type = LossType.TOKEN_LEVEL

    def __init__(self, config):
        self.cfg, self.variant = config, config["variant"]
        self.mode = self.variant["mode"]
        if self.mode not in MODES:
            raise ValueError("Unknown variant mode")
        self.projection_matrix_paths = config["projection_matrix_paths"]
        self.num_teachers = len(self.projection_matrix_paths)
        if self.num_teachers != 1:
            raise ValueError("Initial variant adapters require one teacher")
        self.uses_topk_scores = self.variant["score_mode"] == "topk"
        self.requires_head_inputs = self.mode == "grass_chunk_fixed_sigma"

    def _teacher(self, data, row, teacher_mask, labels, device):
        if self.uses_topk_scores:
            ids = data["teacher_0_topk_ids"][row, :-1].to(device)[teacher_mask]
            scores = data["teacher_0_topk_logprobs"][row, :-1].to(device)[teacher_mask].detach()
            valid = data["teacher_0_topk_valid"][row, :-1].to(device)[teacher_mask].bool()
            chosen = valid & (ids == labels[:, None])
            if not (chosen.sum(-1) == 1).all():
                raise ValueError("Teacher top-k response must include each actual prompt token exactly once")
            chosen_scores = scores.masked_fill(~chosen, 0).sum(-1)
            return chosen_scores, (ids, scores, valid), None
        logits = data["teacher_0_full_logits"][row, :-1].to(device)[teacher_mask].float().detach()
        return selected_logprobs(logits, labels), None, logits

    def __call__(self, logits, data, global_valid_seqs, global_valid_toks):
        logits = to_local_if_dtensor(logits).float()
        total = logits.sum() * 0
        metrics = {"variant_segments": 0, "variant_kept_segments": 0, "variant_alpha_sum": 0., "variant_alpha_count": 0}
        for row in range(len(logits)):
            sample_weight = data["sample_mask"][row].to(logits.device)
            if sample_weight <= 0:
                continue
            mask = data["token_mask"][row, 1:].to(logits.device).bool()
            if not mask.any():
                continue
            s_logits = logits[row, :-1][mask]
            s_labels = data["input_ids"][row, 1:].to(logits.device)[mask]
            current = selected_logprobs(s_logits, s_labels)
            same = self.projection_matrix_paths[0] is None
            teacher_mask = mask if same else data["teacher_0_token_mask"][row, 1:].to(logits.device).bool()
            teacher_labels = s_labels if same else data["teacher_0_input_ids"][row, 1:].to(logits.device)[teacher_mask]
            chosen, topk, teacher_logits = self._teacher(data, row, teacher_mask, teacher_labels, logits.device)
            if same:
                segments = [(i, i + 1, i, i + 1) for i in range(len(current))]
                atom_ids = list(range(len(segments)))
            else:
                segments, atom_ids = segments_from_ids(
                    data["alignment_0_student_chunk_id"][row, 1:].to(logits.device)[mask],
                    data["alignment_0_teacher_chunk_id"][row, 1:].to(logits.device)[teacher_mask],
                    data["alignment_0_pair_valid"][row].to(logits.device))
            metrics["variant_segments"] += len(segments)
            if not segments:
                continue
            if self.mode == "dpca":
                if "behavior_logprobs" not in data:
                    raise ValueError("DPCA requires actual sampled-token behavior logprobs from rollout")
                behavior = data["behavior_logprobs"][row, 1:].to(logits.device)[mask].detach()
                advantages, covered = dpca_advantages(chosen, behavior, segments, self.variant["adv_clamp"])
                value = dpca_objective(current, behavior, advantages, covered, self.variant)
                metrics["variant_kept_segments"] += len(segments)
            elif self.mode == "grass_chunk_fixed_sigma":
                spans = [(ss, se) for ts, te, ss, se in segments]
                weights = current.new_tensor([se - ss for ss, se in spans])
                nll = torch.stack([-current[ss:se].sum() for ss, se in spans])
                teacher_scores = torch.stack([chosen[ts:te].sum() for ts, te, ss, se in segments])
                rate = (teacher_scores + nll.detach()) / weights
                hidden = data["student_head_inputs"][row, :-1].to(logits.device)[mask]
                partition, start = [], 0
                groups = None if same else data["alignment_0_native_group_id"][row].tolist()
                for i in range(1, len(spans) + 1):
                    if i == len(spans) or groups is None or groups[atom_ids[i]] < 0 or groups[atom_ids[i]] != groups[atom_ids[i - 1]]:
                        partition.append((start, i))
                        start = i
                gram = torch.zeros((len(spans), len(spans)), device=logits.device, dtype=torch.float64)
                for first, last in partition:
                    low, high = spans[first][0], spans[last - 1][1]
                    local_spans = [(a - low, b - low) for a, b in spans[first:last]]
                    gram[first:last, first:last] = head_gram(hidden[low:high], s_logits[low:high].softmax(-1), s_labels[low:high], local_spans)
                shrunk, alpha = grass_chunk_shrink(rate, weights, gram, partition, self.variant["sigma2"])
                value = (shrunk.to(nll.dtype) * nll).sum()
                metrics["variant_kept_segments"] += len(segments)
                metrics["variant_alpha_sum"] += float(alpha.sum())
                metrics["variant_alpha_count"] += len(alpha)
            elif self.mode == "ctopd_full":
                paired = [(ts, ss) for ts, te, ss, se in segments if te - ts == se - ss == 1]
                if not paired:
                    continue
                t_rows, s_rows = zip(*paired)
                value = shared_reverse_kl(s_logits[list(s_rows)], teacher_logits[list(t_rows)],
                    self.variant["student_shared_ids"], self.variant["teacher_shared_ids"]).sum()
                metrics["variant_kept_segments"] += len(paired)
            elif self.mode == "simct_full":
                value, keep = simct_objective(s_logits, teacher_logits, s_labels, teacher_labels,
                    segments, self.variant["student_shared_ids"], self.variant["teacher_shared_ids"],
                    "mean_logprob", self.variant["gh_threshold"])
                metrics["variant_kept_segments"] += int(keep.sum())
            else:
                value, kept = self._topk_virtual(s_logits, s_labels, chosen, topk, segments)
                metrics["variant_kept_segments"] += kept
            total = total + sample_weight * value
        total = self.cfg["kl_loss_weight"] * self.cfg["teacher_weights"][0] * total / global_valid_toks.to(logits.device).clamp_min(1)
        metrics.update(loss=float(total.detach()), kl_loss=float(total.detach()), ce_loss=0., kl_loss_scale=1.)
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite variant objective")
        return total, metrics

    def _topk_virtual(self, student_logits, labels, chosen, topk, segments):
        ids, scores, valid = topk
        shared = dict(zip(self.variant["teacher_shared_ids"], self.variant["student_shared_ids"]))
        log_s = student_logits.log_softmax(-1)
        total, kept = student_logits.sum() * 0, 0
        for ts, te, ss, se in segments:
            span = max(te - ts, se - ss) > 1
            if self.mode == "ctopd_topk" and span:
                continue
            positions = [i for i, token in enumerate(ids[ts].tolist()) if bool(valid[ts, i]) and token in shared]
            s_ids = [shared[int(ids[ts, i])] for i in positions]
            s = log_s[ss, s_ids]
            t = scores[ts, positions]
            if span:
                s = torch.cat((s, selected_logit_rows(log_s[ss:se], labels[ss:se]).mean().reshape(1)))
                t = torch.cat((t, chosen[ts:te].mean().reshape(1)))
            if not len(s):
                continue
            if self.mode == "simct_topk" and self.variant["gh_threshold"] > 0 and t.exp().sum().clamp_max(1) < 1 / self.variant["gh_threshold"]:
                continue
            s_norm, t_norm = s.log_softmax(-1), t.log_softmax(-1)
            total = total + (s_norm.exp() * (s_norm - t_norm)).sum()
            kept += 1
        return total, kept
