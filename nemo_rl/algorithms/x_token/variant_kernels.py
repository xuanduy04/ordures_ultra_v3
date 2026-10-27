"""Opt-in algorithm kernels; legacy X-Token does not import this module.

DPCA follows ivanniu/On-Policy-Distill 927a8264 via the existing SimCT port.
SimCT exposes raw-logit author-code and native-logprob paper-score modes.
GRASS uses the exact output-head gradient Gram, not an identity surrogate.
See docs/guides/algorithm-ports.md for transport and alignment boundaries.
"""

import math

import torch


def selected_logprobs(logits, labels):
    if logits.ndim != 2 or labels.shape != logits.shape[:1]:
        raise ValueError("Expected [tokens,vocab] logits and [tokens] labels")
    return logits.log_softmax(-1).gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)


def validate_segments(segments, student_length, teacher_length):
    """Segments are half-open (teacher_start,end,student_start,end), response-only."""
    previous_t = previous_s = 0
    for ts, te, ss, se in segments:
        if not (previous_t <= ts < te <= teacher_length and previous_s <= ss < se <= student_length):
            raise ValueError("Segments must be ordered, nonoverlapping and in range")
        previous_t, previous_s = te, se


def dpca_advantages(teacher_chosen, behavior, segments, adv_clamp):
    """Semantic prior uses chunk totals before expanding to student tokens."""
    validate_segments(segments, behavior.numel(), teacher_chosen.numel())
    advantages = torch.zeros_like(behavior)
    covered = torch.zeros_like(behavior, dtype=torch.bool)
    for ts, te, ss, se in segments:
        teacher_sum = teacher_chosen[ts:te].sum().detach()
        prior = behavior[ss:se].detach()
        total = prior.sum()
        target = teacher_sum.expand_as(prior) / (se - ss) if total.abs() < 1e-8 else teacher_sum / total * prior
        advantages[ss:se] = (target - prior).clamp(-adv_clamp, adv_clamp)
        covered[ss:se] = True
    return advantages.detach(), covered


def dpca_objective(current, behavior, advantages, mask, config):
    if not (current.shape == behavior.shape == advantages.shape == mask.shape):
        raise ValueError("DPCA token arrays must have equal shape")
    if behavior.requires_grad or advantages.requires_grad:
        raise ValueError("DPCA behavior scores and advantages must be detached")
    if not torch.isfinite(current[mask]).all() or not torch.isfinite(behavior[mask]).all():
        raise ValueError("Nonfinite DPCA log probabilities")
    current = current.masked_fill(~mask, 0)
    behavior = behavior.masked_fill(~mask, 0)
    advantages = advantages.masked_fill(~mask, 0)
    delta = (current - behavior).clamp(-config["kl_clamp"], config["kl_clamp"])
    ratio = delta.exp()
    first = -advantages * ratio
    clipped = -advantages * ratio.clamp(1 - config["clip_ratio_low"], 1 + config["clip_ratio_high"])
    upper = torch.maximum(first, clipped)
    dual = torch.minimum(-advantages * config["clip_ratio_c"], upper)
    losses = torch.where(advantages < 0, dual, upper)
    return losses.masked_fill(~mask, 0).sum()


def shared_reverse_kl(student_logits, teacher_logits, student_ids, teacher_ids):
    """Conditional shared-vocabulary RKL (CTOPD); all shared scores are required."""
    if len(student_ids) == 0 or len(student_ids) != len(teacher_ids):
        raise ValueError("A nonempty paired shared vocabulary is required")
    s = student_logits[:, student_ids].log_softmax(-1)
    t = teacher_logits.detach()[:, teacher_ids].log_softmax(-1)
    return (s.exp() * (s - t)).sum(-1)


def simct_virtual_scores(student, teacher, student_labels, teacher_labels, segments,
                         student_ids, teacher_ids, score_mode):
    validate_segments(segments, len(student), len(teacher))
    if score_mode == "mean_logprob":
        student, teacher = student.log_softmax(-1), teacher.log_softmax(-1)
    elif score_mode != "raw_logit":
        raise ValueError("Use mean_logprob or raw_logit explicitly")
    teacher = teacher.detach()
    first_s = [ss for ts, te, ss, se in segments]
    first_t = [ts for ts, te, ss, se in segments]
    shared_s, shared_t = student[first_s][:, student_ids], teacher[first_t][:, teacher_ids]
    spans = [i for i, (ts, te, ss, se) in enumerate(segments) if max(te - ts, se - ss) > 1]
    if not spans:
        return shared_s, shared_t
    s_extra = student.new_full((len(segments), len(spans)), -1e9)
    t_extra = teacher.new_full((len(segments), len(spans)), -1e9)
    for column, row in enumerate(spans):
        ts, te, ss, se = segments[row]
        s_extra[row, column] = selected_logit_rows(student[ss:se], student_labels[ss:se]).mean()
        t_extra[row, column] = selected_logit_rows(teacher[ts:te], teacher_labels[ts:te]).mean()
    return torch.cat((shared_s, s_extra), -1), torch.cat((shared_t, t_extra), -1)


def selected_logit_rows(values, labels):
    return values.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)


def simct_objective(student, teacher, student_labels, teacher_labels, segments,
                    student_ids, teacher_ids, score_mode, gh_threshold):
    s, t = simct_virtual_scores(student, teacher, student_labels, teacher_labels,
        segments, student_ids, teacher_ids, score_mode)
    log_s, log_t = s.log_softmax(-1), t.log_softmax(-1)
    losses = (log_s.exp() * (log_s - log_t)).sum(-1)
    keep = torch.ones_like(losses, dtype=torch.bool)
    if gh_threshold > 0:
        teacher_logp = teacher.detach().log_softmax(-1)
        for row, (ts, te, ss, se) in enumerate(segments):
            mass = teacher_logp[ts, teacher_ids].exp().sum()
            if max(te - ts, se - ss) > 1:
                mass = mass + selected_logit_rows(teacher_logp[ts:te], teacher_labels[ts:te]).mean().exp()
            keep[row] = mass.clamp_max(1) >= 1 / gh_threshold
    return losses.masked_fill(~keep, 0).sum(), keep


def head_gram(hidden, probabilities, labels, atom_spans):
    """Exact Gram of atom NLL gradients w.r.t. a bias-free output-head matrix.

    Uses (h_i dot h_j) * ((p_i-e_yi) dot (p_j-e_yj)), avoiding [tokens,V,H].
    Geometry is a detached statistic; hidden is the actual input to the head.
    """
    h, p = hidden.detach().double(), probabilities.detach().double()
    if h.ndim != 2 or p.ndim != 2 or len(h) != len(p) or labels.shape != (len(h),):
        raise ValueError("Incompatible hidden/probability/label arrays")
    residual = p.clone()
    residual[torch.arange(len(p), device=p.device), labels] -= 1
    token_gram = (h @ h.T) * (residual @ residual.T)
    assignment = p.new_zeros((len(atom_spans), len(p)))
    previous = 0
    for index, (start, end) in enumerate(atom_spans):
        if not previous <= start < end <= len(p):
            raise ValueError("Invalid or overlapping atom span")
        assignment[index, start:end] = 1
        previous = end
    return assignment @ token_gram @ assignment.T


def grass_chunk_shrink(rate, weight, gram, partition, sigma2):
    """Fixed-partition GRASS; never search, merge or split chunks."""
    if not math.isfinite(sigma2) or sigma2 < 0:
        raise ValueError("sigma2 must be finite and nonnegative")
    r, w, g = rate.detach().double(), weight.detach().double(), gram.detach().double()
    if r.ndim != 1 or w.shape != r.shape or g.shape != (len(r), len(r)):
        raise ValueError("GRASS rate/weight/Gram shapes differ")
    if not torch.isfinite(r).all() or not torch.isfinite(g).all() or not (w > 0).all():
        raise ValueError("Invalid GRASS statistics")
    shrunk, alpha = r.clone(), r.new_zeros(len(partition))
    end_before = 0
    for index, (start, end) in enumerate(partition):
        if start != end_before or not start < end <= len(r):
            raise ValueError("GRASS partition must cover every atom once")
        end_before = end
        block, own_r, own_w = g[start:end, start:end], r[start:end], w[start:end]
        pooled = (own_r * own_w).sum() / own_w.sum()
        deviation = own_r - pooled
        distortion = deviation @ block @ deviation
        variance = sigma2 * ((block.diagonal() / own_w).sum() - block.sum() / own_w.sum())
        tolerance = 1e-6 * max(float(block.abs().sum()), 1e-30)
        if distortion < -tolerance or variance < -tolerance:
            raise ValueError("GRASS geometry is not positive-semidefinite within tolerance")
        d, v = distortion.clamp_min(0), variance.clamp_min(0)
        strength = (v / d).clamp(0, 1) if d > 0 else r.new_tensor(float(v > 0))
        alpha[index] = strength
        shrunk[start:end] = (1 - strength) * own_r + strength * pooled
        if not torch.allclose((own_w * shrunk[start:end]).sum(), (own_w * own_r).sum(), rtol=1e-6, atol=1e-10):
            raise ValueError("GRASS chunk credit conservation failed")
    if end_before != len(r):
        raise ValueError("GRASS partition does not cover all atoms")
    return shrunk, alpha
