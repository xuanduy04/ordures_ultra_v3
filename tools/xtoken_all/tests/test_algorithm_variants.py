"""Algorithm/gradient checks; Torch tests execute in the existing remote runtime."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from nemo_rl.algorithms.x_token import variant_kernels as kernels
from nemo_rl.algorithms.x_token.variant_alignment import synchronized_segments
from nemo_rl.algorithms.x_token.variant_loss import VariantLoss


def reference(name):
    source = (ROOT / "tools/xtoken_all/tests/fixtures/variants" / (name + ".py.txt")).read_text(encoding="utf-8")
    tree = ast.parse(source)
    namespace = {"torch": torch, "dataclass": dataclass, "__name__": __name__}
    exec(compile(tree, "<pinned-reference-" + name + ">", "exec"), namespace)
    return namespace


def test_dpca_matches_existing_author_port_values_and_gradients():
    ref = reference("dpca")
    teacher = torch.tensor([-2., -4.])
    prior = torch.tensor([-1., -2., -1., -2., -3.])
    segments = [(0, 1, 0, 2), (1, 2, 2, 5)]
    advantage, mask = kernels.dpca_advantages(teacher, prior, segments, 10.)
    expected_adv = ref["dpca_atom_advantages"](teacher, torch.tensor([2, 3]), prior, 10.)
    torch.testing.assert_close(advantage, expected_adv)
    current = (prior + torch.tensor([.2, -.3, .4, 1., -2.])).requires_grad_()
    expected_current = current.detach().clone().requires_grad_()
    config = dict(clip_ratio_low=.2, clip_ratio_high=.28, clip_ratio_c=10., kl_clamp=20.)
    actual = kernels.dpca_objective(current, prior, advantage, mask, config) / mask.sum()
    expected, _metrics = ref["dpca_policy_loss"](prior, expected_current, advantage, mask.float(), ref["DPCAConfig"]())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, current)[0], torch.autograd.grad(expected, expected_current)[0])


def test_simct_builder_matches_pinned_author_code_in_both_score_modes():
    builder = reference("simct")["_build_virtual_vocab_logits"]
    torch.manual_seed(17)
    student, teacher = torch.randn(4, 7, requires_grad=True), torch.randn(3, 8)
    segments = [(0, 1, 0, 2), (1, 3, 2, 4)]
    s_labels, t_labels = torch.tensor([0, 1, 2, 3]), torch.tensor([2, 3, 4])
    for mode in ("raw_logit", "mean_logprob"):
        owner = SimpleNamespace(args=SimpleNamespace(kd=SimpleNamespace(span_score_mode=mode)),
            student_overlap_token_ids=torch.tensor([0, 1, 3]), teacher_overlap_token_ids=torch.tensor([1, 4, 6]),
            _span_debug_count=3)
        expected = builder(owner, segments, student, teacher, s_labels.tolist(), t_labels.tolist())
        actual = kernels.simct_virtual_scores(student, teacher, s_labels, t_labels,
            segments, [0, 1, 3], [1, 4, 6], mode)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right)
        torch.testing.assert_close(torch.autograd.grad(actual[0].sum(), student, retain_graph=True)[0],
            torch.autograd.grad(expected[0].sum(), student, retain_graph=True)[0])


def test_dpca_chunk_budget_is_distributed_before_token_expansion():
    teacher = torch.tensor([-1., -2., -4.])
    behavior = torch.tensor([-1., -3., -2., -2.])
    advantage, mask = kernels.dpca_advantages(teacher, behavior, [(0, 2, 0, 2), (2, 3, 2, 4)], 10.)
    assert torch.allclose(advantage, torch.tensor([.25, .75, 0., 0.]))
    assert mask.all() and not advantage.requires_grad


@pytest.mark.parametrize("delta", [-30., -2., 0., .3, 3., 30.])
def test_dpca_clipping_and_gradient_match_literal_upstream_equations(delta):
    behavior = torch.tensor([-2., -3.], dtype=torch.float64)
    current = (behavior + delta).requires_grad_()
    advantage = torch.tensor([-2., 1.], dtype=torch.float64)
    config = dict(kl_clamp=20., clip_ratio_low=.2, clip_ratio_high=.28, clip_ratio_c=10.)
    actual = kernels.dpca_objective(current, behavior, advantage, torch.ones(2, dtype=torch.bool), config)
    reference_current = current.detach().clone().requires_grad_()
    ratio = (reference_current - behavior).clamp(-20, 20).exp()
    upper = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(.8, 1.28))
    reference = torch.where(advantage < 0, torch.minimum(-advantage * 10, upper), upper).sum()
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, current)[0], torch.autograd.grad(reference, reference_current)[0])


def test_dpca_masks_nonfinite_padding_before_exponentiation():
    values = torch.tensor([-2., float("nan")], requires_grad=True)
    value = kernels.dpca_objective(values, torch.tensor([-2., float("nan")]),
        torch.tensor([1., float("nan")]), torch.tensor([True, False]),
        dict(kl_clamp=20., clip_ratio_low=.2, clip_ratio_high=.28, clip_ratio_c=10.))
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(values.grad).all()
    assert values.grad[1] == 0


def test_dpca_degenerate_chunk_preserves_teacher_budget():
    teacher = torch.tensor([-6.])
    prior = torch.zeros(3)
    advantage, mask = kernels.dpca_advantages(teacher, prior, [(0, 1, 0, 3)], 10.)
    assert advantage.sum() == teacher.sum()
    assert mask.all()


def test_behavior_scores_cannot_be_current_logits_with_a_live_gradient():
    prior = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        kernels.dpca_objective(prior, prior, prior.detach(), torch.ones(2, dtype=torch.bool), {})


def test_shared_rkl_uses_corresponding_ids_and_stops_teacher_gradient():
    s = torch.tensor([[1., 2., 5., -2.]], requires_grad=True)
    t = torch.tensor([[3., -1., 0.]], requires_grad=True)
    loss = kernels.shared_reverse_kl(s, t, [2, 0], [0, 2]).sum()
    loss.backward()
    assert t.grad is None and s.grad[0, 1] == s.grad[0, 3] == 0
    assert s.grad[0, 0] != 0


def test_simct_span_scores_match_released_virtual_vocabulary_definition():
    torch.manual_seed(3)
    student, teacher = torch.randn(4, 7, dtype=torch.float64), torch.randn(3, 9, dtype=torch.float64)
    s_ids, t_ids = [0, 2, 6], [1, 5, 8]
    s_labels, t_labels = torch.tensor([0, 1, 2, 3]), torch.tensor([3, 4, 5])
    segments = [(0, 1, 0, 2), (1, 3, 2, 4)]
    actual_s, actual_t = kernels.simct_virtual_scores(student, teacher, s_labels, t_labels,
        segments, s_ids, t_ids, "mean_logprob")
    s, t = student.log_softmax(-1), teacher.log_softmax(-1)
    expected_s = torch.cat((s[[0, 2]][:, s_ids], s.new_full((2, 2), -1e9)), -1)
    expected_t = torch.cat((t[[0, 1]][:, t_ids], t.new_full((2, 2), -1e9)), -1)
    expected_s[0, 3] = (s[0, 0] + s[1, 1]) / 2
    expected_s[1, 4] = (s[2, 2] + s[3, 3]) / 2
    expected_t[0, 3] = t[0, 3]
    expected_t[1, 4] = (t[1, 4] + t[2, 5]) / 2
    torch.testing.assert_close(actual_s, expected_s)
    torch.testing.assert_close(actual_t, expected_t)


def test_simct_paper_scores_are_invariant_to_native_row_gauges():
    s, t = torch.randn(2, 5), torch.randn(1, 6)
    args = (torch.tensor([1, 2]), torch.tensor([3]), [(0, 1, 0, 2)], [0, 4], [1, 5])
    a = kernels.simct_virtual_scores(s, t, *args, "mean_logprob")
    b = kernels.simct_virtual_scores(s + torch.tensor([[10.], [-8.]]), t + 7, *args, "mean_logprob")
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)


def test_simct_gh_threshold_masks_low_teacher_candidate_mass():
    s, t = torch.zeros(1, 4), torch.tensor([[10., -10., -10., -10.]])
    loss, keep = kernels.simct_objective(s, t, torch.tensor([1]), torch.tensor([1]),
        [(0, 1, 0, 1)], [1, 2], [1, 2], "mean_logprob", 2.)
    assert not keep.any() and loss == 0


@pytest.mark.parametrize("bias", [False, True])
def test_output_head_gram_matches_explicit_autograd_gradient_vectors(bias):
    torch.manual_seed(4)
    hidden = torch.randn(5, 3, dtype=torch.float64)
    if bias:
        hidden = torch.cat((hidden, torch.ones(5, 1, dtype=torch.float64)), -1)
    weight = torch.randn(7, hidden.shape[-1], dtype=torch.float64, requires_grad=True)
    logits = hidden @ weight.T
    labels = torch.tensor([1, 2, 0, 6, 1])
    spans = [(0, 2), (2, 3), (3, 5)]
    nll = -kernels.selected_logprobs(logits, labels)
    vectors = torch.stack([torch.autograd.grad(nll[a:b].sum(), weight, retain_graph=True)[0].flatten() for a, b in spans])
    actual = kernels.head_gram(hidden, logits.softmax(-1), labels, spans)
    torch.testing.assert_close(actual, vectors @ vectors.T, rtol=1e-10, atol=1e-10)


def test_grass_fixed_partition_matches_matrix_risk_and_conserves_each_chunk():
    r, w = torch.tensor([-3., 2., .2, 1.], dtype=torch.float64), torch.tensor([1., 2., 3., 1.], dtype=torch.float64)
    basis = torch.tensor([[1., 0.], [2., 1.], [1., 3.], [0., 2.]], dtype=torch.float64)
    gram = basis @ basis.T
    result, alpha = kernels.grass_chunk_shrink(r, w, gram, [(0, 3), (3, 4)], .3)
    for ci, (start, end) in enumerate([(0, 3), (3, 4)]):
        weight, rate, block = w[start:end], r[start:end], gram[start:end, start:end]
        p = torch.ones(len(rate), 1, dtype=torch.float64) @ (weight / weight.sum()).unsqueeze(0)
        a = torch.eye(len(rate), dtype=torch.float64) - p
        sigma = .3 * torch.diag(1 / weight)
        d = rate @ a.T @ block @ a @ rate
        v = torch.trace(block @ a @ sigma @ a.T)
        expected = (v / d).clamp(0, 1) if d > 1e-20 else float(v > 1e-20)
        assert abs(float(alpha[ci]) - float(expected)) < 1e-8
        torch.testing.assert_close((weight * result[start:end]).sum(), (weight * rate).sum())
    assert alpha[1] == 0


@pytest.mark.parametrize("bad", [[(1, 3)], [(0, 2), (1, 3)], [(0, 2)]])
def test_grass_rejects_gaps_overlap_and_partial_partitions(bad):
    with pytest.raises(ValueError, match="partition"):
        kernels.grass_chunk_shrink(torch.ones(3), torch.ones(3), torch.eye(3), bad, .1)


def test_byte_sync_preserves_utf8_whitespace_and_minimal_boundaries():
    assert synchronized_segments([b" a", b"b", b"\xc3", b"\xa9"], [b" ", b"ab", b"\xc3\xa9"]) == [(0, 2, 0, 2), (2, 3, 2, 4)]
    with pytest.raises(ValueError, match="bytes differ"):
        synchronized_segments([b" a"], [b"a"])


def fixture_batch(mode="dpca", score_mode="topk", batch=2, device="cpu"):
    torch.manual_seed(5)
    logits = torch.randn(batch, 5, 6, device=device, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 3, 4]] * batch, device=device)
    teacher_logits = torch.randn(batch, 5, 6, device=device)
    logp = teacher_logits.log_softmax(-1)
    ids = torch.arange(6, device=device).expand(batch, 5, -1)
    mask = torch.tensor([[0, 0, 1, 1, 1]] * batch, device=device)
    behavior = torch.zeros(batch, 5, device=device)
    behavior[:, 1:] = logits.detach()[:, :-1].log_softmax(-1).gather(-1, labels[:, 1:, None]).squeeze(-1)
    data = {"input_ids": labels, "token_mask": mask, "sample_mask": torch.ones(batch, device=device),
        "behavior_logprobs": behavior, "teacher_0_topk_ids": ids,
        "teacher_0_topk_logprobs": logp, "teacher_0_topk_valid": torch.ones_like(ids, dtype=torch.bool),
        "teacher_0_full_logits": teacher_logits, "student_head_inputs": torch.randn(batch, 5, 3, device=device)}
    config = {"projection_matrix_paths": [None], "teacher_weights": [1.], "kl_loss_weight": 1.,
        "variant": {"mode": mode, "score_mode": score_mode, "student_shared_ids": [0, 1, 2, 3, 4, 5],
        "teacher_shared_ids": [0, 1, 2, 3, 4, 5], "sigma2": .1, "gh_threshold": 0.,
        "kl_clamp": 20., "clip_ratio_low": .2, "clip_ratio_high": .28, "clip_ratio_c": 10., "adv_clamp": 10.}}
    return config, logits, data


def cross_fixture_batch(mode, score_mode, batch, device):
    config, _old_logits, data = fixture_batch(mode, score_mode, batch, device)
    config["projection_matrix_paths"] = ["numeric-fixture-only; no projection loaded by variant"]
    logits = torch.randn(batch, 6, 6, device=device, requires_grad=True)
    data["input_ids"] = torch.tensor([[0, 1, 2, 3, 4, 5]] * batch, device=device)
    data["token_mask"] = torch.tensor([[0, 0, 1, 1, 1, 1]] * batch, device=device)
    data["teacher_0_input_ids"] = torch.tensor([[0, 2, 3, 4, 5]] * batch, device=device)
    data["teacher_0_token_mask"] = torch.tensor([[0, 1, 1, 1, 1]] * batch, device=device)
    data["alignment_0_student_chunk_id"] = torch.tensor([[-1, -1, 0, 0, 1, 2]] * batch, device=device)
    data["alignment_0_teacher_chunk_id"] = torch.tensor([[-1, 0, 1, 2, 2]] * batch, device=device)
    data["alignment_0_pair_valid"] = torch.ones(batch, 3, dtype=torch.bool, device=device)
    data["alignment_0_native_group_id"] = torch.zeros(batch, 3, dtype=torch.long, device=device)
    data["student_head_inputs"] = torch.randn(batch, 6, 3, device=device)
    data["behavior_logprobs"] = torch.zeros(batch, 6, device=device)
    data["behavior_logprobs"][:, 1:] = logits.detach()[:, :-1].log_softmax(-1).gather(-1, data["input_ids"][:, 1:, None]).squeeze(-1)
    return config, logits, data


@pytest.mark.parametrize("mode,score", [("dpca", "topk"), ("ctopd_full", "full"), ("ctopd_topk", "topk"),
    ("simct_full", "full"), ("simct_topk", "topk"), ("grass_chunk_fixed_sigma", "topk")])
def test_actual_loss_adapter_backpropagates_and_masks_prompts(mode, score):
    config, logits, data = fixture_batch(mode, score)
    total, metrics = VariantLoss(config)(logits, data, torch.tensor(2), torch.tensor(6))
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(logits.grad).all()
    assert logits.grad[:, 0].abs().max() == 0
    assert metrics["variant_kept_segments"] == 6
    assert metrics["num_valid_samples"] == 2


def test_actual_dpca_adapter_refuses_missing_behavior_or_chosen_teacher_scores():
    config, logits, data = fixture_batch()
    del data["behavior_logprobs"]
    with pytest.raises(ValueError, match="behavior"):
        VariantLoss(config)(logits, data, torch.tensor(2), torch.tensor(6))
    config, logits, data = fixture_batch()
    data["teacher_0_topk_valid"].zero_()
    with pytest.raises(ValueError, match="actual prompt token"):
        VariantLoss(config)(logits, data, torch.tensor(2), torch.tensor(6))


@pytest.mark.parametrize("mode,score", [("dpca", "topk"), ("ctopd_full", "full"), ("ctopd_topk", "topk"),
    ("simct_full", "full"), ("simct_topk", "topk"), ("grass_chunk_fixed_sigma", "topk")])
def test_actual_cross_adapter_handles_many_to_one_one_to_many_and_native_groups(mode, score):
    config, logits, data = cross_fixture_batch(mode, score, 2, "cpu")
    total, metrics = VariantLoss(config)(logits, data, torch.tensor(2), torch.tensor(8))
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(logits.grad).all()
    assert metrics["variant_segments"] == 6
    assert metrics["variant_kept_segments"] == (2 if mode.startswith("ctopd") else 6)
    if mode == "grass_chunk_fixed_sigma":
        assert metrics["variant_alpha_count"] == 2
