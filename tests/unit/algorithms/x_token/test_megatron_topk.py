import ast
import __future__
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.algorithms.x_token.opd_integration import validate_xtoken_mode, xtoken_non_student_seq_keys
from nemo_rl.algorithms.x_token.topk_tail import ProjectedTopKTailLoss
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
from .test_opd_integration import setup


def load_definitions(path, names, scope):
    tree = ast.parse(path.read_text())
    nodes = [item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=__future__.annotations.compiler_flag), scope)


def mcore_processor(loss_fn, cp_group, tp_group, packed, microbatches):
    scope = {"partial": partial, "torch": torch,
             "get_context_parallel_world_size": lambda: dist.get_world_size(cp_group) if cp_group is not None else 1,
             "get_context_parallel_group": lambda: cp_group,
             "get_tensor_model_parallel_group": lambda: tp_group,
             "get_tensor_model_parallel_rank": lambda: dist.get_rank(tp_group) if tp_group is not None else 0}
    repo = Path(__file__).resolve().parents[4]
    load_definitions(repo / "nemo_rl/models/megatron/train.py", ["LossPostProcessor"], scope)
    return scope["LossPostProcessor"](loss_fn, {"sequence_packing": {"enabled": packed}}, microbatches)


def distributed_fixture(path, reverse):
    torch.save({"indices": torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [-1, -1]]),
                "likelihoods": torch.tensor([[.7, .3], [.6, .4], [.2, .8], [.4, .6], [0., 0.]])}, path)
    cfg = {"temperature": 1., "gold_loss": False, "xtoken_loss": False, "kd_loss_mode": "sum",
           "dynamic_loss_scaling": False, "ce_loss_scale": 0., "sum_weights_metric": None,
           "normalize_teacher_by_vocab": False, "teacher_gold_loss": [None], "teacher_xtoken_loss": [None],
           "projection_matrix_paths": [str(path)], "student_vocab_size": 5, "teacher_vocab_sizes": [4],
           "teacher_weights": [1.], "kl_loss_weight": .7, "tail_epsilon": 1e-8,
           "reverse_kl": reverse, "exact_token_match_only": True}
    torch.manual_seed(193)
    teacher = torch.randn(2, 11, 4).softmax(-1)
    choices = torch.tensor([[0, 2], [1, 3]]).repeat(6, 1)[:11].unsqueeze(0).expand(2, -1, -1)
    ids = torch.cat([choices, torch.full((2, 11, 1), -1)], -1)
    lp = torch.cat([teacher.gather(-1, choices).log(), torch.full((2, 11, 1), -float("inf"))], -1)
    tm = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
    data = BatchedDataDict(input_ids=torch.zeros(2, 8, dtype=torch.long), input_lengths=torch.tensor([7, 6]),
        token_mask=torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 0, 0]]),
        sample_mask=torch.tensor([1., .5]), teacher_0_topk_ids=ids, teacher_0_topk_logprobs=lp,
        teacher_0_topk_valid=ids >= 0, teacher_0_token_mask=tm,
        alignment_0_student_chunk_id=torch.tensor([[-1, -1, 0, 0, 0, 1, 1, -1], [-1, -1, 0, 0, 1, 1, -1, -1]]),
        alignment_0_teacher_chunk_id=torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1],
                                                  [-1, -1, 0, 0, 1, 1, 1, -1, -1, -1, -1]]),
        alignment_0_pair_valid=torch.tensor([[True, True], [True, False]]),
        alignment_0_pair_is_correct=torch.ones(2, 2, dtype=torch.bool))
    logits = torch.randn(2, 8, 8)
    logits[..., 5:] = 100.
    return cfg, data, logits


def shard_sequence(value, data, cp_rank, cp_size, packed):
    if not packed:
        return _get_tokens_on_this_cp_rank(value, cp_rank, cp_size, seq_dim=1).contiguous()
    # Exercise individual padding plus PP padding on the final packed sample.
    rows = []
    for b, padded in enumerate([8, 16]):
        length = int(data["input_lengths"][b])
        row = torch.nn.functional.pad(value[b, :length], (0, 0, 0, padded - length))
        rows.append(_get_tokens_on_this_cp_rank(row, cp_rank, cp_size, seq_dim=0))
    return torch.cat(rows).unsqueeze(0).contiguous()


def parallel_worker(rank, tp, cp, init_path, cfg, data, full, packed):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_path, rank=rank, world_size=tp * cp)
    try:
        tp_group = cp_group = None
        for c in range(cp):
            group = dist.new_group([c * tp + t for t in range(tp)])
            if rank // tp == c:
                tp_group = group
        for t in range(tp):
            group = dist.new_group([c * tp + t for c in range(cp)])
            if rank % tp == t:
                cp_group = group
        denominator = (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).sum()
        reference_logits = full.clone().requires_grad_(True)
        reference, _ = ProjectedTopKTailLoss(cfg)(reference_logits, data, torch.tensor(1.5), denominator)
        reference_gradient = torch.autograd.grad(reference, reference_logits)[0]
        start = (rank % tp) * (full.shape[-1] // tp)
        end = start + full.shape[-1] // tp
        local = shard_sequence(full, data, rank // tp, cp, packed)[..., start:end].clone().requires_grad_(True)
        params = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 8, 24]), cu_seqlens_q_padded=torch.tensor([0, 8, 24])) if packed else None
        loss_fn = ProjectedTopKTailLoss(cfg)
        # Execute the actual Megatron closure, then its schedule's CP/MB factor.
        closure = mcore_processor(loss_fn, cp_group, tp_group, packed, 3)(data, params, torch.tensor(1.5), denominator)
        wrapped, metrics = closure(local)
        actual = wrapped * cp / 3
        torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-6)
        actual.backward()
        expected = shard_sequence(reference_gradient, data, rank // tp, cp, packed)[..., start:end]
        torch.testing.assert_close(local.grad, expected, rtol=5e-5, atol=5e-6)
        assert torch.isfinite(local.grad).all()
        assert metrics["num_valid_samples"] == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tp,cp", [(2, 1), (1, 2), (2, 2), (4, 1)])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_real_distributed_megatron_layout_matches_dense_loss_and_gradients(tmp_path, tp, cp, packed, reverse):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("CPU Gloo runner required")
    cfg, data, full = distributed_fixture(tmp_path / "projection.pt", reverse)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        mp.start_processes(parallel_worker, args=(tp, cp, str(tmp_path / "rendezvous"), cfg, data, full, packed),
                           nprocs=tp * cp, join=True, start_method="spawn")
    finally:
        torch.set_num_threads(original_threads)


def test_megatron_backend_uses_native_policy_parallel_and_batching_config(setup):
    cfg, *_ = setup
    policy = cfg["policy"]
    policy["dtensor_cfg"]["enabled"] = False
    policy["megatron_cfg"] = {"enabled": True, "tensor_model_parallel_size": 4,
        "context_parallel_size": 2, "pipeline_model_parallel_size": 2, "sequence_parallel": True}
    for packed in (False, True):
        policy["sequence_packing"]["enabled"] = packed
        policy["dynamic_batching"]["enabled"] = not packed
        validate_xtoken_mode(cfg)
    cfg["grpo"]["async_grpo"].update(max_trajectory_age_steps=2, in_flight_weight_updates=True)
    policy["generation"].update(temperature=.7, top_p=.9)
    policy["train_global_batch_size"] = 1
    cfg["grpo"]["reward_shaping"]["enabled"] = True
    validate_xtoken_mode(cfg)


def test_dynamic_microbatch_preserves_teacher_and_pair_axes():
    data = BatchedDataDict(input_ids=torch.zeros(2, 8), token_mask=torch.ones(2, 8),
                          teacher_0_input_ids=torch.ones(2, 13), alignment_0_pair_valid=torch.ones(2, 2))
    data.micro_batch_indices = [[(0, 1), (1, 2)]]
    data.micro_batch_lengths = [[4, 8]]
    batches = list(data.make_microbatch_iterator_with_dynamic_shapes(skip_keys=xtoken_non_student_seq_keys()))
    assert batches[0]["input_ids"].shape == (1, 4)
    assert batches[0]["teacher_0_input_ids"].shape == (1, 13)
    assert batches[0]["alignment_0_pair_valid"].shape == (1, 2)


def test_megatron_sequence_validation_and_microbatch_skip_keys():
    scope = {"torch": torch}
    repo = Path(__file__).resolve().parents[4]
    load_definitions(repo / "nemo_rl/models/megatron/data.py", ["get_and_validate_seqlen"], scope)
    data = BatchedDataDict(input_ids=torch.zeros(2, 8), token_mask=torch.ones(2, 8),
                          teacher_0_input_ids=torch.ones(2, 13), alignment_0_pair_valid=torch.ones(2, 2))
    with pytest.raises(AssertionError):
        scope["get_and_validate_seqlen"](data)
    assert scope["get_and_validate_seqlen"](data, skip_keys=xtoken_non_student_seq_keys()) == (1, 8)
    data["alignment_0_student_chunk_id"] = torch.zeros(2, 7)
    with pytest.raises(AssertionError):
        scope["get_and_validate_seqlen"](data, skip_keys=xtoken_non_student_seq_keys())


def test_actual_megatron_dynamic_iterator_preserves_independent_axes():
    scope = {"torch": torch, "make_processed_microbatch_iterator": lambda raw_iterator, **kwargs: raw_iterator}
    repo = Path(__file__).resolve().parents[4]
    load_definitions(repo / "nemo_rl/models/megatron/data.py",
                     ["get_and_validate_seqlen", "get_microbatch_iterator"], scope)
    data = BatchedDataDict(input_ids=torch.zeros(2, 8), token_mask=torch.ones(2, 8),
                          teacher_0_input_ids=torch.ones(2, 13), alignment_0_pair_valid=torch.ones(2, 2))
    data.micro_batch_indices = [[(0, 1), (1, 2)]]
    data.micro_batch_lengths = [[4, 8]]
    iterator, count, mbs, length, padded = scope["get_microbatch_iterator"](
        data, {"sequence_packing": {"enabled": False}, "dynamic_batching": {"enabled": True}},
        1, None, check_dim_skip_keys=xtoken_non_student_seq_keys())
    assert (count, mbs, length, padded) == (2, 1, 8, 8)
    first = next(iterator)
    assert first["input_ids"].shape == (1, 4)
    assert first["teacher_0_input_ids"].shape == (1, 13)
    assert first["alignment_0_pair_valid"].shape == (1, 2)


@pytest.mark.parametrize("raw", [True, False])
def test_actual_megatron_forward_keeps_raw_kd_logits_and_legacy_sampling_temperature(raw):
    class Capture:
        requires_raw_logits = raw
        def __call__(self, logits, **kwargs):
            return logits.clone(), {}
    scope = {"torch": torch, "partial": partial,
             "model_forward": lambda **kwargs: torch.tensor([[[2., 4.]]]),
             "get_tensor_model_parallel_group": lambda: None, "get_tensor_model_parallel_rank": lambda: 0,
             "get_context_parallel_group": lambda: None, "get_context_parallel_world_size": lambda: 1,
             "LogprobsPostProcessor": type("LP", (), {}), "TopkLogitsPostProcessor": type("TopK", (), {})}
    repo = Path(__file__).resolve().parents[4]
    load_definitions(repo / "nemo_rl/models/megatron/train.py",
                     ["LossPostProcessor", "forward_with_post_processing_fn", "apply_temperature_scaling"], scope)
    cfg = {"sequence_packing": {"enabled": False}, "generation": {"temperature": .5}}
    processor = scope["LossPostProcessor"](Capture(), cfg)
    mb = SimpleNamespace(data_dict={}, input_ids=None, input_ids_cp_sharded=None, attention_mask=None,
                         position_ids=None, packed_seq_params=None, cu_seqlens_padded=None, mtp_loss_mask=None)
    output, closure = scope["forward_with_post_processing_fn"](iter([mb]), None, cfg, processor)
    result, _ = closure(output)
    torch.testing.assert_close(result, torch.tensor([[[2., 4.]]]) if raw else torch.tensor([[[4., 8.]]]))


@pytest.mark.parametrize("age", [0, 1, 2])
def test_native_async_collector_never_scores_student_ids_with_cross_tokenizer_teacher(setup, age):
    from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector
    cfg, *_ = setup
    cfg["grpo"]["async_grpo"]["max_trajectory_age_steps"] = age
    cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
    collector = cls(None, None, {}, cfg, None)
    assert not collector._has_opd_teachers
    assert collector._teacher_logprob_client is None
    assert collector._fresh_xtoken == (age == 0)


def test_legacy_loss_requires_explicit_opt_in_for_packing_and_cp_scaling():
    from unittest.mock import MagicMock

    original = MagicMock(return_value=(torch.tensor(1.), {}))
    wrapper = MagicMock(return_value=original)
    scope = {"partial": partial, "torch": torch,
             "get_context_parallel_world_size": lambda: 2,
             "get_context_parallel_group": lambda: None,
             "get_tensor_model_parallel_group": lambda: None,
             "get_tensor_model_parallel_rank": lambda: 0,
             "SequencePackingLossWrapper": wrapper}
    repo = Path(__file__).resolve().parents[4]
    load_definitions(repo / "nemo_rl/models/megatron/train.py", ["LossPostProcessor"], scope)
    cfg = {"sequence_packing": {"enabled": True, "fuse_loss": False}}
    processor = scope["LossPostProcessor"](original, cfg, num_microbatches=4)
    params = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 8]),
                             cu_seqlens_q_padded=torch.tensor([0, 8]))
    result, _ = processor({}, params)(torch.zeros(1))
    wrapper.assert_called_once()
    assert "packed_seq_params" not in original.call_args.kwargs
    torch.testing.assert_close(result, torch.tensor(1.))


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_actual_nonpacked_megatron_inputs_follow_cp_ownership(cp_rank):
    from contextlib import nullcontext
    input_ids = torch.arange(16).reshape(2, 8)
    positions = torch.arange(8).expand(2, -1).contiguous()
    attention_mask = torch.ones(1, 1, 8, 8, dtype=torch.bool).triu(1)
    mtp = torch.arange(16).reshape(2, 8).float()
    data = BatchedDataDict(input_ids=input_ids, mtp_loss_mask=mtp)
    scope = {"torch": torch, "nullcontext": nullcontext, "ProcessedInputs": SimpleNamespace,
             "get_ltor_masks_and_position_ids": lambda **kwargs: (attention_mask, None, positions),
             "get_context_parallel_world_size": lambda: 2, "get_context_parallel_rank": lambda: cp_rank,
             "_get_tokens_on_this_cp_rank": _get_tokens_on_this_cp_rank}
    repo = Path(__file__).resolve().parents[4]
    load_definitions(repo / "nemo_rl/models/megatron/data.py", ["process_microbatch"], scope)
    result = scope["process_microbatch"](data)
    expected_indices = torch.tensor([0, 1, 6, 7]) if cp_rank == 0 else torch.tensor([2, 3, 4, 5])
    torch.testing.assert_close(result.input_ids, input_ids)
    torch.testing.assert_close(result.input_ids_cp_sharded, input_ids[:, expected_indices])
    torch.testing.assert_close(result.position_ids, positions[:, expected_indices])
    torch.testing.assert_close(result.attention_mask, attention_mask[:, :, expected_indices])
    torch.testing.assert_close(result.mtp_loss_mask, mtp[:, expected_indices])


@pytest.mark.parametrize("with_grad", [True, False])
def test_projection_collective_materializes_contiguous_storage(monkeypatch, with_grad):
    from nemo_rl.algorithms.x_token.parallel import group_all_reduce_sum, group_all_reduce_sum_with_grad
    seen = []
    def reduce(tensor, **kwargs):
        assert tensor.is_contiguous()
        seen.append(tensor.clone())
        tensor.mul_(2)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    base = torch.arange(6.).reshape(2, 3).requires_grad_()
    transposed = base.t()
    assert not transposed.is_contiguous()
    reduction = group_all_reduce_sum_with_grad if with_grad else group_all_reduce_sum
    result = reduction(transposed, object())
    torch.testing.assert_close(result, transposed * 2)
    torch.testing.assert_close(base, torch.arange(6.).reshape(2, 3))
    assert len(seen) == 1 and result.is_contiguous()
    if with_grad:
        result.sum().backward()
        torch.testing.assert_close(base.grad, torch.ones_like(base))
    else:
        assert not result.requires_grad
