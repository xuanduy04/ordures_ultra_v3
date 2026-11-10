import argparse
import gc
import importlib.metadata as md
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def fixture(path):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    generator = torch.Generator().manual_seed(71)
    student_vocab, teacher_vocab, length, teacher_length = 37, 29, 32, 39
    projection_ids = torch.stack([torch.arange(student_vocab) % teacher_vocab,
                                 (torch.arange(student_vocab) + 3) % teacher_vocab], -1)
    projection_ids[-1] = -1
    projection_weights = torch.tensor([.7, .3]).expand(student_vocab, -1).clone()
    projection_weights[-1] = 0
    torch.save({"indices": projection_ids, "likelihoods": projection_weights}, path)
    matrix = torch.zeros(student_vocab, teacher_vocab)
    for s in range(student_vocab - 1):
        for k in range(2):
            matrix[s, projection_ids[s, k]] += projection_weights[s, k]
    lengths = torch.tensor([23, 29])
    inputs = torch.randint(1, student_vocab, (2, length), generator=generator)
    mask = torch.zeros(2, length)
    student_chunks = torch.full((2, length), -1, dtype=torch.long)
    for b, size in enumerate(lengths):
        inputs[b, size:] = 0
        mask[b, 3:size - 1] = 1
        student_chunks[b, 3:10] = 0
        student_chunks[b, 10:18] = 1
        student_chunks[b, 18:size - 1] = 2
    teacher_chunks = torch.full((2, teacher_length), -1, dtype=torch.long)
    teacher_mask = torch.zeros(2, teacher_length)
    for b, size in enumerate([36, 32]):
        teacher_mask[b, 4:size - 1] = 1
        teacher_chunks[b, 4:14] = 0
        teacher_chunks[b, 14:24] = 1
        teacher_chunks[b, 24:size - 1] = 2
    scores = torch.randn(2, teacher_length, teacher_vocab, generator=generator).log_softmax(-1)
    values, ids = scores.topk(4, -1)
    ids = torch.cat([ids, torch.full((2, teacher_length, 1), -1)], -1)
    values = torch.cat([values, torch.full((2, teacher_length, 1), -float("inf"))], -1)
    data = BatchedDataDict(input_ids=inputs, input_lengths=lengths, token_mask=mask,
        sample_mask=torch.tensor([1., .5]), advantages=torch.zeros_like(mask),
        teacher_0_input_ids=torch.zeros(2, teacher_length, dtype=torch.long),
        teacher_0_topk_ids=ids, teacher_0_topk_logprobs=values,
        teacher_0_topk_valid=ids >= 0, teacher_0_token_mask=teacher_mask,
        alignment_0_student_chunk_id=student_chunks,
        alignment_0_teacher_chunk_id=teacher_chunks,
        alignment_0_pair_valid=torch.tensor([[True, True, True], [True, True, False]]),
        alignment_0_pair_is_correct=torch.ones(2, 3, dtype=torch.bool))
    config = {"temperature": 1., "gold_loss": False, "xtoken_loss": False,
        "kd_loss_mode": "sum", "dynamic_loss_scaling": False, "ce_loss_scale": 0.,
        "sum_weights_metric": None, "normalize_teacher_by_vocab": False,
        "teacher_gold_loss": [None], "teacher_xtoken_loss": [None],
        "projection_matrix_paths": [str(path)], "student_vocab_size": student_vocab,
        "teacher_vocab_sizes": [teacher_vocab], "teacher_weights": [1.],
        "kl_loss_weight": .7, "tail_epsilon": 1e-8, "reverse_kl": False,
        "exact_token_match_only": True}
    return data, config, matrix


def dense_reference(logits, data, matrix, denominator, reverse):
    probabilities = logits[..., :37].softmax(-1)
    total = probabilities.sum() * 0
    for b in range(len(data["input_lengths"])):
        for c in range(3):
            if not data["alignment_0_pair_valid"][b, c] or not data["alignment_0_pair_is_correct"][b, c]:
                continue
            selected = (data["alignment_0_student_chunk_id"][b, 1:] == c) & (data["token_mask"][b, 1:] > 0)
            count = selected.sum()
            if count == 0:
                continue
            projected = probabilities[b, :-1][selected].mean(0) @ matrix
            teacher_rows = (data["alignment_0_teacher_chunk_id"][b, 1:] == c) & (data["teacher_0_token_mask"][b, 1:] > 0)
            losses = []
            for row in teacher_rows.nonzero().flatten():
                ids = data["teacher_0_topk_ids"][b, row]
                valid = data["teacher_0_topk_valid"][b, row] & (ids >= 0)
                ids = ids[valid]
                p = data["teacher_0_topk_logprobs"][b, row][valid].exp()
                q = projected[ids]
                p = torch.cat([p, (1 - p.sum()).clamp_min(0).reshape(1)])
                q = torch.cat([q, (1 - q.sum()).clamp_min(0).reshape(1)])
                logp, logq = p.clamp_min(1e-8).log(), q.clamp_min(1e-8).log()
                losses.append((q * (logq - logp)).sum() if reverse else (p * (logp - logq)).sum())
            if losses:
                total = total + torch.stack(losses).mean() * count * data["sample_mask"][b]
    return total * .7 / denominator


def local_positions(data, params, cp_rank, cp_size):
    from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
    bsz, length = data["input_ids"].shape
    ids = torch.arange(bsz * length, device="cuda").reshape(bsz, length)
    if params is None:
        return _get_tokens_on_this_cp_rank(ids, cp_rank, cp_size, seq_dim=1).contiguous()
    cu = params.cu_seqlens_q_padded
    if cu is None:
        cu = params.cu_seqlens_q
    parts = []
    for b, size in enumerate(data["input_lengths"]):
        real = int(size)
        padded = int(cu[b + 1] - cu[b])
        sample = torch.nn.functional.pad(ids[b, :real], (0, padded - real), value=-1)
        parts.append(_get_tokens_on_this_cp_rank(sample, cp_rank, cp_size, seq_dim=0))
    return torch.cat(parts).unsqueeze(0).contiguous()


def gather_dense(logits, positions, data, tp_group, cp_group):
    vocab_parts = [torch.empty_like(logits) for _ in range(dist.get_world_size(tp_group))]
    dist.all_gather(vocab_parts, logits.detach().contiguous(), group=tp_group)
    full_vocab = torch.cat(vocab_parts, -1)
    sequence_parts = [torch.empty_like(full_vocab) for _ in range(dist.get_world_size(cp_group))]
    position_parts = [torch.empty_like(positions) for _ in range(dist.get_world_size(cp_group))]
    dist.all_gather(sequence_parts, full_vocab.contiguous(), group=cp_group)
    dist.all_gather(position_parts, positions, group=cp_group)
    dense = torch.zeros(*data["input_ids"].shape, full_vocab.shape[-1], device="cuda")
    dense_flat = dense.view(-1, dense.shape[-1])
    for part, where in zip(sequence_parts, position_parts):
        valid = where.flatten() >= 0
        dense_flat.index_copy_(0, where.flatten()[valid], part.float().reshape(-1, part.shape[-1])[valid])
    return dense.requires_grad_(True)


def run_case(args):
    from megatron.core import parallel_state as ps
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from nemo_rl.algorithms.x_token.topk_tail import ProjectedTopKTailLoss
    from nemo_rl.models.megatron.data import get_microbatch_iterator, process_global_batch
    from nemo_rl.models.megatron.train import LossPostProcessor, megatron_forward_backward

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    ps.initialize_model_parallel(tensor_model_parallel_size=args.tp, context_parallel_size=args.cp,
                                 pipeline_model_parallel_size=1)
    if rank == 0:
        print("GPU_ENV=" + json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
            "te": md.version("transformer-engine"), "nccl": torch.cuda.nccl.version(), "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(), "tp": args.tp, "cp": args.cp}), flush=True)
    tp_group, cp_group = ps.get_tensor_model_parallel_group(), ps.get_context_parallel_group()
    tp_rank, cp_rank = ps.get_tensor_model_parallel_rank(), ps.get_context_parallel_rank()
    data, kd_config, projection = fixture(Path("/tmp") / f"projection-{rank}.pt")
    results = []
    for packed in ((False, True) if args.packed == -1 else (bool(args.packed),)):
        torch.manual_seed(928)
        torch.cuda.manual_seed_all(928)
        model_parallel_cuda_manual_seed(928)
        cfg = TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4,
            ffn_hidden_size=256, tensor_model_parallel_size=args.tp,
            context_parallel_size=args.cp, pipeline_model_parallel_size=1,
            sequence_parallel=args.tp > 1, bf16=True, params_dtype=torch.bfloat16,
            hidden_dropout=0., attention_dropout=0., normalization="RMSNorm",
            add_bias_linear=False, gated_linear_unit=True, activation_func=torch.nn.functional.silu,
            attention_backend=AttnBackend.fused, cp_comm_type="p2p",
            gradient_accumulation_fusion=False, calculate_per_token_loss=True,
            bias_activation_fusion=False, bias_dropout_fusion=False,
            masked_softmax_fusion=False, persist_layer_norm=False)
        raw = GPTModel(config=cfg, transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=64, max_sequence_length=128, parallel_output=True,
            share_embeddings_and_output_weights=False, position_embedding_type="rope")
        raw = Float16Module(cfg, raw).cuda()
        model = DistributedDataParallel(config=cfg,
            ddp_config=DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False,
                average_in_collective=False, check_for_nan_in_grad=True), module=raw)
        cfg.finalize_model_grads_func = finalize_model_grads
        cfg.no_sync_func = model.no_sync
        optimizer = get_megatron_optimizer(OptimizerConfig(optimizer="sgd", lr=.1,
            min_lr=.1, bf16=True, params_dtype=torch.bfloat16, use_distributed_optimizer=False,
            clip_grad=0., weight_decay=0.), [model])
        policy = {"sequence_packing": {"enabled": packed}, "dynamic_batching": {"enabled": False},
            "megatron_cfg": {"tensor_model_parallel_size": args.tp, "context_parallel_size": args.cp,
                "pipeline_model_parallel_size": 1, "sequence_parallel": args.tp > 1},
            "generation": {"temperature": .7}}
        model.train()
        for step, reverse in enumerate((False, True)):
            model.zero_grad_buffer()
            optimizer.zero_grad()
            kd_config["reverse_kl"] = reverse
            loss_fn = ProjectedTopKTailLoss(kd_config)
            gb = process_global_batch(data, loss_fn, ps.get_data_parallel_group(), batch_idx=0, batch_size=2)
            batch = gb["batch"]
            if packed:
                batch.micro_batch_indices = [[(0, 2)]]
                batch.micro_batch_lengths = [[52]]
            skip_keys = [k for k in batch if k.startswith("teacher_") or k.endswith("pair_valid") or k.endswith("pair_is_correct") or k.endswith("teacher_chunk_id")]
            iterator, nmb, mbs, seq_len, padded_len = get_microbatch_iterator(batch, policy, 1, None, check_dim_skip_keys=skip_keys)
            original_iterator = iterator
            def inspect_inputs():
                for mb in original_iterator:
                    where = local_positions(mb.data_dict, mb.packed_seq_params, cp_rank, args.cp)
                    source = mb.data_dict["input_ids"].flatten()
                    expected_input = source[where.clamp_min(0)] * (where >= 0)
                    torch.testing.assert_close(mb.input_ids_cp_sharded, expected_input,
                        msg=f"Native CP/packed model inputs do not match their declared layout: TP={args.tp}, CP={args.cp}, packing={packed}")
                    yield mb
            iterator = inspect_inputs()
            audits = []
            class AuditProcessor(LossPostProcessor):
                def __call__(self, data_dict, packed_seq_params=None, **kwargs):
                    native = super().__call__(data_dict, packed_seq_params, **kwargs)
                    def checked(logits):
                        positions = local_positions(data_dict, packed_seq_params, cp_rank, args.cp)
                        reference_logits = gather_dense(logits, positions, data_dict, tp_group, cp_group)
                        expected = dense_reference(reference_logits, data_dict, projection.cuda(), gb["global_valid_toks"], reverse)
                        gradient = torch.autograd.grad(expected, reference_logits)[0].flatten(0, 1)
                        local_expected = gradient[positions.clamp_min(0)] * (positions >= 0)[..., None]
                        width = logits.shape[-1]
                        local_expected = local_expected[..., tp_rank * width:(tp_rank + 1) * width]
                        audit = {"loss": expected.item(), "gradient_checked": False,
                            "logits_shape": list(logits.shape), "packed": packed,
                            "cu_seqlens": None if packed_seq_params is None else packed_seq_params.cu_seqlens_q.tolist()}
                        def check_gradient(actual):
                            torch.testing.assert_close(actual.float(), local_expected.float(), rtol=2e-3, atol=2e-6)
                            audit["gradient_checked"] = True
                            audit["max_logit_gradient_error"] = (actual.float() - local_expected).abs().max().item()
                            return actual
                        logits.register_hook(check_gradient)
                        loss, metrics = native(logits)
                        torch.testing.assert_close(loss * args.cp / nmb, expected, rtol=2e-5, atol=2e-6)
                        audits.append(audit)
                        return loss, metrics
                    return checked
            processor = AuditProcessor(loss_fn, policy, num_microbatches=nmb)
            before = [p.detach().clone() for p in model.parameters()]
            started = time.monotonic()
            metrics = megatron_forward_backward(model, policy, iterator, nmb, padded_len, mbs, processor,
                global_valid_seqs=gb["global_valid_seqs"], global_valid_toks=gb["global_valid_toks"])
            assert len(audits) == nmb and all(row["gradient_checked"] for row in audits)
            gradients = [p.main_grad for p in model.parameters() if p.requires_grad]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            norm = sum(g.float().square().sum() for g in gradients).sqrt().item()
            assert norm > 0
            success, _, _ = optimizer.step()
            assert success
            delta = sum((p.detach().float() - old.float()).square().sum() for p, old in zip(model.parameters(), before)).sqrt().item()
            assert delta > 0 and all(torch.isfinite(p).all() for p in model.parameters())
            torch.cuda.synchronize()
            item = {"tp": args.tp, "cp": args.cp, "packed": packed, "reverse_kl": reverse,
                "loss": sum(row["loss"] for row in audits), "microbatches": nmb, "grad_norm_local": norm,
                "parameter_delta_local": delta, "seconds": time.monotonic() - started,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(), "audits": audits}
            results.append(item)
            if rank == 0:
                print("GPU_CASE=" + json.dumps(item), flush=True)
            dist.barrier()
        del optimizer, model, raw
        gc.collect()
        torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        print("GPU_GROUP_PASS=" + json.dumps({"tp": args.tp, "cp": args.cp, "cases": len(results)}), flush=True)
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--cp", type=int, required=True)
    parser.add_argument("--packed", type=int, choices=[-1, 0, 1], required=True)
    run_case(parser.parse_args())
