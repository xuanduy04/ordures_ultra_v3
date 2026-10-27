"""Standalone native-venv worker check, invoked remotely without pytest there."""

import json
import multiprocessing
import os
from pathlib import Path

import torch
from torch.distributed.tensor import DTensor
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def export_fixture(connection, teacher_cpu, device):
    from nemo_rl.algorithms.x_token.ipc import ensure_teacher_ipc_buffer

    torch.cuda.set_device(device)
    shape = teacher_cpu.shape
    storage, payload = ensure_teacher_ipc_buffer(None, None, *shape,
                                                 torch.float32, torch.device(f"cuda:{device}"))
    storage.copy_(teacher_cpu)
    torch.cuda.synchronize()
    connection.send(payload)
    assert connection.recv() == "consumed"


def main():
    # Each torchrun rank shares this tiny checkpoint directory on one node.
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    root = Path(os.environ["XTOKEN_FIXTURE_PATH"])
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(31)
        tokenizer = Tokenizer(models.WordLevel({"<pad>": 0, "<unk>": 1,
                                                **{f"w{index}": index for index in range(2, 32)}}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>")
        tokenizer.save_pretrained(root)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=64, intermediate_size=128,
                                             num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                                             max_position_embeddings=64, pad_token_id=0, tie_word_embeddings=False))
        model.save_pretrained(root)
        (root / "READY").touch()
    else:
        import time
        started = time.monotonic()
        while not (root / "READY").is_file():
            if time.monotonic() - started > 30:
                raise TimeoutError("Tiny checkpoint was not created")
            time.sleep(0.1)

    from transformers import AutoTokenizer
    from nemo_rl.algorithms.x_token.loss import CrossTokenizerDistillationLossFn
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2

    config = {
        "model_name": str(root), "tokenizer": {"name": str(root)}, "precision": "bfloat16",
        "generation_batch_size": 1, "train_global_batch_size": 2 * world, "train_micro_batch_size": 1,
        "logprob_batch_size": 1, "offload_optimizer_for_logprob": False, "max_grad_norm": 1.0,
        "max_total_sequence_length": 16, "make_sequence_length_divisible_by": 4,
        "generation": {"backend": "vllm", "temperature": 1., "colocated": {"enabled": True}},
        "dtensor_cfg": {"enabled": True, "_v2": True, "cpu_offload": False,
                        "sequence_parallel": False, "activation_checkpointing": False,
                        "tensor_parallel_size": 1, "context_parallel_size": 1,
                        "expert_parallel_size": 1, "custom_parallel_plan": None},
        "dynamic_batching": {"enabled": False}, "sequence_packing": {"enabled": False},
        "optimizer": {"name": "torch.optim.AdamW", "kwargs": {"lr": 1e-3, "foreach": False, "fused": False}},
        "scheduler": {"name": "torch.optim.lr_scheduler.ConstantLR", "kwargs": {"factor": 1., "total_iters": 10}},
    }
    if "XTOKEN_VARIANT_MODES" in os.environ:
        config["train_global_batch_size"] = 12
        config["dtensor_cfg"]["fsdp_mode"] = "replicated" if world == 3 else "sharded"
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    worker_class = DTensorPolicyWorkerV2.__ray_metadata__.modified_class
    worker = worker_class(config, tokenizer, init_optimizer=True, init_reference_model=False)
    torch.manual_seed(81 + rank)
    batch = BatchedDataDict(input_ids=torch.randint(2, 32, (2, 8)),
                            input_lengths=torch.tensor([8, 6]),
                            sample_mask=torch.ones(2), token_mask=torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1],
                                                                                 [0, 1, 1, 1, 1, 1, 0, 0]]))
    if "XTOKEN_VARIANT_MODES" in os.environ:
        from tools.xtoken_all.tests.variant_native_probe import exercise_native_variants

        try:
            exercise_native_variants(worker, root, rank, world)
        finally:
            torch.distributed.destroy_process_group()
        return
    exported = worker.get_full_logits_ipc(batch, micro_batch_size=1)
    storage = worker._teacher_ipc_storage
    original = storage.clone()
    pointer = storage.data_ptr()
    # Exercise the persistent export path a second time before consumption.
    exported = worker.get_full_logits_ipc(batch, micro_batch_size=1)
    assert worker._teacher_ipc_storage.data_ptr() == pointer
    torch.testing.assert_close(worker._teacher_ipc_storage, original)
    # An explicit synthetic target perturbation makes the optimizer check nontrivial.
    storage[..., 2] += 3
    torch.cuda.synchronize()
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    producer = context.Process(target=export_fixture, args=(child, storage.cpu(), local_rank))
    producer.start()
    assert parent.poll(90), "External fixture producer did not respond"
    payload = parent.recv()
    for entry in exported["per_sample_handles"]:
        entry["payload_ipc"] = payload
    batch["teacher_0_full_logits_ipc"] = [{"teacher_shards": [entry]} for entry in exported["per_sample_handles"]]
    loss_config = {"gold_loss": False, "xtoken_loss": False, "temperature": 1., "vocab_topk": 32,
                   "uncommon_topk": 32, "reverse_kl": False, "exact_token_match_only": True,
                   "kl_loss_weight": 1., "ce_loss_scale": 0., "dynamic_loss_scaling": False,
                   "kd_loss_mode": "sum", "sum_weights_metric": None, "alpha": 1.,
                   "normalize_teacher_by_vocab": False, "student_vocab_size": 32,
                   "teacher_vocab_sizes": [32], "projection_matrix_paths": [None], "teacher_weights": [1.],
                   "teacher_gold_loss": [None], "teacher_xtoken_loss": [None]}
    def parameter_snapshot():
        value = next(worker.model.parameters()).detach()
        print("XTOKEN_PARAMETER_VIEW=" + json.dumps({"rank": rank,
              "type": type(value).__name__, "shape": list(value.shape)}), flush=True)
        if isinstance(value, DTensor):
            value = value.full_tensor()
        return value.float().cpu().clone()

    before = parameter_snapshot()
    try:
        result = worker.train(batch, CrossTokenizerDistillationLossFn(loss_config))
        torch.cuda.synchronize()
        after = parameter_snapshot()
        assert before.shape == after.shape
        assert not torch.equal(before, after), "Optimizer did not change student parameters"
        assert all(torch.isfinite(value).all() for value in worker.model.parameters())
        parent.send("consumed")
        producer.join(30)
        assert producer.exitcode == 0
    finally:
        if producer.is_alive():
            producer.terminate()
            producer.join(10)
    worker.release_ipc_buffer()
    assert worker._teacher_ipc_storage is None
    print("XTOKEN_NATIVE_WORKER=" + json.dumps({"rank": rank, "world": world,
          "export_reuse": True, "parameters_changed": True, "release": True}), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
