"""Real NCCL gradient/update parity on 2/3/4 GPUs with numerical cross-tokenizer fixtures.

This is an adapter/DDP qualification, not a Gym/vLLM or paper-quality run.
"""

import argparse
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_algorithm_variants import cross_fixture_batch
from nemo_rl.algorithms.x_token.variant_loss import VariantLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, local, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world not in (2, 3, 4):
        raise ValueError("Qualify 2, 3 and 4 ranks explicitly")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    device = torch.device("cuda", local)
    torch.manual_seed(23)
    model = torch.nn.Linear(3, 6, bias=False, device=device, dtype=torch.float64)
    wrapped = DistributedDataParallel(model, device_ids=[local])
    report = {"world_size": world, "gpu_name": torch.cuda.get_device_name(local),
        "capability": torch.cuda.get_device_capability(local), "torch": torch.__version__,
        "cuda": torch.version.cuda, "scope": "actual VariantLoss and NCCL DDP; numeric aligned fixtures; no Gym/vLLM", "modes": []}
    initial = deepcopy(model.state_dict())
    try:
        for mode, score in [("dpca", "topk"), ("ctopd_full", "full"), ("ctopd_topk", "topk"),
                ("simct_full", "full"), ("simct_topk", "topk"), ("grass_chunk_fixed_sigma", "topk")]:
            model.load_state_dict(initial)
            model.zero_grad(set_to_none=True)
            config, _, global_data = cross_fixture_batch(mode, score, 12, device)
            global_data["sample_mask"][0] = .25
            global_data["sample_mask"][1] = 0
            reference = deepcopy(model)
            reference_logits = reference(global_data["student_head_inputs"].double())
            global_data["behavior_logprobs"][:, 1:] = reference_logits.detach()[:, :-1].log_softmax(-1).gather(
                -1, global_data["input_ids"][:, 1:, None]).squeeze(-1).float()
            denominator = (global_data["token_mask"][:, 1:] * global_data["sample_mask"][:, None]).sum()
            expected_loss, _ = VariantLoss(config)(reference_logits, global_data, torch.tensor(12, device=device), denominator)
            expected_loss.backward()
            lower, upper = rank * (12 // world), (rank + 1) * (12 // world)
            local_data = {key: value[lower:upper] for key, value in global_data.items()}
            output = wrapped(local_data["student_head_inputs"].double())
            loss, metrics = VariantLoss(config)(output, local_data, torch.tensor(12, device=device), denominator)
            (loss * world).backward()
            maximum = float((model.weight.grad - reference.weight.grad).abs().max())
            torch.testing.assert_close(model.weight.grad, reference.weight.grad, rtol=2e-5, atol=2e-7)
            model.weight.data.add_(model.weight.grad, alpha=-.01)
            reference.weight.data.add_(reference.weight.grad, alpha=-.01)
            torch.testing.assert_close(model.weight, reference.weight, rtol=2e-5, atol=2e-7)
            summed_loss = loss.detach().clone()
            dist.all_reduce(summed_loss)
            torch.testing.assert_close(summed_loss, expected_loss.detach(), rtol=2e-5, atol=2e-7)
            report["modes"].append({"mode": mode, "loss": float(summed_loss), "max_gradient_error": maximum,
                "global_batch_size": 12, "local_batch_size": 12 // world, "gradient_and_update_parity": True})
            dist.barrier()
        report["success"] = True
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("ALGO_DDP=" + json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
