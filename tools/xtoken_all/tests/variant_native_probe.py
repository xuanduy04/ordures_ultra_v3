"""Opt-in extension of the pre-existing DTensorPolicyWorkerV2 smoke harness."""

import json
import math
import os

import torch
from torch.distributed.tensor import DTensor

from nemo_rl.algorithms.x_token.variant_loss import VariantLoss
from nemo_rl.algorithms.xtoken_off_policy_distillation import xtoken_non_student_seq_keys
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def snapshot(model):
    result = []
    for value in model.parameters():
        value = value.detach()
        if isinstance(value, DTensor):
            value = value.full_tensor()
        result.append(value.float().cpu().clone())
    return result


def exercise_native_variants(worker, root, rank, world):
    if world not in (2, 3, 4) or 12 % world:
        raise ValueError("Expected a 2/3/4-rank topology and global batch12")
    modes = os.environ["XTOKEN_VARIANT_MODES"].split(",")
    report = {"rank": rank, "world_size": world, "global_batch_size": 12,
        "fsdp_mode": worker.cfg["dtensor_cfg"]["fsdp_mode"],
        "scope": "real DTensorPolicyWorkerV2/FSDP2 optimizer; synthetic uniform behavior and teacher-score fixtures; no Gym/vLLM", "modes": []}
    count = 12 // world
    for mode in modes:
        torch.manual_seed(129 + rank)
        ids = torch.randint(2, 32, (count, 8))
        mask = torch.ones_like(ids)
        mask[:, :2] = 0
        teacher_logits = torch.randn(count, 8, 32)
        teacher_logp = teacher_logits.log_softmax(-1)
        top_ids = teacher_logp.topk(3, dim=-1).indices
        top_ids = torch.cat((top_ids, torch.zeros(count, 8, 1, dtype=torch.long)), -1)
        valid = torch.ones_like(top_ids, dtype=torch.bool)
        for row in range(count):
            for position in range(7):
                chosen = ids[row, position + 1]
                if bool((top_ids[row, position, :3] == chosen).any()):
                    valid[row, position, 3] = False
                top_ids[row, position, 3] = chosen
        top_scores = teacher_logp.gather(-1, top_ids)
        chunks = torch.tensor([[-1, -1, 0, 1, 2, 3, 4, 5]] * count)
        batch = BatchedDataDict(input_ids=ids, input_lengths=torch.full((count,), 8),
            sample_mask=torch.ones(count), token_mask=mask,
            behavior_logprobs=torch.full((count, 8), -math.log(30)),
            teacher_0_input_ids=ids.clone(), teacher_0_token_mask=mask.clone(),
            teacher_0_full_logits=teacher_logits,
            teacher_0_topk_ids=top_ids, teacher_0_topk_logprobs=top_scores, teacher_0_topk_valid=valid,
            alignment_0_student_chunk_id=chunks, alignment_0_teacher_chunk_id=chunks.clone(),
            alignment_0_pair_valid=torch.ones(count, 6, dtype=torch.bool),
            alignment_0_native_group_id=torch.zeros(count, 6, dtype=torch.long))
        config = {"projection_matrix_paths": ["synthetic-cross-fixture"], "teacher_weights": [1.], "kl_loss_weight": 1.,
            "variant": {"mode": mode, "score_mode": "full" if mode.endswith("_full") else "topk",
            "student_shared_ids": list(range(32)), "teacher_shared_ids": list(range(32)),
            "sigma2": .1, "gh_threshold": 0., "kl_clamp": 20., "adv_clamp": 10.,
            "clip_ratio_low": .2, "clip_ratio_high": .28, "clip_ratio_c": 10.}}
        before = snapshot(worker.model)
        loss = VariantLoss(config)
        stats = worker.train(batch, loss, check_dim_skip_keys=xtoken_non_student_seq_keys(loss))
        gradient_norm = float(stats["grad_norm"])
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise AssertionError("Native variant must produce a finite nonzero gradient before optimizer decay")
        after = snapshot(worker.model)
        changed = any(not torch.equal(a, b) for a, b in zip(before, after))
        if not changed or not all(torch.isfinite(value).all() for value in after):
            raise AssertionError("Native variant optimizer did not produce a finite parameter update")
        report["modes"].append({"mode": mode, "parameters_changed": changed, "finite_parameters": True, "grad_norm": gradient_norm})
        print("ALGO_NATIVE_STEP=" + json.dumps({"rank": rank, "world_size": world, "mode": mode, "parameters_changed": changed}), flush=True)
    report["success"] = True
    (root / ("native-variants-rank-" + str(rank) + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    print("ALGO_NATIVE=" + json.dumps(report), flush=True)
