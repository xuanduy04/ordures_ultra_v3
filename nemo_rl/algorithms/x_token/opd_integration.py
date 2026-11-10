import asyncio
import math

import torch

from nemo_rl.algorithms.opd_outsource import normalize_teacher_serve, resolve_teacher_specs
from nemo_rl.algorithms.x_token.remote_teacher import XTokenTopKLogprobClient
from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.algorithms.x_token.topk_tail import ProjectedTopKTailLoss
from nemo_rl.data.xtoken_rollout_collate import RolloutCrossTokenizerCollator
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def validate_xtoken_mode(master_config):
    """Reject unsupported execution contracts before allocating workers."""
    policy = master_config["policy"]
    grpo = master_config["grpo"]
    cfg = master_config["on_policy_distillation"]["xtoken"]
    megatron = "megatron_cfg" in policy and policy["megatron_cfg"]["enabled"]
    dtensor = "dtensor_cfg" in policy and policy["dtensor_cfg"]["enabled"]
    if megatron == dtensor:
        raise ValueError("xtoken_topk requires one enabled policy backend")
    if dtensor:
        backend = policy["dtensor_cfg"]
        if "_v2" not in backend or not backend["_v2"]:
            raise ValueError("xtoken_topk requires DTensor V2 for the DTensor backend")
        if backend["tensor_parallel_size"] != 1 or backend["context_parallel_size"] != 1:
            raise ValueError("The DTensor X-Token path requires TP=CP=1; Megatron handles sharded logits")
        if policy["dynamic_batching"]["enabled"] or policy["sequence_packing"]["enabled"]:
            raise ValueError("The DTensor X-Token path requires fixed microbatches without packing")
    async_cfg = grpo["async_grpo"]
    age = async_cfg["max_trajectory_age_steps"]
    if not async_cfg["enabled"] or age < 0:
        raise ValueError("xtoken_topk requires native async collection with a nonnegative trajectory age")
    if age == 0 and "in_flight_weight_updates" in async_cfg and async_cfg["in_flight_weight_updates"]:
        raise ValueError("xtoken_topk requires refit after all rollouts have completed")
    if policy["generation"]["colocated"]["enabled"]:
        raise ValueError("Native OPD xtoken_topk requires non-colocated generation")
    generation = policy["generation"]
    if age == 0 and (generation["temperature"] != 1 or generation["top_p"] != 1 or generation["top_k"]):
        raise ValueError("U7 fresh rollouts require temperature=1, top_p=1 and no generation top-k")
    expected = grpo["num_prompts_per_step"] * grpo["num_generations_per_prompt"]
    if age == 0 and policy["train_global_batch_size"] != expected:
        raise ValueError("xtoken_topk requires one optimizer step per fresh rollout batch")
    if master_config["loss_fn"]["reference_policy_kl_penalty"] != 0:
        raise ValueError("xtoken_topk is KD-only; reference policy KL must be zero")
    for key in ("kl_loss_weight", "tail_epsilon"):
        if not math.isfinite(cfg[key]):
            raise ValueError(key + " must be finite")
    if cfg["kl_loss_weight"] <= 0 or not 0 < cfg["tail_epsilon"] < 1:
        raise ValueError("Require positive KL weight and tail_epsilon in (0, 1)")
    if type(cfg["reverse_kl"]) is not bool:
        raise ValueError("reverse_kl must be a bool")
    for key in ("alignment_max_combination_length", "teacher_max_sequence_length", "teacher_micro_batch_size"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    path = cfg["projection_matrix_path"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError("projection_matrix_path is required")
    specs = [
        normalize_teacher_serve(raw)
        for raw in master_config["on_policy_distillation"]["teacher_model_by_agent_name"].values()
    ]
    if len({(spec["url"], spec["model"]) for spec in specs}) != 1:
        raise ValueError("xtoken_topk currently supports exactly one teacher URL/model pair")


def build_xtoken_loss(cfg, student_vocab_size):
    """Build the U7 projected top-k plus OTHER objective, without PG terms."""
    return ProjectedTopKTailLoss(
        {
            "temperature": 1.0,
            "gold_loss": False,
            "xtoken_loss": False,
            "kd_loss_mode": "sum",
            "dynamic_loss_scaling": False,
            "ce_loss_scale": 0.0,
            "sum_weights_metric": None,
            "normalize_teacher_by_vocab": False,
            "teacher_gold_loss": [None],
            "teacher_xtoken_loss": [None],
            "projection_matrix_paths": [cfg["projection_matrix_path"]],
            "student_vocab_size": student_vocab_size,
            "teacher_vocab_sizes": [cfg["remote"]["model_vocab_size"]],
            "teacher_weights": [1.0],
            "kl_loss_weight": cfg["kl_loss_weight"],
            "reverse_kl": cfg["reverse_kl"],
            "tail_epsilon": cfg["tail_epsilon"],
            "exact_token_match_only": True,
        }
    )


def xtoken_non_student_seq_keys():
    """Name tensors whose second axis is teacher tokens or alignment pairs."""
    return {
        "teacher_0_input_ids",
        "teacher_0_token_mask",
        "teacher_0_topk_ids",
        "teacher_0_topk_logprobs",
        "teacher_0_topk_valid",
        "alignment_0_pair_valid",
        "alignment_0_pair_is_correct",
        "alignment_0_teacher_chunk_id",
        "alignment_0_teacher_exact_partition_mask",
    }


def validate_xtoken_replay_state(state, step):
    """Require resumable batches to belong to the exact restored policy."""
    count = len(state["trajectories"])
    if len(state["trajectory_versions"]) != count or len(state["target_weight_versions"]) != count:
        raise ValueError("Inconsistent replay checkpoint lengths")
    if any(version != step for version in state["trajectory_versions"] + state["target_weight_versions"]):
        raise ValueError("xtoken_topk cannot resume stale/future replay groups from another OPD contract")


class XTokenOPDBatchBuilder:
    """Adapt one native Gym batch without retokenizing sampled student IDs."""

    def __init__(self, master_config, student_tokenizer, teacher_tokenizer):
        self.opd = master_config["on_policy_distillation"]
        self.cfg = self.opd["xtoken"]
        if len(teacher_tokenizer) > self.cfg["remote"]["model_vocab_size"]:
            raise ValueError("Teacher tokenizer IDs exceed remote.model_vocab_size")
        policy = master_config["policy"]
        self.fresh_rollouts = master_config["grpo"]["async_grpo"]["max_trajectory_age_steps"] == 0
        divisor = policy["make_sequence_length_divisible_by"]
        if "megatron_cfg" in policy and policy["megatron_cfg"]["enabled"]:
            backend = policy["megatron_cfg"]
            cp = backend["context_parallel_size"]
            sp = backend["tensor_model_parallel_size"] if backend["sequence_parallel"] else 1
            divisor = math.lcm(divisor, (2 * cp if cp > 1 else 1) * sp)
        stop_ids = student_tokenizer.eos_token_id
        stop_ids = stop_ids if isinstance(stop_ids, list) else [stop_ids]
        if "stop_token_ids" in policy["generation"] and policy["generation"]["stop_token_ids"] is not None:
            stop_ids = stop_ids + list(policy["generation"]["stop_token_ids"])
        self.collator = RolloutCrossTokenizerCollator(
            student_tokenizer,
            [teacher_tokenizer],
            [
                TokenAligner(
                    student_tokenizer,
                    teacher_tokenizer,
                    self.cfg["projection_matrix_path"],
                    self.cfg["alignment_max_combination_length"],
                )
            ],
            policy["max_total_sequence_length"],
            [self.cfg["teacher_max_sequence_length"]],
            divisor,
            [1],
            [token for token in stop_ids if token is not None],
        )

    def __call__(self, batch, policy_version):
        """Align first, then score bounded teacher microbatches; always close HTTP."""
        tokenizer = self.collator.student_tokenizer
        prompts, prompt_ids, rows = [], [], []
        for index, messages in enumerate(batch["message_log"]):
            params = batch["full_result"][index]["responses_create_params"]
            for key in ("temperature", "top_p"):
                if self.fresh_rollouts and key in params and params[key] != 1:
                    raise ValueError("U7 rollout sampling must use " + key + "=1")
            if set(params) & {"tools", "instructions", "previous_response_id"}:
                raise ValueError("xtoken_topk currently supports text-only single-turn Gym rollouts")
            prompt = params["input"]
            if len(prompt) != 1 or prompt[0]["role"] != "user" or not isinstance(prompt[0]["content"], str):
                raise ValueError("xtoken_topk requires one text user message")
            if len(messages) not in (1, 2) or messages[0]["role"] != "user":
                raise ValueError("xtoken_topk rejects tool/multiturn message logs")
            if len(messages) == 2 and (messages[1]["role"] != "assistant" or "generation_logprobs" not in messages[1]):
                raise ValueError("Expected one sampled assistant response")
            expected = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            actual = messages[0]["token_ids"].tolist()
            if actual != list(expected):
                raise ValueError("Gym student prompt IDs differ from the configured chat template")
            prompts.append(prompt)
            prompt_ids.append(actual)
            rows.append(actual + (messages[1]["token_ids"].tolist() if len(messages) == 2 else []))
        generation = {
            "output_ids": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(row, dtype=torch.long) for row in rows],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "unpadded_sequence_lengths": torch.tensor([len(row) for row in rows]),
        }
        prompt_batch = BatchedDataDict(
            prompts=prompts, sample_mask=batch["loss_multiplier"], idx=list(range(len(rows)))
        )
        data = self.collator(prompt_batch, generation, prompt_ids, policy_version, policy_version)
        data["training_version"] = torch.full((len(rows),), policy_version, dtype=torch.long)
        if "behavior_version" in batch:
            data["behavior_version"] = batch["behavior_version"].clone()
        if (
            data["teacher_0_input_ids"].numel() * (self.cfg["remote"]["top_k"] + 1)
            > self.cfg["remote"]["max_score_elements"]
        ):
            raise ValueError("Global top-k batch exceeds max_score_elements")
        optional_alias = self.opd["default_teacher_alias"] if "default_teacher_alias" in self.opd else None
        strict = self.opd["strict_agent_name_match"] if "strict_agent_name_match" in self.opd else False
        specs = resolve_teacher_specs(
            batch["agent_ref"], self.opd["teacher_model_by_agent_name"], optional_alias, strict
        )
        if len({(spec["url"], spec["model"]) for spec in specs}) != 1:
            raise ValueError("A batch must route to one teacher URL/model pair")
        client = XTokenTopKLogprobClient(self.cfg["remote"])
        try:
            scored = []
            size = self.cfg["teacher_micro_batch_size"]
            for start in range(0, len(rows), size):
                scored.append(
                    asyncio.run(
                        client.score_group(
                            specs[start],
                            data["teacher_0_input_ids"][start : start + size],
                            data["teacher_0_input_lengths"][start : start + size],
                        )
                    )
                )
            for key in ("topk_ids", "topk_logprobs", "topk_valid"):
                data["teacher_0_" + key] = torch.cat([part[key] for part in scored])
        finally:
            asyncio.run(client.close())
        return data
