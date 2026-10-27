"""Fresh student rollouts for the v0.6 X-Token training loop."""

import json

import torch

from nemo_rl.data.xtoken_rollout_collate import RolloutCrossTokenizerCollator, _pad_ids
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def validate_on_policy_config(config):
    """Reject sampling/reuse settings outside this strict synchronous path."""
    policy = config.policy
    generation = policy["generation"]
    knobs = config.on_policy
    if generation["backend"] != "vllm" or not generation["colocated"]["enabled"]:
        raise ValueError("On-policy X-Token currently requires colocated vLLM")
    if generation["temperature"] != 1.0 or generation["top_p"] != 1.0 or generation["top_k"] not in (None, -1):
        raise ValueError("Strict OPD requires temperature=1, top_p=1, top_k=null without importance correction")
    if policy["train_global_batch_size"] != config.distillation["num_prompts_per_step"]:
        raise ValueError("Exactly one global optimizer update is required per fresh rollout batch")
    if knobs["max_prompt_tokens"] + generation["max_new_tokens"] > policy["max_total_sequence_length"]:
        raise ValueError("Prompt plus generation budget exceeds student context")
    if knobs["include_eos_loss"]:
        raise ValueError("Cross-tokenizer EOS alignment is not implemented; include_eos_loss must be false")
    if config.loss_fn["dynamic_loss_scaling"] or config.loss_fn["ce_loss_scale"] != 0:
        raise ValueError("OPD recipe requires fixed KD weighting and ce_loss_scale=0")
    if "variant" in config.loss_fn:
        from nemo_rl.algorithms.x_token.variant_loss import MODES

        mode = config.loss_fn["variant"]["mode"]
        if mode not in MODES or len(config.teachers) != 1:
            raise ValueError("Choose an explicit single-teacher variant mode")
        remote = config.teachers[0].remote
        if remote is None or (mode.endswith("_full") and remote["score_mode"] != "full"):
            raise ValueError("Variant score contract unavailable: exact _full modes need a full-score remote endpoint")
        dp = config.cluster["num_nodes"] * config.cluster["gpus_per_node"]
        if policy["train_global_batch_size"] % (dp * policy["train_micro_batch_size"]):
            raise ValueError("Variant global batch must be divisible by DP size times microbatch size")
    if knobs["max_prompt_tokens"] <= 0 or knobs["refit_buffer_size_gb"] <= 0:
        raise ValueError("Prompt and refit buffer limits must be positive")
    if "rollout_backend" in knobs and knobs["rollout_backend"] == "nemo_gym":
        from nemo_rl.algorithms.x_token.gym_compat import validate_ray_address, validate_teacher_endpoint

        validate_teacher_endpoint(config)
        if "worker_python" in knobs["nemo_gym"] or not knobs["nemo_gym"]["actor_python"]:
            raise ValueError("Re-render Gym config with explicit actor_python for r7")
        validate_ray_address(knobs["nemo_gym"]["ray_address"])
        if not generation["vllm_cfg"]["async_engine"] or not generation["vllm_cfg"]["expose_http_server"]:
            raise ValueError("Gym requires an async vLLM engine with its HTTP server exposed")
        if generation["vllm_cfg"]["skip_tokenizer_init"]:
            raise ValueError("Gym HTTP generation needs the student tokenizer initialized")
        if knobs["prompt_source"] != "nemo_gym":
            raise ValueError("Gym batches must retain their real responses_create_params input")
        train = config.data["train"]
        if isinstance(train, list):
            if len(train) != 1:
                raise ValueError("Initial Gym integration requires one explicit dataset")
            train = train[0]
        if train["dataset_name"] != "NemoGymDataset" or train["processor"] != "nemo_gym_data_processor":
            raise ValueError("Use the existing Gym dataset and processor for Gym rollouts")
        if config.distillation["val_period"] or config.distillation["val_at_start"] or config.distillation["val_at_end"]:
            raise ValueError("Initial Gym integration requires separate held-out evaluation")


class OnPolicyRollout:
    """Synchronize weights before sampling and lease each batch for one update.

    This changes the source of training trajectories. The configured X-Token
    surrogate remains unchanged; no trajectory score-function term is added.
    """

    def __init__(self, policy, generation, tokenizer, response_collator,
                 generation_config, knobs, initial_version, refit):
        self.policy = policy
        self.generation = generation
        self.tokenizer = tokenizer
        self.response_collator = response_collator
        self.generation_config = generation_config
        self.knobs = knobs
        self.policy_version = initial_version
        self.synced_version = None
        self.pending_rollout = None
        self.rollout_counter = 0
        self.refit = refit

    def generate(self, prompt_batch, *, training):
        """Sample with the current weights; prefetch contains prompts only."""
        if self.pending_rollout is not None:
            raise RuntimeError("Previous rollout has not completed its optimizer update")
        prompt_ids = [list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False,
        )) for messages in prompt_batch["prompts"]]
        if any(not isinstance(token, int) for ids in prompt_ids for token in ids):
            raise ValueError("Chat template must return integer token IDs")
        if any(not ids or len(ids) > self.knobs["max_prompt_tokens"] for ids in prompt_ids):
            raise ValueError("Empty or over-budget prompt; truncation is not implicit")
        if self.synced_version != self.policy_version:
            print("XTOKEN_REFIT_REQUEST=" + json.dumps({
                "policy_version": self.policy_version,
                "refit_buffer_size_gb": self.knobs["refit_buffer_size_gb"],
                "total_buffer_bytes": int(self.knobs["refit_buffer_size_gb"] * (1024**3)),
            }), flush=True)
            self.refit(self.policy, self.generation, colocated_inference=True,
                       _refit_buffer_size_gb=self.knobs["refit_buffer_size_gb"])
            self.synced_version = self.policy_version
        else:
            self.generation.prepare_for_generation()
        inputs = BatchedDataDict(
            input_ids=_pad_ids(prompt_ids, self.tokenizer, self.knobs["max_prompt_tokens"], 1),
            input_lengths=torch.tensor(list(map(len, prompt_ids)), dtype=torch.long),
        )
        try:
            result = self.generation.generate(inputs, greedy=False)
        finally:
            self.generation.finish_generation()
        if "requires_behavior_logprobs" in self.knobs and self.knobs["requires_behavior_logprobs"]:
            result["behavior_logprobs"] = result["logprobs"]
        self.rollout_counter += 1
        batch = self.response_collator(prompt_batch, result, prompt_ids,
                                      self.policy_version, self.rollout_counter)
        if training:
            self.pending_rollout = (self.policy_version, self.rollout_counter)
        print("XTOKEN_ROLLOUT=" + json.dumps({"policy_version": self.policy_version,
              "rollout_id": self.rollout_counter, "training": training,
              "batch_size": len(prompt_ids), "response_tokens": int(batch["token_mask"].sum())}), flush=True)
        return batch

    def after_update(self):
        """Advance the behavior version only after a successful policy.train."""
        if self.pending_rollout != (self.policy_version, self.rollout_counter):
            raise RuntimeError("No fresh rollout is leased to this optimizer update")
        self.policy_version += 1
        self.pending_rollout = None


def build_on_policy_rollout(config, student_policy, student_tokenizer,
                            teacher_tokenizers, state):
    """Use the existing production IPC refit and generation interfaces."""
    from nemo_rl.algorithms.grpo import refit_policy_generation
    from nemo_rl.algorithms.x_token import TokenAligner

    validate_on_policy_config(config)
    teacher_configs = [teacher.policy_config() for teacher in config.teachers]
    aligners = [None if teacher.projection_matrix_path is None else TokenAligner(
        student_tokenizer, teacher_tokenizers[index], teacher.projection_matrix_path,
        config.data["alignment_max_combination_length"],
    ) for index, teacher in enumerate(config.teachers)]
    if "variant" in config.loss_fn:
        from nemo_rl.algorithms.x_token.variant_alignment import wrap_aligners

        aligners = wrap_aligners(aligners, student_tokenizer, teacher_tokenizers)
    generation = student_policy.xtoken_generation
    gym_backend = "rollout_backend" in config.on_policy and config.on_policy["rollout_backend"] == "nemo_gym"
    stop_ids = generation.cfg["stop_token_ids"]
    if gym_backend:
        stop_ids = [student_tokenizer.eos_token_id]
        if "eos_token_ids" in config.on_policy["nemo_gym"]:
            stop_ids = config.on_policy["nemo_gym"]["eos_token_ids"]
        if not stop_ids or any(type(token) is not int or token < 0 for token in stop_ids):
            raise ValueError("Gym requires explicit valid student terminal token IDs")
    collator = RolloutCrossTokenizerCollator(
        student_tokenizer, teacher_tokenizers, aligners,
        config.policy["max_total_sequence_length"],
        [item["max_total_sequence_length"] for item in teacher_configs],
        config.policy["make_sequence_length_divisible_by"],
        [item["make_sequence_length_divisible_by"] for item in teacher_configs],
        stop_ids,
    )
    if gym_backend:
        from nemo_rl.algorithms.x_token.gym_rollout import (
            NemoGymOnPolicyRollout, create_gym_environment,
        )

        environment = create_gym_environment(config, generation)
        return NemoGymOnPolicyRollout(
            student_policy, generation, student_tokenizer, collator,
            generation.cfg, config.on_policy, state["total_steps"],
            refit_policy_generation, environment=environment,
            gym_config=config.on_policy["nemo_gym"],
        )
    return OnPolicyRollout(student_policy, generation, student_tokenizer, collator,
                           generation.cfg, config.on_policy, state["total_steps"],
                           refit_policy_generation)
