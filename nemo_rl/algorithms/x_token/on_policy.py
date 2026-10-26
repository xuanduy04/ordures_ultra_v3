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
    if knobs["max_prompt_tokens"] <= 0 or knobs["refit_buffer_size_gb"] <= 0:
        raise ValueError("Prompt and refit buffer limits must be positive")


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
    generation = student_policy.xtoken_generation
    collator = RolloutCrossTokenizerCollator(
        student_tokenizer, teacher_tokenizers, aligners,
        config.policy["max_total_sequence_length"],
        [item["max_total_sequence_length"] for item in teacher_configs],
        config.policy["make_sequence_length_divisible_by"],
        [item["make_sequence_length_divisible_by"] for item in teacher_configs],
        generation.cfg["stop_token_ids"],
    )
    return OnPolicyRollout(student_policy, generation, student_tokenizer, collator,
                           generation.cfg, config.on_policy, state["total_steps"],
                           refit_policy_generation)
