"""Gym HTTP rollouts with a synchronous, one-update X-Token version lease."""

import json
import os
from pathlib import Path
import time

import ray
import torch

from nemo_rl.algorithms.x_token.gym_protocol import (
    prepare_example, restore_order, unpack_result, reward_metric_valid,
)
from nemo_rl.algorithms.x_token.on_policy import OnPolicyRollout
from nemo_rl.algorithms.x_token.gym_compat import gym_actor_runtime_env, reject_inherited_gym_config, verify_gym_source
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class NemoGymPromptCollator:
    """Carry real Gym rows; Gym's placeholder message_log is not a prompt."""

    def __call__(self, data):
        return BatchedDataDict(
            gym_examples=[datum["extra_env_info"] for datum in data],
            sample_mask=torch.tensor([datum["loss_multiplier"] for datum in data]),
            idx=[datum["idx"] for datum in data],
        )


class NemoGymOnPolicyRollout(OnPolicyRollout):
    """Finish each Gym batch before scoring/updating; rewards are metrics only."""

    def __init__(self, *args, environment, gym_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment = environment
        self.gym_config = gym_config

    def generate(self, prompt_batch, *, training):
        if self.pending_rollout is not None:
            raise RuntimeError("Previous Gym rollout has not completed its update")
        examples, prompts = [], []
        for index, example in enumerate(prompt_batch["gym_examples"]):
            row, prompt = prepare_example(
                example, index, self.generation_config["max_new_tokens"]
            )
            examples.append(row)
            prompts.append(prompt)
        prompt_ids = [list(self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=False
        )) for prompt in prompts]
        if any(not ids or len(ids) > self.knobs["max_prompt_tokens"] for ids in prompt_ids):
            raise ValueError("Gym prompt exceeds token budget; no implicit truncation")
        returned = []
        stream = None
        try:
            if self.synced_version != self.policy_version:
                print("XTOKEN_REFIT_REQUEST=" + json.dumps({
                    "policy_version": self.policy_version,
                    "refit_buffer_size_gb": self.knobs["refit_buffer_size_gb"],
                    "total_buffer_bytes": int(self.knobs["refit_buffer_size_gb"] * (1024**3)),
                }), flush=True)
                self.refit(self.policy, self.generation, colocated_inference=True,
                           _refit_buffer_size_gb=self.knobs["refit_buffer_size_gb"])
                self.synced_version = self.policy_version
            elif not self.generation.prepare_for_generation():
                raise RuntimeError("Could not wake student generation")
            deadline = time.monotonic() + self.gym_config["result_timeout_s"]
            stream = self.environment.run_rollouts.remote(examples, self.tokenizer, "xtoken_gym")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Gym rollout batch exceeded its deadline")
                ready, _ = ray.wait([stream], timeout=remaining)
                if not ready:
                    raise TimeoutError("Gym rollout batch exceeded its deadline")
                try:
                    reference = next(stream)
                except StopIteration:
                    break
                index, result, _ = ray.get(reference, timeout=max(deadline - time.monotonic(), 0.0))
                returned.append((index, result))
        except BaseException:
            if stream is not None:
                ray.cancel(stream)
            raise
        finally:
            if not self.generation.finish_generation():
                raise RuntimeError("Could not release student generation memory")
        results = restore_order(returned, len(examples))
        rows, rewards, mask = [], [], prompt_batch["sample_mask"].clone()
        for index, result in enumerate(results):
            row, valid, reward = unpack_result(result, prompt_ids[index])
            if len(row) - len(prompt_ids[index]) > self.generation_config["max_new_tokens"]:
                raise ValueError("Gym exceeded requested max_output_tokens")
            rows.append(row)
            mask[index] *= valid
            if valid and reward_metric_valid(examples[index]):
                rewards.append(reward)
        maximum = max(map(len, rows))
        ids = torch.full((len(rows), maximum), self.tokenizer.pad_token_id, dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
        prepared = BatchedDataDict(prompts=prompts, sample_mask=mask, idx=prompt_batch["idx"])
        generation = {"output_ids": ids, "unpadded_sequence_lengths": torch.tensor(list(map(len, rows)))}
        self.rollout_counter += 1
        batch = self.response_collator(prepared, generation, prompt_ids,
                                      self.policy_version, self.rollout_counter)
        if training:
            self.pending_rollout = (self.policy_version, self.rollout_counter)
        print("XTOKEN_ROLLOUT=" + json.dumps({
            "backend": "nemo_gym", "policy_version": self.policy_version,
            "rollout_id": self.rollout_counter, "training": training,
            "batch_size": len(rows), "response_tokens": int(batch["token_mask"].sum()),
            "valid_samples": int((mask > 0).sum()),
            "gym_reward_mean": sum(rewards) / len(rewards) if rewards else None,
            "gym_reward_sample_count": len(rewards),
            "gym_code_qa_scores_excluded": sum(not reward_metric_valid(example) for example in examples),
            "reward_used_in_loss": False,
        }), flush=True)
        return batch

    def close(self):
        try:
            ray.get(self.environment.shutdown.remote(), timeout=self.gym_config["result_timeout_s"])
        finally:
            ray.kill(self.environment)


def create_gym_environment(config, generation):
    """Reuse a pre-provisioned Gym interpreter and venv store; never bootstrap."""
    from nemo_rl.environments.nemo_gym import NemoGym, NemoGymConfig

    cfg = config.on_policy["nemo_gym"]
    reject_inherited_gym_config(os.environ)
    verify_gym_source(cfg["source_root"], cfg["source_contract"])
    source = str(Path(__file__).resolve().parents[3])
    runtime = gym_actor_runtime_env(cfg, source, os.environ)
    venvs = Path(runtime["env_vars"]["NEMO_GYM_VENV_DIR"])
    global_config = dict(cfg["global_config"])
    gym_source = Path(cfg["source_root"]).resolve()
    for path in global_config["config_paths"]:
        resolved = Path(path).resolve()
        resolved.relative_to(gym_source)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
    global_config["uv_venv_dir"] = str(venvs)
    global_config["skip_venv_if_present"] = True
    actor = NemoGym.options(
        num_cpus=1, max_restarts=0, max_task_retries=0,
        runtime_env=runtime,
    ).remote(NemoGymConfig(
        model_name=generation.cfg["model_name"],
        base_urls=generation.dp_openai_server_base_urls,
        ray_num_gpus_per_node=0, ray_namespace=ray.get_runtime_context().namespace,
        initial_global_config_dict=global_config,
        skip_load_from_dotenv=cfg["skip_load_from_dotenv"],
        retain_truncated_token_ids=cfg["retain_truncated_token_ids"],
        invalid_tool_call_patterns=[], thinking_tags=[],
    ))
    try:
        ray.get(actor._spinup.remote(), timeout=cfg["result_timeout_s"])
    except BaseException:
        try:
            ray.get(actor.shutdown.remote(), timeout=cfg["result_timeout_s"])
        finally:
            ray.kill(actor)
        raise
    return actor
