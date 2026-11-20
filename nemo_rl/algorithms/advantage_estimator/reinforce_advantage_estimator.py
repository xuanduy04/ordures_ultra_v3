from __future__ import annotations

import torch

from nemo_rl.algorithms.advantage_estimator.base_advantage_estimator import BaseAdvantageEstimator
from nemo_rl.algorithms.utils import calculate_baseline_and_std_per_prompt, calculate_batch_level_baseline, calculate_kl


class ReinforcePlusPlusAdvantageEstimator(BaseAdvantageEstimator):
    name = "REINFORCE++"
    """REINFORCE++ advantage estimator with optional baseline subtraction and KL penalty in reward.

    Args:
        minus_baseline: If True, subtract per-prompt mean baseline from rewards.
        normalize_rewards (Optional, default True): If False, skip the mean/std normalization (plain REINFORCE).
        use_kl_in_reward: If True, add KL penalty to reward instead of loss.
        per_environment (Optional, default False): If True, normalize each environment (samples sharing an
            ``agent_ref["name"]``) with its own mean and std instead of across the whole batch. An environment with
            no valid tokens gets zero advantages.
    """

    @property
    def is_group_normalization(self) -> bool:
        return self.minus_baseline and not self.batch_level_baseline and not self.normalize_rewards

    def __init__(self, estimator_config: dict, loss_config: dict):
        # Baseline usage
        self.minus_baseline = estimator_config.get("minus_baseline", False)
        self.use_leave_one_out_baseline = estimator_config.get("use_leave_one_out_baseline", False)
        self.batch_level_baseline = estimator_config.get("batch_level_baseline", False)

        if (not self.minus_baseline) and (self.batch_level_baseline or self.use_leave_one_out_baseline):
            print("[WARNING] minus_baseline=False. Ignoring `batch_level_baseline` and `use_leave_one_out_baseline` settings.")
        
        # Normalization
        self.normalize_rewards = estimator_config.get("normalize_rewards", True)
        
        # Per-environment
        self.per_environment = estimator_config.get("per_environment", False)

        # KL in reward
        self.use_kl_in_reward = loss_config["use_kl_in_reward"]
        self.kl_coef = loss_config["reference_policy_kl_penalty"]
        self.kl_type = loss_config["reference_policy_kl_type"]

        # Name
        if not self.normalize_rewards:
            self.name = "REINFORCE (plain)"
        if self.minus_baseline:
            baseline_level_str = (
                ('env-level' if self.per_environment else 'batch-level')
                if self.batch_level_baseline else 'prompt-level'
            )
            self.name += f" w/ {baseline_level_str} baseline"
            if self.use_leave_one_out_baseline:
                self.name += " (LOO baseline)"
        if self.per_environment and self.normalize_rewards:
            self.name += " (per-environment std normalization)"

    def compute_advantage(
        self, prompt_ids, rewards, mask, logprobs_policy=None, logprobs_reference=None, agent_refs=None, **kwargs
    ):
        """Compute REINFORCE++ advantages with optional KL penalty.

        Args:
            prompt_ids: Tensor of shape [batch_size] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding. Used
                for: (1) expanding advantages to token-level shape, (2) global normalization that only considers valid
                tokens.
            logprobs_policy: Policy log probabilities of shape [batch_size, seq_len], required if use_kl_in_reward.
            logprobs_reference: Reference policy log probabilities of shape [batch_size, seq_len], required if
                use_kl_in_reward.
            agent_refs: List of batch_size NeMo-Gym agent references, required if `per_environment` is True.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len], globally normalized across valid tokens.
        """
        if self.minus_baseline:
            if self.batch_level_baseline:
                if self.per_environment:
                    # env-level baseline
                    raise NotImplementedError("Environment-level baseline not implemented")
                else:
                    # batch-level baseline
                    baseline = calculate_batch_level_baseline(
                        rewards, torch.ones_like(rewards), leave_one_out_baseline=self.use_leave_one_out_baseline
                    )
            else:
                # prompt-level baseline
                baseline, _ = calculate_baseline_and_std_per_prompt(
                    prompt_ids, rewards, torch.ones_like(rewards), leave_one_out_baseline=self.use_leave_one_out_baseline
                )
            adv = rewards - baseline
        else:
            # no baseline
            adv = rewards

        # adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)  # I argue that this is a reward-level issue, not an advantage estimator one
        adv = adv.unsqueeze(-1)
        adv = adv.expand(mask.shape)

        # add kl penalty to reward (token-level)
        if self.use_kl_in_reward and logprobs_policy is not None and logprobs_reference is not None:
            kl = calculate_kl(logprobs_policy, logprobs_reference, kl_type=self.kl_type)
            adv = adv - self.kl_coef * kl

        if not self.normalize_rewards:
            return adv * mask

        if self.per_environment:
            # normalization across the 'environment'
            if agent_refs is None:
                raise ValueError("REINFORCE++'s per-environment std normalization requires `agent_refs`")

            env_of_name = {}
            env_ids = torch.tensor(
                [env_of_name.setdefault(agent_ref["name"], len(env_of_name)) for agent_ref in agent_refs],
                device=adv.device,
            )
            zeros_per_env = adv.new_zeros(len(env_of_name))
            num_tokens = zeros_per_env.index_add(0, env_ids, mask.sum(-1, dtype=adv.dtype))
            adv_mean = (zeros_per_env.index_add(0, env_ids, (adv * mask).sum(-1)) / num_tokens)[env_ids].unsqueeze(-1)
            adv_var = zeros_per_env.index_add(0, env_ids, ((adv - adv_mean).pow(2) * mask).sum(-1)) / num_tokens
            adv_rstd = adv_var.clamp(min=1e-8).rsqrt()[env_ids].unsqueeze(-1)
            adv = (adv - adv_mean) * adv_rstd

            return adv.masked_fill((num_tokens == 0)[env_ids].unsqueeze(-1), 0.0)
        else:
            # global normalization across the batch
            adv_mean = (adv * mask).sum() / mask.sum()
            adv_var = ((adv - adv_mean).pow(2) * mask).sum() / mask.sum()
            adv_rstd = adv_var.clamp(min=1e-8).rsqrt()
            adv = (adv - adv_mean) * adv_rstd
            return adv
