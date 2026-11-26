from __future__ import annotations

import torch

from nemo_rl.algorithms.advantage_estimator.base import BaseAdvantageEstimator

DIFFICULTY_SMOOTH_ALPHA = 0.8
DIFFICULTY_GAMMA = 1.0
EPSILON = 1e-6


class TMNAdvantageEstimator(BaseAdvantageEstimator):
    name = "TMN"
    """TMN advantage estimator: prompt-centered rewards divided by a per-environment scale.

    As in the official code, the advantage of response j to prompt i is (r_ij - μ_i) / (σ + EPSILON): μ_i is the mean
    reward of prompt i and σ is the root of the mean unbiased reward variance over the batch's prompts of the same
    environment (samples sharing an ``agent_ref["name"]``). A prompt with a single response gets μ = 0 and variance 1.

    Args:
        use_reweight (Optional, default False): If True (TMN-Reweight), multiply the advantages of the responses above
            μ_i by w_i = exp(DIFFICULTY_GAMMA * (0.5 - p_i)) and the others by 1 / w_i. p_i is the fraction of the
            responses to prompt i above DIFFICULTY_SMOOTH_ALPHA * μ_i + (1 - DIFFICULTY_SMOOTH_ALPHA) * (the mean μ of
            the environment's prompts), or above that mean alone for a prompt with a single response.
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        super().__init__(estimator_config, loss_config)
        self.last_metrics: dict[str, float] = {}
        # Baseline usage
        self._ignore_non_default("minus_baseline", True, "TMN without baseline is not defined.")
        self._ignore_non_default("use_leave_one_out_baseline", False, "TMN with leave-one-out baseline is not defined.")
        self._ignore_non_default("batch_level_baseline", False, "TMN with batch-level baseline is not defined.")

        # Normalization
        self._ignore_non_default("normalize_rewards", True, "TMN without normalization is just 'DR.GRPO'.")

        # Per-environment
        self._ignore_non_default("per_environment", True, "TMN is innately per-environment.")

        # Reweight
        self.use_reweight = estimator_config.get("use_reweight", False)

        # Name
        if self.use_reweight:
            self.name += "-Reweight"

    def compute_advantage(self, prompt_ids, rewards, mask, agent_refs=None, **kwargs):
        """Compute TMN advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size, ...] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding. Used
                only for expanding advantages to token-level shape.
            agent_refs: List of batch_size NeMo-Gym agent references.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        if agent_refs is None:
            raise ValueError("TMN requires `agent_refs`")

        unique_prompts, group_ids = torch.unique(prompt_ids.to(rewards.device), dim=0, return_inverse=True)
        env_of_name = {}
        env_ids = torch.tensor(
            [env_of_name.setdefault(agent_ref["name"], len(env_of_name)) for agent_ref in agent_refs],
            device=rewards.device,
        )

        zeros_per_group = rewards.new_zeros(len(unique_prompts))
        m = zeros_per_group.index_add(0, group_ids, torch.ones_like(rewards))
        mu = zeros_per_group.index_add(0, group_ids, rewards) / m
        var = (zeros_per_group.index_add(0, group_ids, rewards.square()) / m - mu.square()).clamp(min=0) * m / (m - 1)
        single = m == 1
        mu = mu.masked_fill(single, 0.0)
        var = var.masked_fill(single, 1.0)

        group_env = group_ids.new_zeros(len(unique_prompts)).scatter(0, group_ids, env_ids)
        zeros_per_env = rewards.new_zeros(len(env_of_name))
        prompts_per_env = zeros_per_env.index_add(0, group_env, torch.ones_like(mu))
        sigma = (zeros_per_env.index_add(0, group_env, var) / prompts_per_env).sqrt()
        advantages = (rewards - mu[group_ids]) / (sigma[env_ids] + EPSILON)

        if self.use_reweight:
            mu_env = (zeros_per_env.index_add(0, group_env, mu) / prompts_per_env)[group_env]
            smoothed_mu = DIFFICULTY_SMOOTH_ALPHA * mu + (1 - DIFFICULTY_SMOOTH_ALPHA) * mu_env
            threshold = torch.where(single, mu_env, smoothed_mu)
            pass_rate = zeros_per_group.index_add(0, group_ids, (rewards > threshold[group_ids]).to(rewards.dtype)) / m
            weight = torch.exp(DIFFICULTY_GAMMA * (0.5 - pass_rate))[group_ids]
            advantages = advantages * torch.where(rewards > mu[group_ids], weight, 1 / weight)
            self.last_metrics = {"tmn/mean_pass_rate": pass_rate.mean().item()}

        return advantages.unsqueeze(-1).expand(mask.shape)
