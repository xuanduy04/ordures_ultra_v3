from __future__ import annotations

import torch

from nemo_rl.algorithms.advantage_estimator.base import BaseAdvantageEstimator

EPSILON = 1e-6


class TANAdvantageEstimator(BaseAdvantageEstimator):
    name = "TAN"
    """TAN advantage estimator: GRPO advantages standardized per environment over their valid tokens.

    As in the official code, response j to prompt i first gets the GRPO advantage A_ij = (r_ij - μ_i) / (σ_i + EPSILON):
    μ_i and σ_i are the mean and unbiased std of prompt i's rewards, or 0 and 1 for a prompt with a single response.
    The advantages of each environment (samples sharing an ``agent_ref["name"]``) then become (A_ij - μ) / (σ + EPSILON),
    where μ and σ are the mean and unbiased std of A over every valid token of the environment, so longer responses weigh
    more. An environment with at most one valid token gets zero advantages.
    """

    def __init__(self, estimator_config: dict, loss_config: dict):
        super().__init__(estimator_config, loss_config)
        # Baseline usage
        self._ignore_non_default("minus_baseline", True, "TAN without baseline is not defined.")
        self._ignore_non_default("use_leave_one_out_baseline", False, "TAN with leave-one-out baseline is not defined.")
        self._ignore_non_default("batch_level_baseline", False, "TAN with batch-level baseline is not defined.")

        # Normalization
        self._ignore_non_default("normalize_rewards", True, "TAN without normalization is not defined.")

        # Per-environment
        self._ignore_non_default("per_environment", True, "TAN is innately per-environment.")

    def compute_advantage(self, prompt_ids, rewards, mask, agent_refs=None, **kwargs):
        """Compute TAN advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size, ...] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding. Used
                for: (1) expanding advantages to token-level shape, (2) per-environment normalization that only
                considers valid tokens.
            agent_refs: List of batch_size NeMo-Gym agent references.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        if agent_refs is None:
            raise ValueError("TAN requires `agent_refs`")

        unique_prompts, group_ids = torch.unique(prompt_ids.to(rewards.device), dim=0, return_inverse=True)
        env_of_name = {}
        env_ids = torch.tensor(
            [env_of_name.setdefault(agent_ref["name"], len(env_of_name)) for agent_ref in agent_refs],
            device=rewards.device,
        )

        zeros_per_group = rewards.new_zeros(len(unique_prompts))
        m = zeros_per_group.index_add(0, group_ids, torch.ones_like(rewards))
        mu = zeros_per_group.index_add(0, group_ids, rewards) / m
        var = zeros_per_group.index_add(0, group_ids, (rewards - mu[group_ids]).square()) / (m - 1)
        single = m == 1
        mu = mu.masked_fill(single, 0.0)
        var = var.masked_fill(single, 1.0)
        advantages = (rewards - mu[group_ids]) / (var.sqrt()[group_ids] + EPSILON)

        tokens = mask.sum(-1, dtype=rewards.dtype)
        zeros_per_env = rewards.new_zeros(len(env_of_name))
        tokens_per_env = zeros_per_env.index_add(0, env_ids, tokens)
        mu_env = (zeros_per_env.index_add(0, env_ids, tokens * advantages) / tokens_per_env)[env_ids]
        var_env = zeros_per_env.index_add(0, env_ids, tokens * (advantages - mu_env).square()) / (tokens_per_env - 1)
        advantages = (advantages - mu_env) / (var_env.sqrt()[env_ids] + EPSILON)

        return advantages.masked_fill((tokens_per_env <= 1)[env_ids], 0.0).unsqueeze(-1).expand(mask.shape)
