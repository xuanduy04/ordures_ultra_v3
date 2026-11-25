from __future__ import annotations

import torch

from nemo_rl.algorithms.advantage_estimator.base_advantage_estimator import BaseAdvantageEstimator


class JamesSteinAdvantageEstimator(BaseAdvantageEstimator):
    name = "JSPO"
    """JSPO advantage estimator: RLOO baseline shrunk toward the leave-one-prompt-out batch mean.

    The baseline of response j to prompt i is (1 - λ_i) * (mean of the other responses to prompt i) + λ_i * (mean of the
    other prompts' means), with λ_i = v / (v + s + 1e-6) as in the official code: v is the mean of the other prompts'
    unbiased reward variances divided by their response counts, s is the variance of the other prompts' means. Rewards
    are not normalized. Every prompt needs at least 2 responses.

    Args:
        per_environment (Optional, default False): If True, the other prompts are those of the same environment (samples
            sharing an ``agent_ref["name"]``) instead of the whole batch. A prompt alone in its environment (or batch)
            gets λ = 0, i.e. the RLOO baseline.
    """

    @property
    def is_group_normalization(self) -> bool:
        return False

    def __init__(self, estimator_config: dict, loss_config: dict):
        super().__init__(estimator_config, loss_config)
        self.last_metrics: dict[str, float] = {}
        # Baseline usage
        self._ignore_non_default(
            "use_leave_one_out_baseline", True, "JSPO without leave-one-out baseline is not defined."
        )
        self._ignore_non_default("minus_baseline", True, "JSPO without baseline is just 'REINFORCE (plain)'")
        self._ignore_non_default("batch_level_baseline", False, "JSPO already shrinks toward the batch-level baseline.")

        # Normalization
        self._ignore_non_default("normalize_rewards", False, "JSPO with reward normalization is not defined.")

        # Per-environment
        self.per_environment = estimator_config.get("per_environment", False)

        # Name
        if self.per_environment:
            self.name += " (per-environment baseline)"

    def compute_advantage(self, prompt_ids, rewards, mask, agent_refs=None, **kwargs):
        """Compute JSPO advantages.

        Args:
            prompt_ids: Tensor of shape [batch_size, ...] identifying which prompt each sample belongs to.
            rewards: Tensor of shape [batch_size] containing reward for each sample.
            mask: Response token mask of shape [batch_size, seq_len], 1 for valid response tokens, 0 for padding. Used
                only for expanding advantages to token-level shape.
            agent_refs: List of batch_size NeMo-Gym agent references, required if `per_environment` is True.
            **kwargs: Additional arguments (unused).

        Returns:
            Advantages tensor of shape [batch_size, seq_len].
        """
        unique_prompts, group_ids = torch.unique(prompt_ids.to(rewards.device), dim=0, return_inverse=True)
        if self.per_environment:
            if agent_refs is None:
                raise ValueError("JSPO's per-environment baseline requires `agent_refs`")

            env_of_name = {}
            env_ids = torch.tensor(
                [env_of_name.setdefault(agent_ref["name"], len(env_of_name)) for agent_ref in agent_refs],
                device=rewards.device,
            )
        else:
            env_ids = torch.zeros_like(group_ids)

        zeros_per_group = rewards.new_zeros(len(unique_prompts))
        m = zeros_per_group.index_add(0, group_ids, torch.ones_like(rewards))
        total = zeros_per_group.index_add(0, group_ids, rewards)
        mu = total / m
        v = (zeros_per_group.index_add(0, group_ids, rewards.square()) - total * mu) / (m - 1) / m

        group_env = group_ids.new_zeros(len(unique_prompts)).scatter(0, group_ids, env_ids)
        zeros_per_env = rewards.new_zeros(int(env_ids.max()) + 1)
        others = (zeros_per_env.index_add(0, group_env, torch.ones_like(mu))[group_env] - 1).clamp(min=1)
        mu_bar = (zeros_per_env.index_add(0, group_env, mu)[group_env] - mu) / others
        v_bar = (zeros_per_env.index_add(0, group_env, v)[group_env] - v) / others
        s_bar = (zeros_per_env.index_add(0, group_env, mu.square())[group_env] - mu.square()) / others - mu_bar.square()

        lam = (v_bar / (v_bar + s_bar + 1e-6))[group_ids]
        baseline = (1 - lam) * (total[group_ids] - rewards) / (m[group_ids] - 1) + lam * mu_bar[group_ids]
        self.last_metrics = {"jspo/lambda_mean": lam.mean().item()}

        return (rewards - baseline).unsqueeze(-1).expand(mask.shape)
