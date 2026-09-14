

"""Configuration classes for automodel-based training in NeMo RL."""

from typing import Any, NamedTuple, Optional

import torch
from nemo_automodel.components._peft.lora import PeftConfig


class RuntimeConfig(NamedTuple):
    """Runtime configuration for model training and inference.

    This contains all validated runtime settings needed for model initialization,
    parallelization, and training.
    """

    # Model loading configuration
    model_class: type
    model_config: Any  # AutoConfig
    hf_config_overrides: dict[str, Any]

    # Attention configuration
    allow_flash_attn_args: bool
    attn_impl: Optional[str]

    # Training/inference settings
    dtype: torch.dtype
    enable_seq_packing: bool
    max_grad_norm: float

    # Memory management
    cpu_offload: bool
    offload_optimizer_for_logprob: bool

    # Generation configuration
    is_generation_colocated: Optional[bool]

    # Reward model flag
    is_reward_model: bool


class ModelAndOptimizerState(NamedTuple):
    """Container for model and optimizer state.

    This named tuple holds all model-related state including the model itself,
    optimizer, scheduler, and metadata about the model type and configuration.
    """

    model: torch.nn.Module
    model_state_dict_keys: list[str]
    optimizer: Optional[torch.optim.Optimizer]
    scheduler: Optional[Any]
    is_hf_model: bool
    is_moe_model: bool
    is_reward_model: bool
    model_class: type
    model_config: Any
    peft_config: Optional[PeftConfig]
    autocast_enabled: bool
