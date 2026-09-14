

"""MLA (Multi-Latent Attention) Model Provider.

This module provides a minimal provider for models using Multi-Latent Attention,
such as DeepSeek V2/V3 and Kimi K2.
"""

from dataclasses import dataclass

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer_config import MLATransformerConfig


@dataclass
class MLAModelProvider(MLATransformerConfig, GPTModelProvider):
    """Provider for models using Multi-Latent Attention (MLA).

    This class combines MLATransformerConfig (which provides MLA-specific fields
    like q_lora_rank, kv_lora_rank, qk_head_dim, v_head_dim) with GPTModelProvider
    (which provides the model instantiation logic).

    Model-specific defaults (normalization, activation, fusions, etc.) should be
    configured via MEGATRON_DEFAULTS in the respective bridge classes.

    Used by:
        - DeepSeek V2/V3
        - Kimi K2
        - Other MLA-based models
    """

    pass
