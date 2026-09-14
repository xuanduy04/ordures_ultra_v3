

"""
Ministral 3 Model Bridge and Provider implementations.

This module provides support for Ministral 3 vision-language models (3B, 8B, 14B variants).

Reference: https://huggingface.co/mistralai/Ministral-3-3B-Base-2512

Supported models:
- Ministral-3-3B-Base-2512
- Ministral-3-3B-Instruct-2512
- Ministral-3-3B-Reasoning-2512
- Ministral-3-8B-Base-2512
- Ministral-3-8B-Instruct-2512
- Ministral-3-8B-Reasoning-2512
- Ministral-3-14B-Base-2512
- Ministral-3-14B-Instruct-2512
- Ministral-3-14B-Reasoning-2512

Example usage:
    >>> from megatron.bridge import AutoBridge
    >>> bridge = AutoBridge.from_hf_pretrained("mistralai/Ministral-3-3B-Base-2512")
    >>> provider = bridge.to_megatron_provider()
"""

from megatron.bridge.models.ministral3.ministral3_bridge import Ministral3Bridge
from megatron.bridge.models.ministral3.ministral3_provider import (
    Ministral3ModelProvider,
    Ministral3ModelProvider3B,
    Ministral3ModelProvider8B,
    Ministral3ModelProvider14B,
)
from megatron.bridge.models.ministral3.modeling_ministral3 import Ministral3Model


__all__ = [
    # Bridge
    "Ministral3Bridge",
    # Model
    "Ministral3Model",
    # Model Providers
    "Ministral3ModelProvider",
    "Ministral3ModelProvider3B",
    "Ministral3ModelProvider8B",
    "Ministral3ModelProvider14B",
]
