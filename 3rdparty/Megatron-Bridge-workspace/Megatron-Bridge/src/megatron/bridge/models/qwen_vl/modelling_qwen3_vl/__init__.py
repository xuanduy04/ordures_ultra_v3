

"""Qwen3 VL model providers and configurations."""

# Core model components
# Bridges for HuggingFace to Megatron conversion
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel  # noqa: F401
from megatron.bridge.models.qwen_vl.qwen3_vl_bridge import Qwen3VLBridge, Qwen3VLMoEBridge

# Dense and MoE model providers
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import (
    Qwen3VLModelProvider,
    Qwen3VLMoEModelProvider,
)


__all__ = [
    "Qwen3VLModel",
    "Qwen3VLModelProvider",
    "Qwen3VLMoEModelProvider",
    "Qwen3VLBridge",
    "Qwen3VLMoEBridge",
]
