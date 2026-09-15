

"""Custom Llama model implementation for NeMo Automodel."""

from nemo_automodel.components.models.llama.model import LlamaForCausalLM, build_llama_model

__all__ = ["LlamaForCausalLM", "build_llama_model"]
