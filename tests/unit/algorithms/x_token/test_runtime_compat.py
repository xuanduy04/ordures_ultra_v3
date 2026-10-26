"""Compatibility contracts exercised before allocating GPUs."""

import pytest

from nemo_rl.models.automodel.runtime_compat import create_fsdp2_manager, load_base_checkpoint, tensor_to_hf
from nemo_rl.models.automodel.runtime_compat import use_hf_meta_init, hf_model_from_config
from types import SimpleNamespace
from unittest.mock import patch


def test_legacy_constructor_is_unchanged():
    class Legacy:
        def __init__(self, dp_size, **kwargs):
            self.dp_size, self.kwargs = dp_size, kwargs

    manager = create_fsdp2_manager(Legacy, dp_size=8, cp_size=1)
    assert manager.dp_size == 8 and manager.kwargs == {"cp_size": 1}


def test_modern_checkpoint_omits_only_unused_peft_argument():
    class Checkpointer:
        def load_base_model(self, model, model_name):
            return model, model_name

    assert load_base_checkpoint(Checkpointer(), "weights", model_name="fixture", peft_init_method=None) == ("weights", "fixture")
    with pytest.raises(NotImplementedError):
        load_base_checkpoint(Checkpointer(), "weights", model_name="fixture", peft_init_method="normal")


def test_legacy_checkpoint_preserves_peft_argument():
    class Checkpointer:
        def load_base_model(self, model, peft_init_method):
            return peft_init_method

    assert load_base_checkpoint(Checkpointer(), "weights", peft_init_method="normal") == "normal"


def test_modern_refit_adapter_uses_public_state_dict_conversion():
    class Adapter:
        def to_hf(self, state, exclude_key_regex):
            assert exclude_key_regex == r".*_extra_state.*"
            return {"hf.weight": state["fused.weight"]}

    assert tensor_to_hf(Adapter(), "fused.weight", 7, quantization=False) == [("hf.weight", 7)]
    with pytest.raises(NotImplementedError):
        tensor_to_hf(Adapter(), "fused.weight", 7, quantization=True)


def test_legacy_refit_adapter_preserves_quantization():
    class Adapter:
        def convert_single_tensor_to_hf(self, name, tensor, exclude_key_regex, quantization):
            return [(name, tensor, quantization)]

    assert tensor_to_hf(Adapter(), "weight", 9, quantization=True) == [("weight", 9, True)]


def test_hf_meta_initialization_keeps_production_hf4_and_custom_models():
    config = SimpleNamespace(architectures=["Qwen3ForCausalLM"])
    registry = SimpleNamespace(model_arch_name_to_cls={})
    with patch("transformers.__version__", "4.57.1"):
        assert not use_hf_meta_init(config, registry)
    with patch("transformers.__version__", "5.3.0"):
        assert use_hf_meta_init(config, registry)
        registry.model_arch_name_to_cls["Qwen3ForCausalLM"] = object
        assert not use_hf_meta_init(config, registry)


def test_hf_meta_initializer_rejects_ignored_backend_options():
    with pytest.raises(NotImplementedError):
        hf_model_from_config(None, None, {"backend": "custom"}, None)
