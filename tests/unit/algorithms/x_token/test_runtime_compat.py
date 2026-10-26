"""Compatibility contracts exercised before allocating GPUs."""

import pytest

from nemo_rl.models.automodel.runtime_compat import create_fsdp2_manager, load_base_checkpoint


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
