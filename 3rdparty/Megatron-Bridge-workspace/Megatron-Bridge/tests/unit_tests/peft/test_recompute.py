

"""Unit tests for PEFT-specific recompute helpers."""

from types import SimpleNamespace

import torch

from megatron.bridge.peft import recompute as recompute_mod
from megatron.bridge.peft.recompute import maybe_enable_recompute_inputs_grad


class DummyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


class DummyTransformerBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_input_requires_grad = None

    def forward(self, hidden_states, *args, **kwargs):
        self.last_input_requires_grad = hidden_states.requires_grad
        return hidden_states


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(recompute_method="uniform")
        self.block = DummyTransformerBlock()

        # Frozen base parameter (not trainable)
        self.base = torch.nn.Linear(1, 1, bias=False)
        self.base.weight.requires_grad = False

        # Trainable adapter parameter whose name contains ".adapter."
        # Use a ModuleDict with key "adapter" so that the full parameter
        # name includes the expected substring (".adapter.") used by
        # maybe_enable_recompute_inputs_grad.
        self.adapter = torch.nn.ModuleDict({"adapter": DummyAdapter()})

    def modules(self):
        for module in super().modules():
            yield module


def _patch_transformer_block(monkeypatch):
    import megatron.core.transformer.transformer_block as transformer_block

    monkeypatch.setattr(
        transformer_block,
        "TransformerBlock",
        DummyTransformerBlock,
        raising=False,
    )


def test_maybe_enable_recompute_inputs_grad_patches_block(monkeypatch):
    _patch_transformer_block(monkeypatch)
    recompute_mod.PEFT_RECOMPUTE_PATCHED.clear()

    model = DummyModel()
    patched_registry = maybe_enable_recompute_inputs_grad(model, set())

    assert id(model) in patched_registry

    patched_forward = model.block.forward

    input_tensor = torch.zeros(2, 2)
    assert input_tensor.requires_grad is False

    model.block(input_tensor)
    assert model.block.last_input_requires_grad is True

    # Second invocation should be a no-op (no duplicate patch)
    maybe_enable_recompute_inputs_grad(model, patched_registry)
    assert model.block.forward is patched_forward
