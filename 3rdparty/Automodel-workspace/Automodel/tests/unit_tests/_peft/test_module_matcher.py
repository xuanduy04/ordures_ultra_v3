

from nemo_automodel.shared.import_utils import safe_import
HAS_TE, transformer_engine = safe_import("transformer_engine")
from nemo_automodel.components._peft.module_matcher import _is_linear_module
import pytest
import torch.nn as nn
import torch

@pytest.mark.parametrize(("module", "expected"),
    [(nn.Linear(10, 10), True),
    (nn.Conv1d(10, 10, 1), False),
    (nn.Conv2d(10, 10, 1), False),
    (nn.Conv3d(10, 10, 1), False),
    (nn.ConvTranspose1d(10, 10, 1), False),
    (nn.ConvTranspose2d(10, 10, 1), False),
    (nn.ConvTranspose3d(10, 10, 1), False),
])
def test_is_linear_module(module, expected):
    assert _is_linear_module(module) == expected

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAS_TE, reason="transformer_engine is not installed")
def test_is_linear_module_transformer_engine():
    assert _is_linear_module(transformer_engine.pytorch.Linear(10, 10))
