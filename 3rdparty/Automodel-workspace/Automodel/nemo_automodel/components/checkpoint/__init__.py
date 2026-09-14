
import torch
from packaging.version import parse as vparse

from ._torch_backports import apply_async_checkpoint_patch as _nemo__apply_async_patch
from ._torch_backports import apply_patches as _nemo__apply_patches

if vparse(torch.__version__).base_version <= "2.7.1":
    _nemo__apply_patches()

if vparse(torch.__version__).base_version >= "2.9.0":
    _nemo__apply_async_patch()
