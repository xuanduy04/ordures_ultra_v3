

"""Combined projection modules for efficient transformer implementations."""

from nemo_automodel.components.models.common.combined_projection.combined_mlp import CombinedGateUpMLP
from nemo_automodel.components.models.common.combined_projection.combined_qkv import CombinedQKVAttentionMixin

__all__ = ["CombinedQKVAttentionMixin", "CombinedGateUpMLP"]
