

"""
VLM dataset utilities.

Public API re-exports:
- Makers: functions to build conversation examples from HF datasets
- Providers: classes that build PyTorch datasets bound to HF processors
- Collate fns: model-specific batch builders
"""

from megatron.bridge.data.energon.energon_provider import EnergonProvider
from megatron.bridge.data.vlm_datasets.collate import (
    COLLATE_FNS,
    default_collate_fn,
    nemotron_nano_v2_vl_collate_fn,
    phi4_mm_collate_fn,
    qwen2_5_collate_fn,
)
from megatron.bridge.data.vlm_datasets.conversation_dataset import VLMConversationDataset
from megatron.bridge.data.vlm_datasets.hf_dataset_makers import (
    make_cord_v2_dataset,
    make_cv17_dataset,
    make_llava_video_178k_dataset,
    make_medpix_dataset,
    make_raven_dataset,
    make_rdr_dataset,
)
from megatron.bridge.data.vlm_datasets.hf_provider import HFDatasetConversationProvider
from megatron.bridge.data.vlm_datasets.mock_provider import MockVLMConversationProvider
from megatron.bridge.data.vlm_datasets.preloaded_provider import PreloadedVLMConversationProvider


__all__ = [
    # Makers
    "make_rdr_dataset",
    "make_cord_v2_dataset",
    "make_medpix_dataset",
    "make_cv17_dataset",
    "make_raven_dataset",
    "make_llava_video_178k_dataset",
    # Dataset types/providers
    "VLMConversationDataset",
    "HFDatasetConversationProvider",
    "PreloadedVLMConversationProvider",
    "MockVLMConversationProvider",
    "EnergonProvider",
    # Collation utilities
    "COLLATE_FNS",
    "default_collate_fn",
    "qwen2_5_collate_fn",
    "phi4_mm_collate_fn",
]
