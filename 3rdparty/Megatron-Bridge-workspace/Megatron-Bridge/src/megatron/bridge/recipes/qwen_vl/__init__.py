

# Qwen3 models
from .qwen3_vl import (
    qwen3_vl_8b_finetune_config,
    qwen3_vl_8b_pretrain_config,
    qwen3_vl_30b_a3b_finetune_config,
    qwen3_vl_30b_a3b_pretrain_config,
    qwen3_vl_235b_a22b_finetune_config,
    qwen3_vl_235b_a22b_pretrain_config,
)


__all__ = [
    # Qwen3-VL pretrain configs
    "qwen3_vl_8b_pretrain_config",
    "qwen3_vl_30b_a3b_pretrain_config",
    "qwen3_vl_235b_a22b_pretrain_config",
    # Qwen3-VL finetune configs (with PEFT support)
    "qwen3_vl_8b_finetune_config",
    "qwen3_vl_30b_a3b_finetune_config",
    "qwen3_vl_235b_a22b_finetune_config",
]
