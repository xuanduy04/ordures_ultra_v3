

# Qwen2 models
from .qwen2 import (
    qwen2_1p5b_finetune_config,
    qwen2_1p5b_pretrain_config,
    qwen2_7b_finetune_config,
    qwen2_7b_pretrain_config,
    qwen2_72b_finetune_config,
    qwen2_72b_pretrain_config,
    qwen2_500m_finetune_config,
    qwen2_500m_pretrain_config,
    qwen25_1p5b_finetune_config,
    qwen25_1p5b_pretrain_config,
    qwen25_7b_finetune_config,
    qwen25_7b_pretrain_config,
    qwen25_14b_finetune_config,
    qwen25_14b_pretrain_config,
    qwen25_32b_finetune_config,
    qwen25_32b_pretrain_config,
    qwen25_72b_finetune_config,
    qwen25_72b_pretrain_config,
    qwen25_500m_finetune_config,
    qwen25_500m_pretrain_config,
)

# Qwen3 models
from .qwen3 import (
    qwen3_1p7b_finetune_config,
    qwen3_1p7b_pretrain_config,
    qwen3_4b_finetune_config,
    qwen3_4b_pretrain_config,
    qwen3_8b_finetune_config,
    qwen3_8b_pretrain_config,
    qwen3_14b_finetune_config,
    qwen3_14b_pretrain_config,
    qwen3_32b_finetune_config,
    qwen3_32b_pretrain_config,
    qwen3_600m_finetune_config,
    qwen3_600m_pretrain_config,
)

# Qwen3 MoE models
from .qwen3_moe import (
    qwen3_30b_a3b_finetune_config,
    qwen3_30b_a3b_pretrain_config,
    qwen3_235b_a22b_finetune_config,
    qwen3_235b_a22b_pretrain_config,
)

# Qwen3-Next models
from .qwen3_next import (
    qwen3_next_80b_a3b_finetune_config,
    qwen3_next_80b_a3b_pretrain_config,
)


__all__ = [
    # Qwen2 models
    "qwen2_500m_pretrain_config",
    "qwen2_1p5b_pretrain_config",
    "qwen2_7b_pretrain_config",
    "qwen2_72b_pretrain_config",
    "qwen2_500m_finetune_config",
    "qwen2_1p5b_finetune_config",
    "qwen2_7b_finetune_config",
    "qwen2_72b_finetune_config",
    # Qwen2.5 models
    "qwen25_500m_pretrain_config",
    "qwen25_1p5b_pretrain_config",
    "qwen25_7b_pretrain_config",
    "qwen25_14b_pretrain_config",
    "qwen25_32b_pretrain_config",
    "qwen25_72b_pretrain_config",
    "qwen25_500m_finetune_config",
    "qwen25_1p5b_finetune_config",
    "qwen25_7b_finetune_config",
    "qwen25_14b_finetune_config",
    "qwen25_32b_finetune_config",
    "qwen25_72b_finetune_config",
    # Qwen3 models
    "qwen3_600m_pretrain_config",
    "qwen3_1p7b_pretrain_config",
    "qwen3_4b_pretrain_config",
    "qwen3_8b_pretrain_config",
    "qwen3_14b_pretrain_config",
    "qwen3_32b_pretrain_config",
    "qwen3_600m_finetune_config",
    "qwen3_1p7b_finetune_config",
    "qwen3_4b_finetune_config",
    "qwen3_8b_finetune_config",
    "qwen3_14b_finetune_config",
    "qwen3_32b_finetune_config",
    # Qwen3 MoE models
    "qwen3_30b_a3b_pretrain_config",
    "qwen3_30b_a3b_finetune_config",
    "qwen3_235b_a22b_pretrain_config",
    "qwen3_235b_a22b_finetune_config",
    # Qwen3-Next models
    "qwen3_next_80b_a3b_pretrain_config",
    "qwen3_next_80b_a3b_finetune_config",
]
