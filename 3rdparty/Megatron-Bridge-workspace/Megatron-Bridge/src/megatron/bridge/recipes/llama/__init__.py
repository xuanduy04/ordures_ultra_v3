

# Llama2 models
from .llama2 import (
    llama2_7b_pretrain_config,
)

# Llama3 models
from .llama3 import (
    llama3_8b_16k_pretrain_config,
    llama3_8b_64k_pretrain_config,
    llama3_8b_128k_pretrain_config,
    # Llama3 finetune models
    llama3_8b_finetune_config,
    llama3_8b_low_precision_pretrain_config,
    llama3_8b_pretrain_config,
    llama3_70b_16k_pretrain_config,
    llama3_70b_64k_pretrain_config,
    llama3_70b_finetune_config,
    llama3_70b_pretrain_config,
    # Llama3.1 finetune models
    llama31_8b_finetune_config,
    # Llama3.1 models
    llama31_8b_pretrain_config,
    llama31_70b_finetune_config,
    llama31_70b_pretrain_config,
    llama31_405b_finetune_config,
    llama31_405b_pretrain_config,
    # Llama3.2 finetune models
    llama32_1b_finetune_config,
    # Llama3.2 models
    llama32_1b_pretrain_config,
    llama32_3b_finetune_config,
    llama32_3b_pretrain_config,
)


__all__ = [
    # Llama2 models
    "llama2_7b_pretrain_config",
    # Llama3 models
    "llama3_8b_pretrain_config",
    "llama3_8b_16k_pretrain_config",
    "llama3_8b_64k_pretrain_config",
    "llama3_8b_128k_pretrain_config",
    "llama3_8b_low_precision_pretrain_config",
    "llama3_70b_pretrain_config",
    "llama3_70b_16k_pretrain_config",
    "llama3_70b_64k_pretrain_config",
    # Llama3.1 models
    "llama31_8b_pretrain_config",
    "llama31_70b_pretrain_config",
    "llama31_405b_pretrain_config",
    # Llama3.2 models
    "llama32_1b_pretrain_config",
    "llama32_3b_pretrain_config",
    # Llama3 finetune models
    "llama3_8b_finetune_config",
    "llama3_70b_finetune_config",
    # Llama3.1 finetune models
    "llama31_8b_finetune_config",
    "llama31_70b_finetune_config",
    "llama31_405b_finetune_config",
    # Llama3.2 finetune models
    "llama32_1b_finetune_config",
    "llama32_3b_finetune_config",
]
