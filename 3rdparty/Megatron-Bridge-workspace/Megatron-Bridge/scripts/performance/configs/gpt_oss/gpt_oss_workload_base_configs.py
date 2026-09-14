

"""Parallelism presets for GPT performance configs.

Config naming convention:
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

V1: GBS=512
V2: GBS=1280

Use --config_variant to select a variant.
Use --list_config_variants to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_GPT_OSS_120B_CONFIG = WorkloadBaseConfig(
    num_gpus=64,
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    global_batch_size=512,
    micro_batch_size=1,
)


# =============================================================================
# GPT-OSS 120B Pretrain - V1 (GBS=512)
# =============================================================================

GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16_V1 = replace(
    BASE_GPT_OSS_120B_CONFIG,
    expert_model_parallel_size=64,
    micro_batch_size=4,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16_V1 = replace(
    BASE_GPT_OSS_120B_CONFIG,
    expert_model_parallel_size=64,
    micro_batch_size=4,
    recompute_modules=["layernorm", "moe_act"],
)


GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16_V1 = replace(
    BASE_GPT_OSS_120B_CONFIG,
    expert_model_parallel_size=64,
    micro_batch_size=4,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16_V1 = replace(
    BASE_GPT_OSS_120B_CONFIG,
    expert_model_parallel_size=64,
    micro_batch_size=4,
    recompute_modules=["layernorm", "moe_act"],
)


GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16_V1 = replace(
    BASE_GPT_OSS_120B_CONFIG,
    pipeline_model_parallel_size=4,
    recompute_modules=["layernorm", "moe_act"],
)


# =============================================================================
# GPT-OSS 120B Pretrain - V2 (GBS=1280)
# =============================================================================

GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16_V2 = replace(
    GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16_V1,
    global_batch_size=1280,
)


GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16_V2 = replace(
    GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16_V1,
    global_batch_size=1280,
)


GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16_V2 = replace(
    GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16_V1,
    global_batch_size=1280,
)


GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16_V2 = replace(
    GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16_V1,
    global_batch_size=1280,
)


GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16_V2 = replace(
    GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16_V1,
    global_batch_size=1280,
)


__all__ = [
    # V1 (GBS=512)
    "GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16_V1",
    "GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16_V1",
    "GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16_V1",
    "GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16_V1",
    # V2 (GBS=1280)
    "GPT_OSS_120B_PRETRAIN_CONFIG_GB300_BF16_V2",
    "GPT_OSS_120B_PRETRAIN_CONFIG_GB200_BF16_V2",
    "GPT_OSS_120B_PRETRAIN_CONFIG_B300_BF16_V2",
    "GPT_OSS_120B_PRETRAIN_CONFIG_B200_BF16_V2",
    "GPT_OSS_120B_PRETRAIN_CONFIG_H100_BF16_V2",
]
