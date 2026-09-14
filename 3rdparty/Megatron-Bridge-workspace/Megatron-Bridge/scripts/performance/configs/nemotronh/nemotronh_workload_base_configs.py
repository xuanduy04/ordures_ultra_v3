

"""Parallelism presets for Nemotron performance configs.

Config naming convention:
    {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}

Use --config_variant to select a variant.
Use --list_config_variants to see available variants interactively.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_NEMOTRONH_56B_CONFIG = WorkloadBaseConfig(
    num_gpus=64,
    global_batch_size=192,
    cuda_graph_impl="transformer_engine",
)


# =============================================================================
# NemotronH 56B Pretrain - V1
# =============================================================================

NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS_V1 = replace(
    BASE_NEMOTRONH_56B_CONFIG,
    tensor_model_parallel_size=2,
    cuda_graph_scope=["mamba", "attn"],
)


NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS_V1 = replace(
    BASE_NEMOTRONH_56B_CONFIG,
    tensor_model_parallel_size=2,
    cuda_graph_scope=["mamba", "attn"],
)


NEMOTRONH_56B_PRETRAIN_CONFIG_B300_FP8_CS_V1 = replace(
    BASE_NEMOTRONH_56B_CONFIG,
    tensor_model_parallel_size=2,
    cuda_graph_scope=["mamba", "attn"],
)


NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS_V1 = replace(
    BASE_NEMOTRONH_56B_CONFIG,
    tensor_model_parallel_size=2,
    cuda_graph_scope=["mamba", "attn"],
)


NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS_V1 = replace(
    BASE_NEMOTRONH_56B_CONFIG,
    tensor_model_parallel_size=8,
    cuda_graph_scope=["mamba"],
)


__all__ = [
    "NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS_V1",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS_V1",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_B300_FP8_CS_V1",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS_V1",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS_V1",
]
