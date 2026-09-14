

"""Parallelism presets for Qwen3 performance configs."""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_QWEN3_VL_235B_A22B_CONFIG = WorkloadBaseConfig(
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    global_batch_size=1024,
)


BASE_QWEN3_VL_30B_A3B_CONFIG = WorkloadBaseConfig(
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    global_batch_size=512,
)

BASE_QWEN3_NEXT_80B_A3B_CONFIG = WorkloadBaseConfig(
    expert_model_parallel_size=64,
    expert_tensor_parallel_size=1,
    global_batch_size=1024,
)

# Qwen3 235B A22B presets ----------------------------------------------------


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB300_BF16 = replace(
    BASE_QWEN3_VL_235B_A22B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=64,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_CS = replace(
    BASE_QWEN3_VL_235B_A22B_CONFIG,
    num_gpus=64,
    tensor_model_parallel_size=1,
    expert_model_parallel_size=64,
    micro_batch_size=1,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_MX = QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_CS


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB200_BF16 = replace(
    BASE_QWEN3_VL_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB200_FP8_CS = replace(
    BASE_QWEN3_VL_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    expert_model_parallel_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB200_FP8_MX = QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB200_FP8_CS


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_B200_BF16 = replace(
    BASE_QWEN3_VL_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    moe_a2a_overlap=True,
)


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_B200_FP8_CS = replace(
    BASE_QWEN3_VL_235B_A22B_CONFIG,
    num_gpus=64,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=8,
    moe_a2a_overlap=True,
)


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_B200_FP8_MX = QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_B200_FP8_CS


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_H100_BF16 = replace(
    BASE_QWEN3_VL_235B_A22B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=32,
    moe_a2a_overlap=True,
)


QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_H100_FP8_CS = replace(
    BASE_QWEN3_VL_235B_A22B_CONFIG,
    num_gpus=256,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,
    expert_model_parallel_size=32,
    moe_a2a_overlap=True,
)


# Qwen3 30B A3B presets ------------------------------------------------------


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB300_BF16 = replace(
    BASE_QWEN3_VL_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB300_FP8_CS = replace(
    BASE_QWEN3_VL_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=8,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB300_FP8_MX = QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB300_FP8_CS


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB200_BF16 = replace(
    BASE_QWEN3_VL_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_CS = replace(
    BASE_QWEN3_VL_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["attn", "moe_router", "moe_preprocess"],
)


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX = replace(
    BASE_QWEN3_VL_30B_A3B_CONFIG,
    num_gpus=8,
    micro_batch_size=4,
    moe_flex_dispatcher_backend="hybridep",
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_B200_BF16 = replace(
    BASE_QWEN3_VL_30B_A3B_CONFIG,
    num_gpus=8,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_B200_FP8_CS = replace(
    BASE_QWEN3_VL_30B_A3B_CONFIG,
    num_gpus=8,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_B200_FP8_MX = QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_B200_FP8_CS


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_H100_BF16 = replace(
    BASE_QWEN3_VL_30B_A3B_CONFIG,
    num_gpus=16,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=12,
    moe_a2a_overlap=True,
    cuda_graph_impl="transformer_engine",
    cuda_graph_scope=["moe_router", "moe_preprocess"],
)


QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_H100_FP8_CS = replace(
    BASE_QWEN3_VL_30B_A3B_CONFIG,
    num_gpus=16,
    pipeline_model_parallel_size=2,
    virtual_pipeline_model_parallel_size=12,
    moe_a2a_overlap=True,
)


__all__ = [
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB300_BF16",
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB200_BF16",
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_B200_BF16",
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_B200_FP8_CS",
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_B200_FP8_MX",
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_H100_BF16",
    "QWEN3_VL_235B_A22B_PRETRAIN_CONFIG_H100_FP8_CS",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB300_BF16",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB300_FP8_MX",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB200_BF16",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_B200_BF16",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_B200_FP8_CS",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_B200_FP8_MX",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_H100_BF16",
    "QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_H100_FP8_CS",
]
