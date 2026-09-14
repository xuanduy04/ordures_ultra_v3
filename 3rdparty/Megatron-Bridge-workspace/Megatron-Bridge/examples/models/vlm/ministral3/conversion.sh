#!/usr/bin/env bash


# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Import HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
    --megatron-path ${WORKSPACE}/models/Ministral-3-3B-Instruct-2512-BF16

# Export Megatron → HF
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model mistralai/Ministral-3-3B-Instruct-2512-BF16 \
    --megatron-path ${WORKSPACE}/models/Ministral-3-3B-Instruct-2512-BF16/iter_0000000 \
    --hf-path ${WORKSPACE}/models/Ministral-3-3B-Instruct-2512-BF16-hf-export

# Round-trip validation
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id mistralai/Ministral-3-3B-Instruct-2512-BF16 --tp 2 --pp 2
