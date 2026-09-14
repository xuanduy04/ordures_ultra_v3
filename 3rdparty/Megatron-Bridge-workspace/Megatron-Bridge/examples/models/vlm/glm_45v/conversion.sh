#!/usr/bin/env bash


# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# Import HF → Megatron
uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model zai-org/GLM-4.5V \
    --megatron-path ${WORKSPACE}/models/GLM-4.5V

# Export Megatron → HF
uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model zai-org/GLM-4.5V \
    --megatron-path ${WORKSPACE}/models/GLM-4.5V/iter_0000000 \
    --hf-path ${WORKSPACE}/models/GLM-4.5V-hf-export

# Round-trip validation
# Note: GLM-4.5V is a large MoE model, adjust parallelism as needed
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
      --hf-model-id zai-org/GLM-4.5V --tp 1 --pp 2 --ep 4 --trust-remote-code
