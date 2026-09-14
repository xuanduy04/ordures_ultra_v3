#!/usr/bin/env bash


# Workspace directory for checkpoints and results
WORKSPACE=${WORKSPACE:-/workspace}

# GLM-4.5V is a large MoE model (106B parameters)
# Using TP=1, PP=2, EP=4 for inference (8 GPUs minimum)

# Inference with Hugging Face checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path zai-org/GLM-4.5V \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp 1 \
    --pp 2 \
    --ep 4 \
    --trust_remote_code

# Inference with imported Megatron checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path zai-org/GLM-4.5V \
    --megatron_model_path ${WORKSPACE}/models/GLM-4.5V/iter_0000000 \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp 1 \
    --pp 2 \
    --ep 4 \
    --trust_remote_code

# Inference with exported HF checkpoints
uv run python -m torch.distributed.run --nproc_per_node=8 examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path ${WORKSPACE}/models/GLM-4.5V-hf-export \
    --image_path "https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png" \
    --prompt "Describe this image." \
    --max_new_tokens 50 \
    --tp 1 \
    --pp 2 \
    --ep 4 \
    --trust_remote_code
