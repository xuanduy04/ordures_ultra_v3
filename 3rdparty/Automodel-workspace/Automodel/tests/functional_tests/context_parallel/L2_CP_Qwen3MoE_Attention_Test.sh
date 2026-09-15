#!/bin/bash


set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)
export CUDA_VISIBLE_DEVICES="0,1"

# Run Qwen3MoE attention layer CP test with 2 GPUs
torchrun --nproc_per_node=2 --nnodes=1 \
    tests/functional_tests/context_parallel/run_attention_cp.py \
    --model_type qwen3_moe
