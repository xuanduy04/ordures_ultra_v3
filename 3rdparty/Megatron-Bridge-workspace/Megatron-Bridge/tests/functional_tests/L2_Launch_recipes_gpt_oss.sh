#!/bin/bash


set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export CUDA_VISIBLE_DEVICES="0,1"

# Run GPT-OSS recipe functional tests on 2 GPUs
# This script tests recipe configurations with their default settings to ensure
# they can run basic training without crashes

# Test pretrain recipes (20B)
uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA tests/functional_tests/recipes/test_gpt_oss_recipes_pretrain.py

uv run python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 -m coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA tests/functional_tests/recipes/test_gpt_oss_recipes_finetune.py
coverage combine -q


