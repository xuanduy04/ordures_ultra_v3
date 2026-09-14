#!/bin/bash


set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export CUDA_VISIBLE_DEVICES="0,1"

# Run recipe functional tests on 2 GPUs
# This script tests recipe configurations with their default settings to ensure
# they can run basic training without crashes
uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA tests/functional_tests/recipes/test_llama_recipes_pretrain_3b.py
coverage combine -q
