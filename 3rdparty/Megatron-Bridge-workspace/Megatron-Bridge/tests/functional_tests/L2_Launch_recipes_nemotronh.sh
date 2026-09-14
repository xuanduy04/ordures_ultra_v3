#!/bin/bash


set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export CUDA_VISIBLE_DEVICES="0,1"

# Run Nemotron H and Nemotron Nano v2 recipe functional tests on 2 GPUs
# This script tests recipe configurations with their default settings to ensure
# they can run basic training without crashes

# Test pretrain recipes (4B and 9B v2)
uv run python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 -m coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA tests/functional_tests/recipes/test_nemotronh_recipes_pretrain.py

# Test finetune recipes (9B v2 with LoRA and full SFT)
uv run python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 -m coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA tests/functional_tests/recipes/test_nemotronh_recipes_finetune.py

coverage combine -q

