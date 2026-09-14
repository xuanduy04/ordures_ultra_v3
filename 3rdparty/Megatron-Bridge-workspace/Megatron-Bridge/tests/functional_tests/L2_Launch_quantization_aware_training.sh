

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

# Ensure required packages are installed
pip install -q datasets

export CUDA_VISIBLE_DEVICES="0,1"

uv run coverage run --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ --parallel-mode -m pytest \
  -o log_cli=true -o log_cli_level=INFO -v -s -x -m "not pleasefixme" --tb=short -rA \
  tests/functional_tests/quantization/test_qat_workflow.py
coverage combine -q


