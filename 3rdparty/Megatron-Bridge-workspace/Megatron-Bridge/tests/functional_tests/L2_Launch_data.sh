

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

CUDA_VISIBLE_DEVICES="0,1" uv run coverage run -a --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ -m pytest \
    -o log_cli=true \
    -o log_cli_level=INFO \
    --disable-warnings \
    -vs tests/functional_tests/data -m "not pleasefixme"
