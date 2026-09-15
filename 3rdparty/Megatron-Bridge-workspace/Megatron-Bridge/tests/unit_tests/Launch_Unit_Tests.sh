

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

echo "=================================================="
echo "🧪 UNIT TESTS"
echo "=================================================="

# Display MCore commit SHA if triggered from MCore CI
if [ -f "/opt/Megatron-Bridge/.mcore_commit_sha" ]; then
    echo "📦 MCore commit: $(cat /opt/Megatron-Bridge/.mcore_commit_sha)"
fi
echo ""

# Skip timeout on Azure runners because the machines are slower
TIMEOUT_ARG="--timeout=2"
if [[ "${GHA_RUNNER:-}" == *"azure"* ]]; then
    TIMEOUT_ARG=""
fi

CUDA_VISIBLE_DEVICES="0,1" uv run coverage run -a --data-file=/opt/Megatron-Bridge/.coverage --source=/opt/Megatron-Bridge/ -m pytest \
    $TIMEOUT_ARG \
    -o log_cli=true \
    -o log_cli_level=INFO \
    --disable-warnings \
    -vs tests/unit_tests -m "not pleasefixme"
