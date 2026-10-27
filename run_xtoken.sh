#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Edit these two paths if the deployed driver or run configuration changes.
PYTHON="${NRL_PYTHON:-/opt/nemo_rl_venv/bin/python}"
CONFIG="$SOURCE/tools/xtoken_all/pilot_200.yaml"
ARM="${1:-cross}"
if [ "$#" -gt 0 ]; then shift; fi
test -x "$PYTHON"
source "$SOURCE/custom_utils/utils_training/use_custom_opt.sh" "$SOURCE" "${CUSTOM_NLTK_DATA_DIR:-/workspace/storage-shared/nlp/vai-1.4-30b/nemo-rl/custom_opt_ultra_selfbuilt_nano/nltk_data}"
export PYTHONPATH="$SOURCE/tools/xtoken_all:$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 NRL_IGNORE_VERSION_MISMATCH=1
export NRL_VLLM_ASYNC_TIMEOUT_SECONDS=36000 TORCH_NCCL_BLOCKING_WAIT=1 NCCL_TIMEOUT=36000
exec "$PYTHON" -B "$SOURCE/tools/xtoken_all/prepare_pilot.py" --config "$CONFIG" --arm "$ARM" --run "$@"
