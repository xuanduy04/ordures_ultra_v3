#!/usr/bin/env bash
# use_custom_opt.sh - Override installed NeMo RL packages with custom
# implementations by prepending their project directories to PYTHONPATH.
#
# Usage:
#   source custom_utils/use_custom_opt.sh [CUSTOM_NEMO_RL_DIR] [CUSTOM_NLTK_DATA_DIR]
#
#   CUSTOM_NEMO_RL_DIR: path to the custom nemo-rl tree
#                       (default: ${USE_CUSTOM_OPT_SCRIPT_DIR}/nemo-rl).
#   CUSTOM_NLTK_DATA_DIR: path to the custom nltk data
#                         (default: ${USE_CUSTOM_OPT_SCRIPT_DIR}/nltk_data).
#
# Behavior:
#   Walk the custom tree (max depth 100) for every directory that contains a
#   pyproject.toml, setup.py, or setup.cfg.  Each such directory is treated as a
#   Python project root and prepended to PYTHONPATH so custom code takes
#   priority over installed packages.
#
#   "Megatron-LM" and "Megatron-Bridge" directories are skipped because
#   they override the pre-existing huggingface libraries, causing import issues,
#   "vllm" directories in "3rdparty" are skipped as we WANT to use the pre-existing one.
#
#   Also exports the env variable CUSTOM_NLTK_DATA_DIR

set -euo pipefail

USE_CUSTOM_OPT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_NEMO_RL_DIR="${1:-${USE_CUSTOM_OPT_SCRIPT_DIR}/nemo-rl}"

if [[ ! -d "$CUSTOM_NEMO_RL_DIR" ]]; then
    printf "ERROR: Custom NeMo RL directory not found: ${CUSTOM_NEMO_RL_DIR}\n" >&2
    return 1 2>/dev/null || exit 1
fi

printf "Adding custom NeMo RL overrides from '${CUSTOM_NEMO_RL_DIR}'\n"

python_paths=()

# Walk the tree for Python project roots, skipping nested 3rdparty dirs.
while IFS= read -r -d '' proj_file; do
    proj_dir="$(dirname "$proj_file")"
    # We don't skip nested "3rdparty" path-segments as they have been cleaned in v0.7.0, duplicates are intentional (and exclusive)

    # Skip anything under a "Megatron-LM" or "Megatron-Bridge" directory - importing it causes conflict with the prebuilt packages.
    if [[ "$proj_dir" == *"/Megatron-LM/"* || "$proj_dir" == *"/Megatron-LM" ]]; then
        printf "  [SKIP] (contains /Megatron-LM) $proj_dir\n"
        continue
    fi
    if [[ "$proj_dir" == *"/Megatron-Bridge/"* || "$proj_dir" == *"/Megatron-Bridge" ]]; then
        printf "  [SKIP] (contains /Megatron-Bridge) $proj_dir\n"
        continue
    fi

    # Skip the vllm directory - use the prebuilt one.
    if [[ "$proj_dir" == *"/3rdparty/"* ]] && ([[ "$proj_dir" == *"/vllm/"* || "$proj_dir" == *"/vllm" ]]); then
        printf "  [SKIP] (contains 3rdparty/*vllm) $proj_dir\n"
        continue
    fi

    python_paths+=("$proj_dir")
    printf "  [add] $proj_dir\n"
done < <(find "$CUSTOM_NEMO_RL_DIR" -maxdepth 100 \
    \( -name pyproject.toml -o -name setup.py -o -name setup.cfg \) \
    -print0 2>/dev/null)

if [[ ${#python_paths[@]} -eq 0 ]]; then
    printf "WARNING: No Python packages or projects found under ${CUSTOM_NEMO_RL_DIR}\n" >&2
    return 0 2>/dev/null || exit 0
fi

# Also add CUSTOM_NEMO_RL_DIR to PYTHONPATH
unique_paths=("$CUSTOM_NEMO_RL_DIR" "${python_paths[@]}")
# Deduplicate (a dir may have both pyproject.toml and setup.py)
unique_paths=($(printf '%s\n' "${unique_paths[@]}" | awk '!seen[$0]++'))

export PYTHONPATH="$(IFS=:; echo "${unique_paths[*]}")${PYTHONPATH:+:$PYTHONPATH}"

printf "PYTHONPATH updated (${#unique_paths[@]} directories added)\n"


# Add the missing nltk package.
CUSTOM_NLTK_DATA_DIR="${2:-${USE_CUSTOM_OPT_SCRIPT_DIR}/nltk_data}"

# Check if the folder exists and is not empty
PUNKT_TAB_DIR="${CUSTOM_NLTK_DATA_DIR}/tokenizers/punkt_tab"
if [ -d "$PUNKT_TAB_DIR" ] && [ -n "$(ls -A "$PUNKT_TAB_DIR" 2>/dev/null)" ]; then
    printf "Found existing punkt_tab data in %s\n" "$PUNKT_TAB_DIR"
else
    printf "WARNING: punkt_tab folder does not exist or is empty.\n" >&2
fi

export CUSTOM_NLTK_DATA_DIR
printf "CUSTOM_NLTK_DATA_DIR=${CUSTOM_NLTK_DATA_DIR}\n"

printf "\n"