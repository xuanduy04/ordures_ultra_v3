#!/usr/bin/env bash
# use_custom_opt.sh — Override installed NeMo RL packages with custom
# implementations by prepending their project directories to PYTHONPATH.
#
# Source this file in your shell before running NeMo RL to use custom opt code.
#
# Usage:
#   source custom_utils/use_custom_opt.sh [CUSTOM_NEMO_RL_DIR]
#
#   CUSTOM_NEMO_RL_DIR: path to the custom nemo-rl tree
#                       (default: ${OPT_SCRIPT_DIR}/nemo-rl).
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
#   Also adds the missing nltk package (assumed to be at ${OPT_SCRIPT_DIR}/nltk).

set -euo pipefail

OPT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUSTOM_NEMO_RL_DIR="${1:-${OPT_SCRIPT_DIR}/nemo-rl}"

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

    # Skip anything under a "Megatron-LM" or "Megatron-Bridge" directory - importing it causes conflict with the inbuilt packages.
    if [[ "$proj_dir" == *"/Megatron-LM/"* || "$proj_dir" == *"/Megatron-LM" ]]; then
        printf "  [SKIPPED] $proj_dir (contains /Megatron-LM)\n"
        continue
    fi
    if [[ "$proj_dir" == *"/Megatron-Bridge/"* || "$proj_dir" == *"/Megatron-Bridge" ]]; then
        printf "  [SKIPPED] $proj_dir (contains /Megatron-Bridge)\n"
        continue
    fi

    # Skip the vllm directory - use the inbuilt one.
    if [[ "$proj_dir" == *"/3rdparty/"* ]] && ([[ "$proj_dir" == *"/vllm/"* || "$proj_dir" == *"/vllm" ]]); then
        printf "  [SKIPPED] $proj_dir (contains 3rdparty/*vllm)\n"
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

# Also add CUSTOM_NEMO_RL_DIR
unique_paths=("$CUSTOM_NEMO_RL_DIR" "${python_paths[@]}")
# Deduplicate (a dir may have both pyproject.toml and setup.py)
unique_paths=($(printf '%s\n' "${unique_paths[@]}" | awk '!seen[$0]++'))

export PYTHONPATH="$(IFS=:; echo "${unique_paths[*]}")${PYTHONPATH:+:$PYTHONPATH}"

printf "PYTHONPATH updated (${#unique_paths[@]} directories added)\n"


# Add the missing nltk package.
export CUSTOM_NLTK_DATA_DIR="${OPT_SCRIPT_DIR}/nltk_data"
printf "CUSTOM_NLTK_DATA_DIR=${CUSTOM_NLTK_DATA_DIR}\n"

printf "\n"
