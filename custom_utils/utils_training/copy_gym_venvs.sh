#!/bin/bash
# copy_gym_venvs.sh — Copy math_with_judge's .venv to every resource server that lacks one.
#
# Usage:
#   NEMO_GYM_VENV_DIR=/opt/gym_venvs ./copy_gym_venvs.sh /abs/path/to/config.yaml
#
# The script reads env.nemo_gym.config_paths from the YAML, skips non-resource-server
# entries, and copies math_with_judge's .venv to any target that does not already have
# bin/python.  Runs on a single node; fan it out to every node before launching training.

set -euo pipefail

CONFIG="$1"
VENV_ROOT="${NEMO_GYM_VENV_DIR:-/opt/gym_venvs}"
SOURCE="$VENV_ROOT/resources_servers/math_with_judge/.venv"

printf "Duplicating NeMo-Gym venv's for environments in '${CONFIG}'\n"

if [ ! -f "$SOURCE/bin/python" ]; then
    printf "FATAL: math_with_judge .venv not found at $SOURCE\n" >&2
    exit 1
fi

python3 -c "
from omegaconf import OmegaConf
from pathlib import Path

paths = OmegaConf.select(OmegaConf.load('${CONFIG}'), 'env.nemo_gym.config_paths') or []
for p in paths:
    if not p.startswith('resources_servers/'):
        continue
    server_dir = Path(p).parent.parent
    if server_dir.name == 'math_with_judge':
        continue
    print(server_dir)
" | while read server_dir; do
    target="$VENV_ROOT/$server_dir/.venv"
    if [ -f "$target/bin/python" ]; then
        printf "[skip] $server_dir  (.venv already present)\n"
        continue
    fi
    mkdir -p "$(dirname "$target")"
    cp -r "$SOURCE" "$target"
    printf "[copy] $SOURCE -> $target\n"
done

printf "\n"
