#!/bin/bash

set -euo pipefail

GIT_VERSION=$(git version | awk '{print $3}')
GIT_MAJOR=$(echo $GIT_VERSION | awk -F. '{print $1}')
GIT_MINOR=$(echo $GIT_VERSION | awk -F. '{print $2}')

if [[ $GIT_MAJOR -eq 2 && $GIT_MINOR -lt 31 ]]; then
    echo "Git version must be at least 2.31.0. Found $GIT_VERSION"
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PACKAGE_ROOT=$(realpath $SCRIPT_DIR/../nemo_rl)

ruff check $PACKAGE_ROOT --fix
ruff format $PACKAGE_ROOT