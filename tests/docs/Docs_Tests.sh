

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)

cd ${PROJECT_ROOT}/docs
uv run --no-sync coverage run -a --data-file=${PROJECT_ROOT}/tests/.coverage --source=${PROJECT_ROOT}/nemo_rl -m sphinx.cmd.build -b doctest . _build/doctest
ls ${PROJECT_ROOT}/tests
