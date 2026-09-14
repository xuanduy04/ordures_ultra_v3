

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

UNIT_TEST=false
CPU=false
TEST_DIR="tests/"
TEST_NAME=""
ADDITIONAL_ARGS=""

for i in "$@"; do
    case $i in
        --UNIT_TEST=?*) UNIT_TEST="${i#*=}";;
        --CPU=?*) CPU="${i#*=}";;
        --TEST_NAME=?*) TEST_NAME="${i#*=}";;
        *) ;;
    esac
    shift
done

if [[ "$CPU" == "false" ]]; then
    export CUDA_VISIBLE_DEVICES="0,1"
else
    export ADDITIONAL_ARGS="--cpu --with_downloads"
fi

if [[ "$UNIT_TEST" == "true" ]]; then
    export TEST_DIR="tests/unit_tests"
else
    export TEST_DIR="tests/functional_tests/$TEST_NAME"
fi

coverage run \
    --data-file=/workspace/.coverage \
    --source=/workspace/ \
    --parallel-mode \
    -m pytest \
    $TEST_DIR \
    -o log_cli=true \
    -o log_cli_level=INFO \
    -vs -m "not pleasefixme" --tb=short -rA \
    $ADDITIONAL_ARGS
coverage combine -q
