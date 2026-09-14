#!/bin/bash


# NVRx Straggler Detection Example Launch Script
#
# Usage:
#   ./examples/resiliency/straggler_detection/run_straggler_detection.sh
#   ./examples/resiliency/straggler_detection/run_straggler_detection.sh --nproc 4
#   ./examples/resiliency/straggler_detection/run_straggler_detection.sh --train-iters 200

set -euo pipefail

# Default values
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
REPORT_INTERVAL="${REPORT_INTERVAL:-5.0}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --nproc)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --train-iters)
            TRAIN_ITERS="$2"
            shift 2
            ;;
        --report-interval)
            REPORT_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running NVRx Straggler Detection Example"
echo "  GPUs: ${NPROC_PER_NODE}"
echo "  Iterations: ${TRAIN_ITERS}"
echo "  Report Interval: ${REPORT_INTERVAL}s"
echo ""

uv run python -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    "${SCRIPT_DIR}/straggler_detection_example.py" \
    --train-iters "${TRAIN_ITERS}" \
    --report-interval "${REPORT_INTERVAL}"
