#!/usr/bin/env bash
set -euo pipefail
ARM="${1:?Specify same, same235 or cross}"
ACTION="${2:?Specify render, check, launch or assess}"
: "${XTOKEN_ALL_ROOT:?Set persistent xtoken_all root}"
: "${NRL_PYTHON:?Use the existing driver interpreter}"
: "${NEMO_RL_VENV_DIR:?Use the restored worker venv store}"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SOURCE_ROOT"
export PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 HF_HUB_OFFLINE=1
case "$ARM" in same|same235|cross) ;; *) echo 'Invalid arm' >&2; exit 2 ;; esac
RUN_ID="$("$NRL_PYTHON" -B -c 'import sys,yaml; print(yaml.safe_load(open("tools/xtoken_all/pilot_200.yaml"))["arms"][sys.argv[1]]["run_id"])' "$ARM")"
export XTOKEN_RUN_DIR="$XTOKEN_ALL_ROOT/runs/$RUN_ID"
case "$ACTION" in
  render)
    : "${XTOKEN_RAY_ADDRESS:?Set the local verified Ray head host:port}"
    "$NRL_PYTHON" -B tools/xtoken_all/prepare_pilot.py --arm "$ARM" --ray-address "$XTOKEN_RAY_ADDRESS"
    ;;
  check)
    "$NRL_PYTHON" -B tools/xtoken_all/preflight.py --run-dir "$XTOKEN_RUN_DIR"
    evidence=()
    if [ -n "${XTOKEN_TEACHER_MODE_EVIDENCE:-}" ]; then
      evidence=(--server-mode-evidence "$XTOKEN_TEACHER_MODE_EVIDENCE")
    fi
    "$NRL_PYTHON" -B tools/xtoken_all/probe_teacher.py --config "$XTOKEN_RUN_DIR/effective.yaml" "${evidence[@]}"
    ;;
  launch)
    : "${XTOKEN_ALLOCATION_RECEIPT:?Capture a fresh owned four-GPU allocation first}"
    "$NRL_PYTHON" -B tools/xtoken_all/ray_stream_probe.py --run-dir "$XTOKEN_RUN_DIR" --allocation-receipt "$XTOKEN_ALLOCATION_RECEIPT"
    bash tools/xtoken_all/launch.sh
    ;;
  assess)
    "$NRL_PYTHON" -B tools/xtoken_all/check_pilot.py --run-dir "$XTOKEN_RUN_DIR"
    ;;
  *) echo 'Invalid action' >&2; exit 2 ;;
esac
