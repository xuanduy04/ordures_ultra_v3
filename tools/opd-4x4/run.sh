#!/usr/bin/env bash
set -euo pipefail
umask 077

fail() { printf '%s\n' "$*" >&2; exit 1; }

mode=${1:-train}
case "$mode" in
  train|rerun|status|config) ;;
  -h|--help)
    printf '%s\n' 'Usage: ./run.sh [train|rerun|status|config]' \
      'train resumes the latest manual run, or the deployed run on first use.' \
      'rerun starts a separate run from the verified step_5.' \
      'config prints the merged input YAML for train. status shows the selected checkpoint.'
    exit 0
    ;;
  *) fail "Unknown command: $mode (see ./run.sh --help)" ;;
esac
(( $# <= 1 )) || fail "Expected a single command"

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profile="$bundle/opd-4x4.yaml"
root=/workspace/storage-shared/nlp/users/tungks/xtoken
runs="$root/runs"
code="$root/deploy-6753e8d-d2732d3"
py=/opt/nemo_rl_venv/bin/python
resume_run="$runs/bridge-20261120-train-4x4-d2732d3-c66e2597"
seed_run="$runs/bridge-20261120-alignment-4x4-92a5086-dd308089"
manual="$runs/manual-4x4-d2732d3"
expected_host=rl-opd-dev-8gpu-2-0-0

host=$(hostname -s)
[[ "$host" == "$expected_host" ]] || fail "Expected $expected_host, found $host"
if [[ "$mode" == train || "$mode" == rerun ]]; then
  exec 9>"$manual.lock"
  flock -n 9 || fail "A manual OPD launch is already active"
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -n "$gpu_pids" ]] ||
     pgrep -f '^/opt/nemo_rl_venv/bin/python .*examples\.nemo_gym\.run_grpo_nemo_gym' >/dev/null; then
    fail "A job or GPU process is active on $host. Run ./run.sh status."
  fi
fi

latest_checkpoint() {
  [[ -d "$1/checkpoints" ]] || return 0
  find "$1/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
    awk '/^step_[0-9]+$/' | sort -V | tail -1
}

source_run=$resume_run
if [[ "$mode" == rerun ]]; then
  source_run=$seed_run
  step_dir=step_5
else
  # Use the most recently modified manual run with a finalized checkpoint.
  while IFS= read -r candidate; do
    step_dir=$(latest_checkpoint "$candidate")
    if [[ -n "$step_dir" ]]; then
      source_run=$candidate
      break
    fi
  done < <(find "$runs" -mindepth 1 -maxdepth 1 -type d -name "${manual##*/}-*" \
    -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2-)
  step_dir=$(latest_checkpoint "$source_run")
fi
[[ -n "$step_dir" ]] || fail "No finalized checkpoint in $source_run/checkpoints"
step=${step_dir#step_}
checkpoint="$source_run/checkpoints/$step_dir"

if [[ "$mode" == status ]]; then
  printf 'RUN_DIR=%s\nCHECKPOINT=%s\n' "$source_run" "$checkpoint"
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
  if [[ -s "$source_run/driver.log" ]]; then
    grep -aE 'Loss:|Total step time:|OutOfMemoryError|Max number of steps' "$source_run/driver.log" | tail -16 || true
  fi
  exit 0
fi

[[ -s "$profile" && -x "$py" && -s "$code/nemo_rl/algorithms/x_token/compiled_alignment.py" ]] ||
  fail "Missing YAML, Python runtime or deployed d2732d3 source"
[[ -s "$checkpoint/config.yaml" ]] || fail "Missing checkpoint config: $checkpoint/config.yaml"

gym=/workspace/storage-shared/nlp/users/tungks/xtoken_all/ordures_ultra_v3_gym
megatron=/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM
export PYTHONPATH="$code:$gym:$megatron${PYTHONPATH:+:$PYTHONPATH}"
export NEMO_RL_PY_EXECUTABLES_SYSTEM=0 NEMO_RL_VENV_DIR=/opt/ray_venvs
export NEMO_GYM_VENV_DIR=/opt/gym_venvs NRL_FORCE_REBUILD_VENVS=false
export UV_OFFLINE=1 HF_HUB_OFFLINE=1 RAY_ADDRESS=local PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/tmp" RAY_TMPDIR=/tmp
export NRL_MEGATRON_REUSE_OPTIMIZER_LOAD_STORAGE=1
export NRL_MEGATRON_CHECKPOINT_DIR="$runs/bridge-20261119-f24-447820b9/OPD-Nemotron65-Flash-8gpu-77xdzi8q/hf-to-mcore"
cd "$code"

render_config() {
  "$py" -B - "$checkpoint/config.yaml" "$profile" "$1" "${2:-}" <<'PY'
import sys
from pathlib import Path
from urllib.parse import urlsplit

from omegaconf import OmegaConf
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

base_path, profile_path, action, run_dir = sys.argv[1:]
register_omegaconf_resolvers()
config = OmegaConf.merge(load_config(base_path), load_config(profile_path))
if action == "info":
    print(config.grpo.max_num_steps, config.on_policy_distillation.xtoken.alignment_backend)
elif action == "show":
    sensitive = {"api_key", "access_token", "token", "password", "authorization", "secret", "client_secret"}

    def redact(value):
        if isinstance(value, dict):
            return {
                key: "<redacted>" if str(key).lower() in sensitive and item else redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    print(OmegaConf.to_yaml(OmegaConf.create(redact(OmegaConf.to_container(config)))), end="")
else:
    config.checkpointing.checkpoint_dir = str(Path(run_dir) / "checkpoints")
    config.logger.log_dir = str(Path(run_dir) / "logs")
    config.env.nemo_gym.nemo_gym_log_dir = str(Path(run_dir) / "gym-logs")
    OmegaConf.save(config, Path(run_dir) / "effective.yaml")
    teachers = config.on_policy_distillation.teacher_model_by_agent_name
    hosts = sorted({urlsplit(teacher.url).hostname for teacher in teachers.values()} - {None})
    print(",".join(hosts))
PY
}

if [[ "$mode" == config ]]; then
  printf 'Base checkpoint: %s\n' "$checkpoint" >&2
  render_config show
  exit 0
fi

info=$(render_config info)
read -r max_steps backend <<< "$info"
[[ "$max_steps" =~ ^[0-9]+$ ]] || fail "Invalid grpo.max_num_steps in $profile"
[[ "$backend" == python || "$backend" == numba ]] || fail "Invalid alignment backend in $profile"
(( step < max_steps )) || fail "Checkpoint is at step $step (target $max_steps). Use ./run.sh rerun to restart from step_5."

for path in training_info.json train_dataloader.pt replay_buffer.pt policy/weights/iter_0000000/.metadata; do
  [[ -s "$checkpoint/$path" ]] || fail "Incomplete checkpoint: $checkpoint/$path"
done
jq -e --argjson step "$step" '.current_step == $step' "$checkpoint/training_info.json" >/dev/null

run=$(mktemp -d "$manual-XXXXXXXX")
mkdir -p "$run/checkpoints" "$run/numba-cache"
cp -al -- "$checkpoint" "$run/checkpoints/$step_dir"
cp -- "$profile" "$run/opd-4x4.yaml"
profile="$run/opd-4x4.yaml"
teacher_hosts=$(render_config write "$run")
export NUMBA_CACHE_DIR="$run/numba-cache"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}${teacher_hosts:+$teacher_hosts,}localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
printf 'RESUME_FROM=%s\nRUN_DIR=%s\nCONFIG=%s\n' "$checkpoint" "$run" "$run/effective.yaml"

if [[ "$backend" == numba ]]; then
  env CUDA_VISIBLE_DEVICES='' "$py" -B -c 'import numba'
  env CUDA_VISIBLE_DEVICES='' "$py" -B \
    "$code/tests/unit/algorithms/x_token/test_compiled_alignment.py" \
    TestCompiledAlignment.test_dp_scores_and_traceback
fi

"$py" -B -u -m examples.nemo_gym.run_grpo_nemo_gym \
  --config "$run/effective.yaml" 2>&1 | tee "$run/driver.log"
