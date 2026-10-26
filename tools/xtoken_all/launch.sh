#!/usr/bin/env bash
# The operator closes allocation, source, assets, teacher and Gym gates first.
set -euo pipefail
: "${XTOKEN_ALL_ROOT:?Set the persistent xtoken_all root}"
: "${XTOKEN_RUN_DIR:?Set a freshly rendered run directory}"
: "${NRL_PYTHON:?Set the existing driver interpreter}"
: "${NEMO_RL_VENV_DIR:?Set the existing policy/generation worker venv store}"
: "${XTOKEN_ALLOCATION_RECEIPT:?Set the fresh allocation/Ray ownership receipt}"
export PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PYTHONPATH"
"$NRL_PYTHON" - "$XTOKEN_ALL_ROOT" "$XTOKEN_RUN_DIR" "$XTOKEN_ALLOCATION_RECEIPT" <<'PY'
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tools/xtoken_all'))
from integrity import sha256, source_snapshot
root, run, allocation = map(Path, sys.argv[1:])
run.resolve().relative_to(root.resolve())
assert not (run / 'driver.log').exists(), 'Refuse to reuse a run'
meta = json.loads((run / 'provenance.json').read_text())
assert meta['rollout_backend'] == 'nemo_gym'
assert hashlib.sha256((run / 'effective.yaml').read_bytes()).hexdigest() == meta['effective_config_sha256']
assert source_snapshot(Path.cwd())['sha256'] == meta['source_sha256'], 'Source changed since render'
from omegaconf import OmegaConf
cfg = OmegaConf.to_container(OmegaConf.load(run / 'effective.yaml'), resolve=True)
assert sha256(cfg['data']['train']['data_path']) == meta['data_sha256']
assert sha256(cfg['teachers'][0]['projection_matrix_path']) == meta['projection_sha256']
for name in ('teacher-probe', 'assets-check', 'source-check', 'gym-preflight'):
    receipt = json.loads((run / (name + '.json')).read_text())
    assert receipt['success'] and receipt['effective_config_sha256'] == meta['effective_config_sha256'], name
    if name == 'teacher-probe':
        assert receipt['server_logprobs_mode_verified'] and receipt['top_k'] == 20
lease = json.loads(allocation.read_text())
assert lease['success'] and lease['gpu_indices'] == [0, 1, 2, 3] and lease['ray_ownership_verified']
assert 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(lease['checked_at_utc'])).total_seconds() < 300
raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
devices = {int(line.split(',')[0]): line.split(',')[1].strip() for line in raw.splitlines()}
assert [devices[i] for i in range(4)] == lease['gpu_uuids']
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
assert not any(line.split(',')[0].strip() in lease['gpu_uuids'] for line in apps.splitlines()), 'Target GPUs are occupied'
import ray
ray.init(address=lease['ray_address'], log_to_driver=False)
try:
    nodes = [node['NodeID'] for node in ray.nodes() if node['Alive']]
    assert sorted(nodes) == sorted(lease['ray_node_ids']) and ray.cluster_resources()['GPU'] == 4
finally:
    ray.shutdown()
for worker in ('nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2',
               'nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker'):
    assert (Path(os.environ['NEMO_RL_VENV_DIR']) / worker / 'bin/python').is_file(), worker
PY
set +e
"$NRL_PYTHON" -u -m examples.run_xtoken_on_policy_distillation \
    --config "$XTOKEN_RUN_DIR/effective.yaml" 2>&1 | tee "$XTOKEN_RUN_DIR/driver.log"
statuses=("${PIPESTATUS[@]}")
set -e
printf '{"driver_exit_code":%s,"tee_exit_code":%s}\n' "${statuses[0]}" "${statuses[1]}" > "$XTOKEN_RUN_DIR/process-exit.json"
test "${statuses[0]}" -eq 0 && test "${statuses[1]}" -eq 0
