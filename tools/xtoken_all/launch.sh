#!/usr/bin/env bash
# The operator closes allocation, source, assets, teacher and Gym gates first.
set -euo pipefail
: "${XTOKEN_ALL_ROOT:?Set the persistent xtoken_all root}"
: "${XTOKEN_RUN_DIR:?Set a freshly rendered run directory}"
: "${NRL_PYTHON:?Set the existing driver interpreter}"
: "${NEMO_RL_VENV_DIR:?Set the existing policy/generation worker venv store}"
: "${XTOKEN_ALLOCATION_RECEIPT:?Set the fresh allocation/Ray ownership receipt}"
export PYTHONDONTWRITEBYTECODE=1 UV_OFFLINE=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$("$NRL_PYTHON" -B -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["gpu_indices"])))' "$XTOKEN_ALLOCATION_RECEIPT")"
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "$SOURCE_ROOT"
"$NRL_PYTHON" - "$XTOKEN_ALL_ROOT" "$XTOKEN_RUN_DIR" "$XTOKEN_ALLOCATION_RECEIPT" <<'PY'
import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tools/xtoken_all'))
from integrity import sha256, source_snapshot
from teacher_probe_contract import validate_teacher_probe
from tokenizer_contract import projection_digest
root, run, allocation = map(Path, sys.argv[1:])
run.resolve().relative_to(root.resolve())
assert not (run / 'driver.log').exists(), 'Refuse to reuse a run'
meta = json.loads((run / 'provenance.json').read_text())
assert meta['rollout_backend'] == 'nemo_gym'
assert hashlib.sha256((run / 'effective.yaml').read_bytes()).hexdigest() == meta['effective_config_sha256']
assert source_snapshot(Path.cwd())['sha256'] == meta['source_sha256'], 'Source changed since render'
from omegaconf import OmegaConf
cfg = OmegaConf.to_container(OmegaConf.load(run / 'effective.yaml'), resolve=True)
from nemo_rl.algorithms.x_token.gym_compat import (
    reject_inherited_gym_config, verify_gym_source, gym_actor_runtime_env, validate_gym_actor_report, validate_teacher_endpoint,
)
validate_teacher_endpoint(cfg)
reject_inherited_gym_config(os.environ)
actor_runtime = gym_actor_runtime_env(cfg['on_policy']['nemo_gym'], Path.cwd(), os.environ)
verify_gym_source(cfg['on_policy']['nemo_gym']['source_root'], cfg['on_policy']['nemo_gym']['source_contract'])
assert sha256(cfg['data']['train']['data_path']) == meta['data_sha256']
assert projection_digest(cfg['teachers'][0]['projection_matrix_path'], sha256) == meta['projection_sha256']
for name in ('teacher-probe', 'assets-check', 'source-check', 'gym-preflight', 'ray-stream-check'):
    receipt = json.loads((run / (name + '.json')).read_text())
    assert receipt['success'] and receipt['effective_config_sha256'] == meta['effective_config_sha256'], name
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(receipt['checked_at_utc'])).total_seconds()
    assert 0 <= age < cfg['on_policy']['nemo_gym']['receipt_max_age_s'], 'Expired receipt: ' + name
    if name == 'teacher-probe':
        policy_path = Path.cwd() / 'tools/xtoken_all/teacher_probe_policy.yaml'
        policy = OmegaConf.to_container(OmegaConf.load(policy_path), resolve=True)
        assert receipt['teacher_probe_policy_sha256'] == sha256(policy_path)
        validate_teacher_probe(receipt, policy, cfg)
    if name in ('gym-preflight', 'ray-stream-check'):
        validate_gym_actor_report(receipt['actor_runtime'], actor_runtime)
    if name == 'gym-preflight':
        services = receipt['service_imports']
        assert len(services) == 3 and {item['service'] for item in services} == {'model', 'qa', 'agent'}
        assert all(item['app_import_completed'] and not item['servers_started'] for item in services)
lease = json.loads(allocation.read_text())
assert lease['success'] and len(lease['gpu_indices']) == cfg['cluster']['gpus_per_node'] and lease['ray_ownership_verified']
assert lease['ray_address'] == cfg['on_policy']['nemo_gym']['ray_address'], 'Ray head differs from rendered config'
assert 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(lease['checked_at_utc'])).total_seconds() < 300
raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
devices = {int(line.split(',')[0]): line.split(',')[1].strip() for line in raw.splitlines()}
assert [devices[i] for i in lease['gpu_indices']] == lease['gpu_uuids']
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
assert not any(line.split(',')[0].strip() in lease['gpu_uuids'] for line in apps.splitlines()), 'Target GPUs are occupied'
import ray
ray.init(address=lease['ray_address'], log_to_driver=False)
try:
    nodes = [node['NodeID'] for node in ray.nodes() if node['Alive']]
    assert sorted(nodes) == sorted(lease['ray_node_ids']) and ray.cluster_resources()['GPU'] == len(lease['gpu_indices'])
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
