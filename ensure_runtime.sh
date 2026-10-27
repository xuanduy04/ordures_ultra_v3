#!/usr/bin/env bash
# Bootstrap paths may be edited here or supplied through these environment variables.
set -euo pipefail
XTOKEN_ENSURE_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XTOKEN_ALL_ROOT="${XTOKEN_ALL_ROOT:-/workspace/storage-shared/nlp/tungks/xtoken_all}"
XTOKEN_RUNTIME_ARCHIVE="${XTOKEN_RUNTIME_ARCHIVE:-$XTOKEN_ALL_ROOT/tmp/runtime-opt-20261027.tar.gz}"
XTOKEN_SUPER_SOURCE="${XTOKEN_SUPER_SOURCE:-/workspace/storage-shared/nlp/vai-1.4-30b/nemo-rl/custom_opt_ultra_selfbuilt_nano/nemo-rl}"
CUSTOM_NLTK_DATA_DIR="${CUSTOM_NLTK_DATA_DIR:-$(dirname "$XTOKEN_SUPER_SOURCE")/nltk_data}"
[[ -z "${NRL_PYTHON:-}" || "$NRL_PYTHON" == /opt/nemo_rl_venv/bin/python ]] || { echo 'This audited bundle uses /opt/nemo_rl_venv/bin/python; another NRL_PYTHON profile is preserved, not overridden.' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Runtime requires Linux x86_64.' >&2; exit 2; }
XTOKEN_BOOTSTRAP_PY=""
for candidate in /usr/bin/python3 /usr/local/bin/python3 python3 /opt/nemo_rl_venv/bin/python; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -I -B -c 'import sys; assert sys.version_info >= (3, 9)' >/dev/null 2>&1; then
    XTOKEN_BOOTSTRAP_PY="$(command -v "$candidate")"; break
  fi
done
[[ -n "$XTOKEN_BOOTSTRAP_PY" ]] || { echo 'Need a working bootstrap Python >=3.9; no packages or downloads are required.' >&2; exit 2; }
"$XTOKEN_BOOTSTRAP_PY" -I -B - --source-root "$XTOKEN_ENSURE_SOURCE_ROOT" --archive "$XTOKEN_RUNTIME_ARCHIVE" --super-source "$XTOKEN_SUPER_SOURCE" --nltk-data "$CUSTOM_NLTK_DATA_DIR" "$@" <<'XTOKEN_ENSURE_PY'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile

# Identity and fixed prefixes of the independently audited archive, not a solver recipe.
ARCHIVE_SHA256 = '64abccb824aaa0fda749087a3961dad7e804d985b40a0773deaa63fce27f508d'
ARCHIVE_BYTES = 13336103402
MIN_FREE_BYTES = 45 * 1024**3
PYTHON_HOME = Path('/root/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu')
DRIVER = Path('/opt/nemo_rl_venv')
RAY_STORE = Path('/opt/ray_venvs')
GYM_STORE = Path('/opt/gym_venvs')
SOURCE_ALIAS = Path('/opt/nemo-rl')
NODE_STATE_DIR = Path('/tmp')
COMMON = {'torch': '2.10.0+cu129', 'ray': '2.55.1', 'transformers': '4.57.1'}
VLLM_VERSION = '0.1.dev14482+gd229daa09.d20260727.precompiled'
ROLES = (
    ('policy', Path('/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2')),
    ('generation', Path('/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker')),
    ('gym_actor', Path('/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym')),
)
SERVICES = (
    ('model', Path('/opt/gym_venvs/responses_api_models/vllm_model/.venv')),
    ('agent', Path('/opt/gym_venvs/responses_api_agents/simple_agent/.venv')),
    ('qa', Path('/opt/gym_venvs/resources_servers/general_qa/.venv')),
)
DRIVER_PROBE = '''import importlib.metadata as md,json,sys
versions={name:md.version(name) for name in ('torch','ray','transformers')}
import torch,ray,transformers,encodings,ssl,zlib
print('XTOKEN_DRIVER_PROBE='+json.dumps({'versions':versions,'python':sys.version.split()[0],'prefix':sys.prefix,'base_prefix':sys.base_prefix}))
'''


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def occupied(prefixes):
    """Never print cmdlines or environments, which can contain credentials."""
    matches = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            executable = os.readlink(directory / 'exe').removesuffix(' (deleted)')
            arguments = (directory / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(value == str(prefix) or value.startswith(str(prefix) + '/')
                   for value in [executable, *arguments] for prefix in prefixes):
                matches.append(int(directory.name))
        except OSError:
            continue
    return matches


def guard_idle(prefixes):
    pids = occupied(prefixes)
    if pids:
        raise RuntimeError('Runtime prefixes are in use by PIDs ' + ','.join(map(str, pids)) + '; nothing stopped or overwritten')


def validate_sources(args):
    source, super_source = args.source_root.resolve(), args.super_source.resolve()
    if not (source / 'nemo_rl/algorithms/x_token/gym_compat.py').is_file():
        raise FileNotFoundError('Use the X-Token source checkout containing this helper')
    for relative in ('nemo_rl', '3rdparty/vllm', '3rdparty/Automodel-workspace',
                     '3rdparty/Megatron-LM-workspace', '3rdparty/Megatron-Bridge-workspace'):
        if not (super_source / relative).is_dir():
            raise FileNotFoundError('Stage the existing Super ZIP on shared storage first: ' + str(super_source / relative))
    gym = super_source / '3rdparty/Gym-workspace/Gym'
    spec = importlib.util.spec_from_file_location('xtoken_ensure_gym_contract', source / 'nemo_rl/algorithms/x_token/gym_compat.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.reject_inherited_gym_config(os.environ)
    report = module.verify_gym_source(gym, source / 'tools/xtoken_all/super-source-contract.json')
    if not (args.nltk_data / 'tokenizers/punkt_tab/english/abbrev_types.txt').is_file():
        raise FileNotFoundError('Stage the existing NLTK punkt_tab data: ' + str(args.nltk_data))
    return source, super_source, gym, report


def ensure_source_alias(super_source, check_only):
    if SOURCE_ALIAS.is_symlink() and not SOURCE_ALIAS.exists():
        raise RuntimeError('Dangling source alias preserved: ' + str(SOURCE_ALIAS))
    if SOURCE_ALIAS.exists():
        if not SOURCE_ALIAS.is_dir():
            raise NotADirectoryError('Existing source binding is not a directory; preserved: ' + str(SOURCE_ALIAS))
        return 'existing_preserved'
    if check_only:
        raise FileNotFoundError('Source binding is absent: ' + str(SOURCE_ALIAS))
    guard_idle([DRIVER, PYTHON_HOME, RAY_STORE, GYM_STORE])
    SOURCE_ALIAS.symlink_to(super_source, target_is_directory=True)
    return 'created_shared_source_alias'


def probe(command, env, marker):
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    if result.returncode:
        # Keep bounded diagnostics; no HTTP credentials are passed to these probes.
        raise RuntimeError('Runtime import/API check failed: ' + str(command[0]) + '\n' + result.stderr[-3500:])
    rows = [json.loads(line.split('=', 1)[1]) for line in result.stdout.splitlines() if line.startswith(marker + '=')]
    if len(rows) != 1:
        raise RuntimeError('Missing unique runtime probe report: ' + marker)
    return rows[0]


def check_runtime(source, gym, nltk_data, existing_only):
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join((str(source), str(gym))), PYTHONDONTWRITEBYTECODE='1',
               UV_OFFLINE='1', HF_HUB_OFFLINE='1', NRL_IGNORE_VERSION_MISMATCH='1',
               NEMO_RL_VENV_DIR=str(RAY_STORE), NEMO_GYM_VENV_DIR=str(GYM_STORE),
               CUSTOM_NLTK_DATA_DIR=str(nltk_data))
    reports = []
    if PYTHON_HOME.exists() and not (PYTHON_HOME / 'bin/python3.12').is_file():
        raise FileNotFoundError('Existing base Python is incomplete; preserved: ' + str(PYTHON_HOME))
    requirements = [('driver', DRIVER), *ROLES, *SERVICES]
    for role, prefix in requirements:
        if not prefix.exists() and existing_only:
            continue
        executable = prefix / 'bin/python'
        if not executable.is_file():
            raise FileNotFoundError('Existing/incomplete runtime preserved: ' + str(executable))
        if role == 'driver':
            report = probe([str(executable), '-B', '-c', DRIVER_PROBE], env, 'XTOKEN_DRIVER_PROBE')
            if report['python'] != '3.12.12' or Path(report['prefix']) != prefix or Path(report['base_prefix']) != PYTHON_HOME:
                raise RuntimeError('Driver Python version/prefix differs from the audited runtime')
            versions = report['versions']
        elif role in dict(ROLES):
            command = [str(executable), '-B', str(source / 'tools/xtoken_all/runtime_probe.py'), '--role', role]
            if role == 'gym_actor':
                command.extend(['--gym-root', str(gym)])
            report = probe(command, env, 'XTOKEN_RUNTIME_PROBE')
            if Path(report['python_prefix']) != prefix:
                raise RuntimeError('Worker lost its virtualenv prefix: ' + role)
            versions = {key: report[key] for key in COMMON}
            if role == 'generation' and report['vllm'] != VLLM_VERSION:
                raise RuntimeError('Generation vLLM build differs from the audited runtime')
            if role == 'policy' and report['nemo-automodel'] != '0.2.0':
                raise RuntimeError('Policy Automodel differs from the audited runtime')
        else:
            report = probe([str(executable), '-B', str(source / 'tools/xtoken_all/gym_service_probe.py'),
                '--service', role, '--gym-root', str(gym)], env, 'XTOKEN_GYM_SERVICE_PROBE')
            if Path(report['python_prefix']) != prefix:
                raise RuntimeError('Gym service lost its virtualenv prefix: ' + role)
            versions = None
        if versions is not None and versions != COMMON:
            raise RuntimeError('Runtime versions differ: ' + role + ' ' + json.dumps(versions))
        reports.append({'role': role, 'report': report})
    return reports


def install_missing(archive, roots, required_prefixes):
    """Only install absent venv leaves; never merge into an existing environment."""
    if shutil.which('tar') is None:
        raise FileNotFoundError('GNU tar is required; use a prepared node image, no package installation is attempted')
    if archive.stat().st_size != ARCHIVE_BYTES or digest(archive) != ARCHIVE_SHA256:
        raise RuntimeError('Pinned runtime archive size/SHA256 mismatch; nothing extracted')
    if shutil.disk_usage(roots[0].parent).free < MIN_FREE_BYTES:
        raise RuntimeError('Need at least45 GiB free for the audited archive and its filesystem metadata')
    stage = Path(tempfile.mkdtemp(prefix='.xtoken-runtime-stage-', dir=roots[0].parent))
    installed = []
    completed = False
    try:
        # This exact immutable archive was audited for traversal and link closure.
        # GNU tar retains its 1.17M hardlinks; Python/ZIP copying would expand them.
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(stage), '--no-same-owner'], check=True)
        driver, ray, services, python = [stage / str(path).lstrip('/') for path in roots]
        leaves = [python, driver, *sorted(ray.iterdir()), *sorted(services.glob('*/*/.venv'))]
        if not all((stage / str(path).lstrip('/')).is_dir() for path in required_prefixes):
            raise RuntimeError('Archive does not contain the required runtime prefixes')
        # Validate every destination before publishing any prefix.
        for item in leaves:
            relative = item.relative_to(stage)
            target = Path('/') / relative
            if target.is_symlink():
                raise RuntimeError('Existing runtime prefix is a symlink; preserved: ' + str(target))
            if target.exists():
                if target == roots[-1]:
                    for critical in ('bin/python3.12', 'lib/python3.12/encodings/__init__.py'):
                        if digest(target / critical) != digest(item / critical):
                            raise RuntimeError('Existing base Python differs; preserved: ' + str(target))
        destinations = [(item, Path('/') / item.relative_to(stage)) for item in leaves]
        guard_idle([target for _, target in destinations if not target.exists()])
        for item, target in destinations:
            if target.exists():
                continue
            guard_idle([target])
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.stat().st_dev == target.parent.stat().st_dev:
                item.rename(target)
            else:
                transfer = Path(tempfile.mkdtemp(prefix='.xtoken-runtime-copy-', dir=target.parent))
                shutil.copytree(item, transfer / 'payload', symlinks=True)
                (transfer / 'payload').rename(target)
                transfer.rmdir()
            installed.append(str(target))
        completed = True
        return installed
    finally:
        # Retain the private stage on failure for inspection; never delete a live prefix.
        if completed:
            assert stage.parent.resolve() == roots[0].parent.resolve() and stage.name.startswith('.xtoken-runtime-stage-')
            shutil.rmtree(stage)
        else:
            print('XTOKEN_RUNTIME_STAGE=' + str(stage), flush=True)


def ensure(args):
    source, super_source, gym, contract = validate_sources(args)
    roots = [DRIVER, RAY_STORE, GYM_STORE, PYTHON_HOME]
    required = [DRIVER, *[prefix for _, prefix in ROLES], *[prefix for _, prefix in SERVICES], PYTHON_HOME]
    missing = [prefix for prefix in required if not prefix.exists()]
    if any(prefix.is_symlink() for prefix in required):
        raise RuntimeError('Runtime environment prefixes must be directories, not aliases')
    if args.check_only and missing:
        raise FileNotFoundError('Runtime not provisioned: ' + ', '.join(map(str, missing)))
    if missing and not args.archive.is_file():
        raise FileNotFoundError('Stage the already audited runtime archive on shared storage: ' + str(args.archive))
    if missing:
        guard_idle([prefix for prefix in required if prefix not in missing])
    binding = ensure_source_alias(super_source, args.check_only)
    if missing:
        check_runtime(source, gym, args.nltk_data, existing_only=True)
        installed = install_missing(args.archive, roots, required)
    else:
        installed = []
    reports = check_runtime(source, gym, args.nltk_data, existing_only=False)
    return {'success': True, 'state': 'ready_for_training_preflight', 'restored_prefixes': installed,
        'reused': not installed, 'archive_sha256': ARCHIVE_SHA256 if installed else None,
        'source_binding': binding, 'source_root': str(source), 'gym_source': str(gym),
        'source_contract': contract, 'host': socket.gethostname(), 'libc': platform.libc_ver(),
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': reports,
        'cuda_operations_requested': False, 'gpu_training_qualified': False}


def main():
    parser = argparse.ArgumentParser(description='Restore missing audited runtime; verify existing environments before reuse')
    for flag in ('source-root', 'archive', 'super-source', 'nltk-data'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--check-only', action='store_true', help='Read-only verification; no restore, source alias or receipt writes')
    args = parser.parse_args()
    receipt = NODE_STATE_DIR / ('xtoken-runtime-ready-' + socket.gethostname() + '-' + str(os.geteuid()) + '.json')
    lock = None
    locked = False
    try:
        if not args.check_only:
            import fcntl
            lock = receipt.with_suffix('.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another ensure_runtime setup is running on this node; retry after it finishes') from None
            locked = True
        result = ensure(args)
        if not args.check_only:
            temporary = receipt.with_suffix('.pending')
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(receipt)
        print('XTOKEN_RUNTIME_ACTION=' + ('reused_without_extraction' if result['reused'] else 'restored_missing_environments'), flush=True)
        print('XTOKEN_RUNTIME_READY=' + json.dumps(result), flush=True)
    except Exception as error:
        failure = {'success': False, 'error': type(error).__name__ + ': ' + str(error),
            'checked_at_utc': datetime.now(timezone.utc).isoformat()}
        if locked:
            temporary = receipt.with_suffix('.pending')
            temporary.write_text(json.dumps(failure, indent=2) + '\n')
            temporary.replace(receipt)
        print('XTOKEN_RUNTIME_FAILED=' + json.dumps(failure), file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
XTOKEN_ENSURE_PY
export NRL_PYTHON=/opt/nemo_rl_venv/bin/python
export NEMO_RL_VENV_DIR=/opt/ray_venvs NEMO_GYM_VENV_DIR=/opt/gym_venvs
export XTOKEN_GYM_SOURCE="$XTOKEN_SUPER_SOURCE/3rdparty/Gym-workspace/Gym"
export CUSTOM_NLTK_DATA_DIR
