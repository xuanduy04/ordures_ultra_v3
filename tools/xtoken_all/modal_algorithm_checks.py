"""Budget-bounded CPU, NCCL and native-worker checks on a pinned existing image.

Invoke with the pre-existing Modal SDK, not a new local Python environment.
Each GPU call has a 480-second execution ceiling and a separate startup bound.
"""

from datetime import datetime, timezone
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time
import uuid

import modal


ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/work/source")
IMAGE_DIGEST = "nvcr.io/nvidia/nemo-rl@sha256:060a3337d167e66230ce8b3dac9e9b733ad25bc997f8a0121eebccf81cbf1b23"
IMAGE = modal.Image.from_registry(IMAGE_DIGEST)
for local, remote in (("nemo_rl", "nemo_rl"), ("tools/xtoken_all", "tools/xtoken_all"),
                      ("examples", "examples"), ("custom_utils", "custom_utils"),
                      ("tests/unit/models/automodel", "tests/unit/models/automodel"),
                      ("tests/unit/algorithms/x_token", "tests/unit/algorithms/x_token")):
    IMAGE = IMAGE.add_local_dir(str(ROOT / local), "/work/source/" + remote, ignore=["**/__pycache__/**"])
for name in ("pyproject.toml", "uv.lock", "default_runtime_env.yaml", ".python-version", "run_xtoken.sh", "ensure_runtime.sh"):
    IMAGE = IMAGE.add_local_file(str(ROOT / name), "/work/source/" + name)
app = modal.App("nemo-algorithm-qualification-20261028")
PYTHON = "/opt/nemo_rl_venv/bin/python"
POLICY_PYTHON = "/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python"
MODES = "dpca,ctopd_full,ctopd_topk,simct_full,simct_topk,grass_chunk_fixed_sigma"


@contextmanager
def locked_ledger(ledger):
    """Cross-process lock; an abandoned lock fails closed rather than overspends."""
    lock = ledger.with_name(ledger.name + ".lock")
    deadline = time.monotonic() + 10
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("Budget ledger is locked; inspect its owner before retry")
            time.sleep(.05)
    try:
        yield
    finally:
        lock.rmdir()


def safe_log(value):
    lines = [line if not any(key in line for key in ("env_vars", "MODAL_IDENTITY_TOKEN", "TOKEN_SECRET", "os.environ"))
             else "[environment line omitted]" for line in value.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\bhf_[A-Za-z0-9]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b", "[credential redacted]", text)


def child_env():
    env = {key: value for key, value in os.environ.items() if key not in {
        "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_IDENTITY_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
        "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}}
    env.update(PYTHONPATH="/work/source", NRL_IGNORE_VERSION_MISMATCH="1", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", UV_OFFLINE="1", OMP_NUM_THREADS="2",
               VLLM_USE_DEEP_GEMM="0", VLLM_MOE_USE_DEEP_GEMM="0")
    return env


def source_fingerprint():
    digest = hashlib.sha256()
    count = 0
    for path in sorted(Path("/work/source").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(path.relative_to("/work/source").as_posix().encode() + b"\0" + path.read_bytes())
            count += 1
    return {"source_payload_sha256": digest.hexdigest(), "source_payload_files": count}


def execute(command, env, timeout):
    with subprocess.Popen(command, cwd="/work/source", env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as run:
        try:
            stdout, stderr = run.communicate(timeout=timeout)
            code = run.returncode
        except subprocess.TimeoutExpired:
            os.killpg(run.pid, signal.SIGTERM)
            try:
                stdout, stderr = run.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(run.pid, signal.SIGKILL)
                stdout, stderr = run.communicate()
            code = 124
            stderr += "\nQualification command exceeded its scoped timeout."
    return {"exit_code": code, "stdout": safe_log(stdout), "stderr": safe_log(stderr)}


@app.function(image=IMAGE, cpu=2, memory=8192, timeout=420, startup_timeout=120, scaledown_window=2, retries=0)
def cpu_checks():
    env = child_env()
    light = execute([PYTHON, "-B", "-m", "pytest", "--noconftest", "-o", "addopts=", "-p", "no:cacheprovider",
                     "tools/xtoken_all/tests", "-q", "--tb=short"], env, 180)
    unit = execute([PYTHON, "-B", "-m", "pytest", "--confcutdir=tests/unit/algorithms/x_token", "-o", "addopts=",
                    "-p", "no:cacheprovider", "tests/unit/algorithms/x_token", "-q", "--tb=short"], env, 180)
    # Reuse the driver's pure-Python pytest tool only after the worker's own
    # site-packages. Never prepend the driver's Torch/native packages.
    sites = execute([PYTHON, "-B", "-c", "import json,site;print(json.dumps(site.getsitepackages()))"], env, 15)
    driver_sites = json.loads(sites["stdout"])
    worker_pytest = "import sys;sys.path.extend(" + repr(driver_sites) + ");import pytest;raise SystemExit(pytest.main(sys.argv[1:]))"
    automodel = execute([POLICY_PYTHON, "-B", "-c", worker_pytest, "--noconftest", "-o", "addopts=", "-p", "no:cacheprovider",
                        "tests/unit/models/automodel/test_automodel_train.py", "-k", "TestForwardWithPostProcessingFn",
                        "-q", "-rs", "--tb=short"], dict(env, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"), 90)
    return {**source_fingerprint(), "light": light, "unit": unit, "automodel": automodel,
            "success": light["exit_code"] == unit["exit_code"] == 0 and automodel["exit_code"] in (0, 5)}


@app.function(image=IMAGE, cpu=1, memory=2048, timeout=90, startup_timeout=120, scaledown_window=2, retries=0)
def runtime_inventory():
    paths = [Path(PYTHON), *sorted(Path("/opt/ray_venvs").glob("*/bin/python"))]
    results = []
    script = "import importlib.util,json,sys; print(json.dumps({'python':sys.executable,'automodel_spec':str(importlib.util.find_spec('nemo_automodel')),'torch_spec':str(importlib.util.find_spec('torch'))}))"
    for path in paths:
        if "dtensor" in str(path) or str(path) == PYTHON:
            results.append({"interpreter": str(path), **execute([str(path), "-B", "-c", script], child_env(), 20)})
    mesh_source = execute([PYTHON, "-B", "-c", "from pathlib import Path; p=Path('/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/nemo_automodel/components/distributed'); print((p/'fsdp2.py').read_text()); s=(p/'parallelizer.py').read_text().splitlines(); print(chr(10).join(s[190:250]+s[575:635]))"], child_env(), 10)
    return {"interpreters": results, "mesh_source": mesh_source,
            "success": mesh_source["exit_code"] == 0 and all(row["exit_code"] == 0 for row in results)}


@app.function(image=IMAGE, cpu=8, memory=16384, timeout=480, startup_timeout=120, scaledown_window=2, retries=0)
def gpu_checks(count, accelerator):
    if count not in (2, 3, 4):
        raise ValueError("Physical allocation must match the requested 2/3/4 GPUs")
    root = Path(tempfile.mkdtemp(prefix="nemo-algo-"))
    env = child_env()
    metadata_run = execute([PYTHON, "-B", "-c", "import json,torch; print(json.dumps({'gpu_count':torch.cuda.device_count(),'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'torch':torch.__version__,'cuda':torch.version.cuda}))"], env, 30)
    if metadata_run["exit_code"]:
        return {"success": False, "metadata": metadata_run}
    metadata = json.loads(metadata_run["stdout"].splitlines()[-1])
    if metadata["gpu_count"] != count:
        raise ValueError("Physical CUDA allocation differs from requested GPU count")
    if not all(accelerator in name for name in metadata["gpu_names"]):
        raise ValueError("Physical accelerator differs from the explicitly requested hardware")
    print("ALGO_PHASE=" + json.dumps({"phase": "nccl", **metadata}), flush=True)
    launcher = [PYTHON, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=" + str(count)]
    ddp = execute(launcher + ["tools/xtoken_all/tests/variant_ddp_probe.py", "--output", str(root / "ddp.json")], env, 150)
    result = {**metadata, **source_fingerprint(), "image_digest": IMAGE_DIGEST, "ddp": ddp}
    if ddp["exit_code"]:
        print("ALGO_FAILURE=" + json.dumps(ddp)[-6000:], flush=True)
        result["success"] = False
        return result
    result["ddp_report"] = json.loads((root / "ddp.json").read_text())
    native_launcher = [POLICY_PYTHON, "-B", "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=" + str(count)]
    native_env = dict(env, XTOKEN_FIXTURE_PATH=str(root / "native"), XTOKEN_VARIANT_MODES=MODES)
    print("ALGO_PHASE=native", flush=True)
    result["native"] = execute(native_launcher + ["tests/unit/algorithms/x_token/native_worker_smoke.py"], native_env, 210)
    if result["native"]["exit_code"]:
        print("ALGO_FAILURE=" + json.dumps(result["native"])[-6000:], flush=True)
    if result["native"]["exit_code"] == 0:
        result["native_reports"] = [json.loads(path.read_text()) for path in sorted((root / "native").glob("native-variants-rank-*.json"))]
        if len(result["native_reports"]) != count:
            raise ValueError("Missing native worker result for one or more ranks")
    if count == 2 and result["native"]["exit_code"] == 0:
        print("ALGO_PHASE=legacy_ipc", flush=True)
        legacy_env = dict(env, XTOKEN_FIXTURE_PATH=str(root / "legacy"))
        result["legacy_ipc"] = execute(native_launcher + ["tests/unit/algorithms/x_token/native_worker_smoke.py"], legacy_env, 90)
    result["success"] = result["native"]["exit_code"] == 0 and ("legacy_ipc" not in result or result["legacy_ipc"]["exit_code"] == 0)
    return result


@app.local_entrypoint()
def main(phase: str, count: int, output_dir: str, accelerator: str = "B200"):
    profile = os.environ["MODAL_PROFILE"]
    if profile not in ("nguyenngocminh30012005", "nguyenepti") or phase not in ("cpu", "gpu", "inspect"):
        raise ValueError("Use the two authorized profiles and cpu/gpu phase")
    if accelerator not in ("B200", "H100") or count not in (2, 3, 4):
        raise ValueError("Explicit B200/H100 and 2/3/4 GPU topology required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    billed = subprocess.run(["modal", "billing", "report", "--for", "this month", "--show-resources", "--json"],
                            capture_output=True, text=True, check=True)
    rows = json.loads(billed.stdout)
    reported = sum((Decimal(row["cost"]) for row in rows), Decimal(0))
    ledger = output / (profile + "-reservations.json")
    # Rates verified at https://modal.com/pricing on 2026-10-28. Reserve startup,
    # 480s execution and 2s idle, CPU8/RAM16GiB, plus conservative rounding.
    gpu_bounds = {"B200": {2: "3", 3: "4", 4: "5"}, "H100": {2: "2", 3: "3", 4: "4"}}
    bound = Decimal(gpu_bounds[accelerator][count]) if phase == "gpu" else Decimal("0.15")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    label = profile + "-" + phase + "-" + str(count) + "-" + accelerator + "-" + stamp + "-" + uuid.uuid4().hex[:8]
    with locked_ledger(ledger):
        previous = json.loads(ledger.read_text()) if ledger.is_file() else []
        reserved = sum((Decimal(row["upper_bound_usd"]) for row in previous), Decimal(0))
        allowed = max(reported, reserved) + bound + Decimal("2") <= Decimal("20")
        guard = {"profile": profile, "reported_month_usage_usd": str(reported), "local_reserved_upper_usd": str(reserved),
             "call_bound_usd": str(bound), "reserve_usd": "2", "authorized_budget_usd": "20", "allowed": allowed,
             "accelerator": accelerator, "startup_timeout_s": 120, "scaledown_window_s": 2,
             "note": "Task began at reported usage zero on both profiles; CLI usage can lag and is not remaining credits"}
        (output / (label + "-guard.json")).write_text(json.dumps(guard, indent=2))
        if allowed:
            previous.append({"id": label, "upper_bound_usd": str(bound), "state": "reserved"})
            ledger.write_text(json.dumps(previous, indent=2))
    print("ALGO_BUDGET=" + json.dumps(guard), flush=True)
    if not allowed:
        raise RuntimeError("Budget guard rejected this call")
    # Windows managed-worktree .git paths are not always understood by WSL Git.
    # The remote payload hash remains authoritative; metadata must not discard results.
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.PIPE).strip()
        source_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.PIPE).strip())
    except subprocess.CalledProcessError:
        source_commit, source_dirty = None, None
    submitted = time.monotonic()
    result = runtime_inventory.remote() if phase == "inspect" else cpu_checks.remote() if phase == "cpu" else gpu_checks.with_options(gpu=accelerator + ":" + str(count)).remote(count, accelerator)
    elapsed = time.monotonic() - submitted
    result["submit_to_result_seconds"] = elapsed
    if phase == "gpu":
        rate = {"B200": Decimal("0.001736"), "H100": Decimal("0.001097")}[accelerator]
        # Submission-to-result includes queue and all application load. Allow a
        # further 30s for the 2s idle window/teardown and CPU32/RAM64GiB (above
        # this tiny fixture's request). This is an upper estimate, never a bill.
        settled = ((Decimal(str(elapsed + 30)) * (count * rate + Decimal("0.0000131") * 32 + Decimal("0.00000222") * 64))
                   + Decimal("0.05")).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
        with locked_ledger(ledger):
            current = json.loads(ledger.read_text())
            record = next(row for row in current if row["id"] == label)
            record.update(initial_upper_bound_usd=str(bound), upper_bound_usd=str(min(bound, settled)),
                            state="completed" if result["success"] else "failed", submit_to_result_seconds=elapsed,
                            bound_basis="observed submission-to-result plus30s, GPU rate and CPU32/RAM64; not billed actual")
            ledger.write_text(json.dumps(current, indent=2))
    result["source_commit_at_dispatch"] = source_commit
    result["source_dirty_at_dispatch"] = source_dirty
    (output / (label + "-result.json")).write_text(json.dumps(result, indent=2))
    print("ALGO_RESULT=" + json.dumps({"id": label, "success": result["success"], "result_file": str(output / (label + "-result.json"))}), flush=True)
    if not result["success"]:
        print(json.dumps(result)[-20000:], flush=True)
        raise RuntimeError("Qualification failed; inspect persisted evidence before retry")
