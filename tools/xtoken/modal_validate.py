"""Validate the existing production image on Modal without installing dependencies."""

import json
import os
from pathlib import Path

import modal


G_IMAGE_REF = os.environ["XTOKEN_IMAGE"]
G_VOLUME_NAME = os.environ["XTOKEN_TEST_VOLUME"]
app = modal.App("ordures-xtoken-http")
volume = modal.Volume.from_name(G_VOLUME_NAME, create_if_missing=True, version=2)
G_REGISTRY_SECRET = os.environ.get("XTOKEN_REGISTRY_SECRET")
registry_secret = modal.Secret.from_name(G_REGISTRY_SECRET) if G_REGISTRY_SECRET else None
image = modal.Image.from_registry(G_IMAGE_REF, secret=registry_secret).env(
    {"XTOKEN_IMAGE": G_IMAGE_REF, "XTOKEN_TEST_VOLUME": G_VOLUME_NAME,
     **({"XTOKEN_REGISTRY_SECRET": G_REGISTRY_SECRET} if G_REGISTRY_SECRET else {})}
)
source_image = image
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    for directory in ("nemo_rl", "tests", "examples"):
        source_image = source_image.add_local_dir(
            root / directory, "/opt/nemo-rl/" + directory,
            ignore=["**/__pycache__/**"],
        )


@app.function(image=source_image, volumes={"/evidence": volume}, cpu=4, memory=16384, timeout=600, retries=0)
def cpu_math() -> dict:
    """Check source-only tensor/alignment behavior in the supplied native image."""
    import hashlib
    import subprocess
    import sys
    import xml.etree.ElementTree as element_tree

    paths = [
        "tests/unit/algorithms/x_token/test_token_aligner.py",
        "tests/unit/algorithms/x_token/test_loss_utils.py",
        "tests/unit/algorithms/x_token/test_utils.py",
        "tests/unit/algorithms/x_token/test_xtoken_off_policy_distillation.py",
        "tests/unit/algorithms/x_token/test_on_policy.py",
        "tests/unit/algorithms/x_token/test_remote_teacher.py",
        "tests/unit/algorithms/x_token/test_runtime_compat.py",
        "tests/unit/data/test_cross_tokenizer_collate.py",
    ]
    checksum = hashlib.sha256()
    for path in sorted(Path("/opt/nemo-rl/nemo_rl").rglob("*.py")):
        checksum.update(str(path.relative_to("/opt/nemo-rl")).encode())
        checksum.update(path.read_bytes())
    env = {**os.environ, "UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", "--junitxml=/evidence/cpu-math.xml", *paths]
    with Path("/evidence/cpu-math.log").open("w") as log:
        process = subprocess.Popen(command, cwd="/opt/nemo-rl", env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    result = {"image": G_IMAGE_REF, "source_sha256": checksum.hexdigest(), "exit_code": code,
              "scope": "auxiliary CPU alignment/loss tests, not custom production-image qualification"}
    report = Path("/evidence/cpu-math.xml")
    if report.is_file():
        result["suites"] = [suite.attrib for suite in element_tree.parse(report).getroot()]
    Path("/evidence/cpu-math.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    if code:
        raise RuntimeError("CPU math tests failed: " + str(code))
    return result


@app.function(image=image, volumes={"/evidence": volume}, cpu=2, memory=8192, timeout=600, retries=0)
def probe() -> dict:
    """Record native interpreters and installed versions before overlaying source."""
    import subprocess
    import sys

    interpreters = {Path(sys.executable), Path("/opt/nemo_rl_venv/bin/python")}
    interpreters.update(Path("/opt/ray_venvs").glob("*/bin/python"))
    script = """
import importlib.metadata as metadata
import json, sys
packages = ['torch', 'ray', 'transformers', 'vllm', 'nemo-automodel', 'transformer-engine', 'flash-attn', 'pytest', 'numpy', 'omegaconf', 'pydantic']
versions = {}
for name in packages:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({'executable': sys.executable, 'python': sys.version, 'versions': versions}))
"""
    inventories = []
    for executable in sorted(interpreters):
        if executable.is_file():
            inventories.append(json.loads(subprocess.check_output([str(executable), "-c", script], text=True, timeout=60)))
    result = {"image": G_IMAGE_REF, "inventories": inventories}
    fingerprint = Path("/opt/nemo_rl_container_fingerprint")
    if fingerprint.is_file():
        result["fingerprint"] = json.loads(fingerprint.read_text())
    Path("/evidence/probe.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print("XTOKEN_PROBE_JSON=" + json.dumps(result), flush=True)
    return result


@app.function(image=source_image, volumes={"/evidence": volume}, cpu=2, memory=16384, timeout=180, retries=0)
def compatibility() -> dict:
    """Catch known backend API mismatches on CPU before reserving GPUs."""
    import subprocess

    executable = "/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python"
    script = """
import inspect, json
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2
signature = inspect.signature(FSDP2Manager)
required = ['dp_size', 'dp_replicate_size', 'tp_size', 'cp_size', 'ep_size', 'pp_size',
            'sequence_parallel', 'use_hf_tp_plan', 'mp_policy', 'offload_policy', 'backend',
            'world_size', 'activation_checkpointing', 'custom_tp_plan', 'defer_fsdp_grad_sync']
variadic = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
adapted = {'config', 'device_mesh'}.issubset(signature.parameters)
missing = [] if variadic or adapted else [key for key in required if key not in signature.parameters]
print('XTOKEN_API=' + json.dumps({'compatible': not missing, 'signature': str(signature), 'missing_kwargs': missing}))
"""
    process = subprocess.run([executable, "-c", script], cwd="/opt/nemo-rl",
                              env={**os.environ, "UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1"},
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=150)
    markers = [line.split("XTOKEN_API=", 1)[1] for line in process.stdout.splitlines() if line.startswith("XTOKEN_API=")]
    result = {"image": G_IMAGE_REF, "exit_code": process.returncode, "compatible": False}
    if markers:
        result.update(json.loads(markers[-1]))
    Path("/evidence/compatibility.log").write_text(process.stdout)
    Path("/evidence/compatibility.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print("XTOKEN_COMPATIBILITY=" + json.dumps(result), flush=True)
    return result


@app.function(image=source_image, volumes={"/evidence": volume}, cpu=2, memory=16384, timeout=180, retries=0)
def runtime_api() -> dict:
    """Inspect the installed native APIs before implementing version adapters."""
    import subprocess

    executable = "/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python"
    script = """
import inspect, json
from nemo_automodel.components.distributed import fsdp2, config
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.distributed import mesh_utils
result = {'fsdp_init': inspect.getsource(fsdp2.FSDP2Manager.__init__),
          'fsdp_config': inspect.getsource(config.FSDP2Config),
          'fsdp_methods': [name for name in dir(fsdp2.FSDP2Manager) if not name.startswith('__')],
          'mesh_functions': [name for name in dir(mesh_utils) if 'mesh' in name],
          'mesh_source': inspect.getsource(mesh_utils),
          'checkpoint_init': str(inspect.signature(Checkpointer)),
          'checkpoint_load': inspect.getsource(Checkpointer.load_base_model),
          'checkpoint_config': str(inspect.signature(CheckpointingConfig))}
for name in ['parallelize', 'parallelize_model', '_setup_distributed']:
    method = getattr(fsdp2.FSDP2Manager, name, None)
    if method is not None:
        result[name] = inspect.getsource(method)
print('XTOKEN_RUNTIME_API=' + json.dumps(result))
"""
    process = subprocess.run([executable, "-c", script], cwd="/opt/nemo-rl", text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=150)
    Path("/evidence/runtime-api.log").write_text(process.stdout)
    markers = [line.split("XTOKEN_RUNTIME_API=", 1)[1] for line in process.stdout.splitlines() if line.startswith("XTOKEN_RUNTIME_API=")]
    result = json.loads(markers[-1]) if markers else {"error": process.stdout}
    Path("/evidence/runtime-api.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    for key, value in result.items():
        print(key + "=" + (value if isinstance(value, str) else json.dumps(value)), flush=True)
    if process.returncode:
        raise RuntimeError("Native API inspection failed")
    return {"success": True}


@app.function(image=image, volumes={"/evidence": volume}, cpu=1, memory=2048, timeout=60, retries=0)
def receipts() -> dict:
    """Recover persisted primitive JSON without importing torch on the client."""
    import xml.etree.ElementTree as element_tree

    volume.reload()
    result = {}
    for stage in ["cpu-math", "math-one", "math-two", "gpu-two", "compatibility"]:
        path = Path(f"/evidence/{stage}.json")
        if path.is_file():
            item = json.loads(path.read_text())
            report = Path(f"/evidence/{stage}.xml")
            if report.is_file():
                item["suites"] = [suite.attrib for suite in element_tree.parse(report).getroot()]
            result[stage] = item
    print("XTOKEN_SAVED_RECEIPTS=" + json.dumps(result), flush=True)
    return result


def run_gpu_tests(stage, paths):
    """Persist the complete subprocess log, JUnit and exact source identity."""
    import hashlib
    import subprocess
    import sys

    import torch

    checksum = hashlib.sha256()
    for path in sorted(Path("/opt/nemo-rl/nemo_rl").rglob("*.py")):
        checksum.update(str(path.relative_to("/opt/nemo-rl")).encode())
        checksum.update(path.read_bytes())
    env = {**os.environ, "UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short", f"--junitxml=/evidence/{stage}.xml", *paths]
    with Path(f"/evidence/{stage}.log").open("w") as log:
        process = subprocess.Popen(command, cwd="/opt/nemo-rl", env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    result = {"image": G_IMAGE_REF, "source_sha256": checksum.hexdigest(), "exit_code": code,
              "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "scope": "stock NGC source-overlay validation; custom production image pending"}
    Path(f"/evidence/{stage}.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print("XTOKEN_GPU_RECEIPT=" + json.dumps(result), flush=True)
    if code:
        raise RuntimeError(stage + " tests failed: " + str(code))
    return result


@app.function(image=source_image, volumes={"/evidence": volume}, gpu="B200", cpu=8, memory=32768, timeout=480, retries=0)
def gpu_one(math_only: bool) -> dict:
    paths = ["tests/unit/algorithms/x_token/test_gpu_contract.py"]
    if math_only:
        paths += ["-k", "not native_dtensor"]
    return run_gpu_tests("math-one" if math_only else "gpu-one", paths)


@app.function(image=source_image, volumes={"/evidence": volume}, gpu="B200:2", cpu=12, memory=65536, timeout=600, retries=0)
def gpu_two(math_only: bool) -> dict:
    paths = ["tests/unit/algorithms/x_token/test_loss_utils.py", "tests/unit/algorithms/x_token/test_gpu_contract.py"]
    if math_only:
        paths += ["-k", "not native_dtensor"]
    return run_gpu_tests("math-two" if math_only else "gpu-two", paths)


@app.function(image=source_image, volumes={"/evidence": volume}, gpu="B200:2", cpu=12, memory=65536, timeout=600, retries=0)
def end_to_end(mode: str) -> dict:
    """Use tiny offline fixtures with the actual NeMo entrypoint and vLLM refit."""
    import subprocess
    import sys

    if mode not in {"off", "on"}:
        raise ValueError(mode)
    command = [sys.executable, "-m", "tests.unit.algorithms.x_token.runtime_e2e",
               "--mode", mode, "--output", f"/evidence/e2e-{mode}"]
    log_path = Path(f"/evidence/e2e-{mode}.log")
    environment = {**os.environ, "UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
                   "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd="/opt/nemo-rl", env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    output = log_path.read_text()
    rollouts = [json.loads(line.split("XTOKEN_ROLLOUT=", 1)[1]) for line in output.splitlines()
                if line.startswith("XTOKEN_ROLLOUT=")]
    success = code == 0 and "Max steps reached, stopping training." in output
    if mode == "on":
        success &= [item["policy_version"] for item in rollouts if item["training"]] == [0, 1]
    result = {"image": G_IMAGE_REF, "mode": mode, "exit_code": code,
              "success": success, "rollouts": rollouts, "fixture": "offline tiny Llama with permuted teacher token IDs",
              "scope": "stock candidate; custom production image and full-model training unqualified"}
    Path(f"/evidence/e2e-{mode}.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    print("XTOKEN_E2E=" + json.dumps(result), flush=True)
    if not success:
        raise RuntimeError("end-to-end " + mode + " did not complete two updates")
    return result


@app.function(image=source_image, volumes={"/evidence": volume}, cpu=2, memory=8192, timeout=180, retries=0)
def prepare_http_fixture() -> dict:
    """Create tiny fixtures remotely, before reserving the teacher GPU."""
    from tests.unit.algorithms.x_token.runtime_e2e import create_model
    from nemo_rl.algorithms.xtoken_off_policy_distillation import MasterConfig
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from omegaconf import OmegaConf

    register_omegaconf_resolvers()
    for mode in ("off", "on"):
        config = MasterConfig(**OmegaConf.to_container(load_config(
            f"examples/configs/xtoken_remote_{mode}_policy_distillation.yaml"), resolve=True))
        assert config.teachers[0].remote["model_vocab_size"] == 151936
    create_model(Path("/evidence/http-fixtures/teacher"), teacher=True)
    volume.commit()
    return {"teacher_model": "/evidence/http-fixtures/teacher"}


@app.function(image=source_image, volumes={"/evidence": volume}, gpu="B200", cpu=12, memory=65536, timeout=900, retries=0)
def http_student(teacher_url: str, modes: list[str]) -> dict:
    """Run actual training with a separate, unchanged vLLM teacher process."""
    import subprocess
    import sys
    import math

    run_gpu_tests("dense-http-math", ["tests/unit/algorithms/x_token/test_gpu_contract.py", "-k", "dense"])
    results = {}
    for mode in modes:
        command = [sys.executable, "-m", "tests.unit.algorithms.x_token.runtime_e2e",
                   "--mode", mode, "--output", f"/evidence/http-{mode}", "--remote-url", teacher_url]
        log_path = Path(f"/evidence/http-{mode}.log")
        environment = {**os.environ, "UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
                       "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd="/opt/nemo-rl", env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        output = log_path.read_text()
        rollouts = [json.loads(line.split("XTOKEN_ROLLOUT=", 1)[1]) for line in output.splitlines()
                    if line.startswith("XTOKEN_ROLLOUT=")]
        success = code == 0 and "Max steps reached, stopping training." in output
        steps = [json.loads(line.split("XTOKEN_TRAIN_STEP=", 1)[1]) for line in output.splitlines()
                 if line.startswith("XTOKEN_TRAIN_STEP=")]
        scores = [json.loads(line.split("XTOKEN_TEACHER_SCORE=", 1)[1]) for line in output.splitlines()
                  if line.startswith("XTOKEN_TEACHER_SCORE=")]
        success &= [step["step"] for step in steps] == [1, 2]
        success &= all(math.isfinite(step["loss"]) and math.isfinite(step["grad_norm"]) and step["grad_norm"] > 0 for step in steps)
        success &= len(scores) >= 2 and all(score["model_vocab_size"] == 32 for score in scores)
        if mode == "on":
            success &= [item["policy_version"] for item in rollouts if item["training"]] == [0, 1]
        result = {"image": G_IMAGE_REF, "teacher_image": os.environ["XTOKEN_TEACHER_IMAGE"],
                  "mode": mode, "exit_code": code, "success": success, "rollouts": rollouts,
                  "training_steps": steps, "teacher_scores": scores,
                  "topology": "1 B200 student + 1 B200 separate vLLM teacher",
                  "fixture": "tiny Llama, permuted tokenizer IDs and separate chat templates",
                  "scope": "mechanism smoke test; no full 235B or company-node qualification"}
        results[mode] = result
        Path(f"/evidence/http-{mode}.json").write_text(json.dumps(result, indent=2))
        volume.commit()
        print("XTOKEN_HTTP_E2E=" + json.dumps(result), flush=True)
        if not success:
            raise RuntimeError("HTTP full pipeline failed in " + mode + "; see persisted log")
    return results


def run_http_pipeline(modes):
    """Own and tear down the isolated teacher; never contact company servers."""
    import secrets
    import time
    import urllib.error
    import urllib.request

    fixture = prepare_http_fixture.remote()
    teacher_image = modal.Image.from_registry(os.environ["XTOKEN_TEACHER_IMAGE"])
    key = secrets.token_urlsafe(32)
    authentication = modal.Secret.from_dict({"VLLM_API_KEY": key})
    sandbox = modal.Sandbox.create(
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", fixture["teacher_model"], "--served-model-name", "xtoken-fixture",
        "--host", "0.0.0.0", "--port", "8000", "--dtype", "bfloat16",
        "--max-model-len", "64", "--max-num-seqs", "4", "--max-num-batched-tokens", "128",
        "--gpu-memory-utilization", "0.1", "--enforce-eager", "--max-logprobs", "-1",
        image=teacher_image, app=app, gpu="B200", cpu=8, memory=32768,
        volumes={"/evidence": volume}, secrets=[authentication], encrypted_ports=[8000], timeout=900,
        env={"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"},
    )
    print("XTOKEN_TEACHER_SANDBOX=" + sandbox.object_id, flush=True)
    try:
        url = sandbox.tunnels(timeout=50)[8000].url
        deadline = time.monotonic() + 180
        while True:
            try:
                request = urllib.request.Request(url + "/v1/models", headers={"Authorization": "Bearer " + key})
                with urllib.request.urlopen(request, timeout=5) as response:
                    models = json.load(response)
                if "xtoken-fixture" not in {model["id"] for model in models["data"]}:
                    raise RuntimeError("Unexpected test teacher model")
                break
            except (urllib.error.URLError, TimeoutError):
                if sandbox.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Isolated vLLM teacher did not become ready")
                time.sleep(3)
        request = urllib.request.Request(url + "/version", headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(request, timeout=5) as response:
            version = json.load(response)["version"]
        if version != "0.31.0":
            raise RuntimeError("Unexpected isolated teacher version: " + version)
        print("XTOKEN_TEACHER_READY=" + version, flush=True)
        function = http_student.with_options(secrets=[authentication], env={"XTOKEN_TEACHER_IMAGE": os.environ["XTOKEN_TEACHER_IMAGE"]})
        print(function.remote(url, modes))
    finally:
        sandbox.terminate()
        root = Path(__file__).resolve().parents[2] / "internal_only/xtoken-http"
        root.mkdir(parents=True, exist_ok=True)
        (root / "teacher-stdout.log").write_text(sandbox.stdout.read())
        (root / "teacher-stderr.log").write_text(sandbox.stderr.read())


@app.local_entrypoint()
def main(stage: str) -> None:
    if stage in {"gpu1", "gpu2", "off", "opd"}:
        result = compatibility.remote()
        if not result["compatible"]:
            raise RuntimeError("Native backend API preflight failed; no GPU allocated. See compatibility.json in the evidence Volume.")
    if stage == "probe":
        probe.remote()
    elif stage == "cpu":
        print(cpu_math.remote())
    elif stage == "compat":
        print(compatibility.remote())
    elif stage == "runtime-api":
        print(runtime_api.remote())
    elif stage == "receipts":
        print(receipts.remote())
    elif stage in {"gpu1", "math1"}:
        print(gpu_one.remote(stage == "math1"))
    elif stage in {"gpu2", "math2"}:
        print(gpu_two.remote(stage == "math2"))
    elif stage in {"off", "opd"}:
        print(end_to_end.remote("off" if stage == "off" else "on"))
    elif stage in {"http-off", "http-opd", "http-both"}:
        run_http_pipeline(["off", "on"] if stage == "http-both" else ["off" if stage == "http-off" else "on"])
    else:
        raise ValueError(stage)
