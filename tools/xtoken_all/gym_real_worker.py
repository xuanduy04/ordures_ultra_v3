"""Actual remote staging and Gym canary work; no Modal SDK in worker roles."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import zipfile

import yaml


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def payload_hash():
    value = hashlib.sha256()
    for root in ("nemo_rl", "tools/xtoken_all", "examples", "custom_utils"):
        for path in sorted((Path("/work/source") / root).rglob("*.py")):
            value.update(path.relative_to("/work/source").as_posix().encode() + b"\0" + path.read_bytes())
    return value.hexdigest()


def staging(config, arm):
    """Download public, revision-pinned assets only on the remote scratch volume."""
    from huggingface_hub import snapshot_download

    cache = Path("/cache")
    cache.mkdir(exist_ok=True)
    (cache / ("imports-" + arm + ".json")).write_text(json.dumps({"success": False, "state": "preparation_pending"}))
    site = cache / "supplement"
    site.mkdir(exist_ok=True)
    packages = config["supplement"]["packages"]
    stamp = site / "packages.json"
    if not stamp.is_file() or json.loads(stamp.read_text()) != packages:
        subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--target", str(site),
                        "--no-deps", *packages], check=True, timeout=300)
        stamp.write_text(json.dumps(packages))
    archive = cache / "assets.zip"
    if not archive.is_file() or digest(archive) != config["assets"]["sha256"]:
        temporary = archive.with_suffix(".partial")
        urllib.request.urlretrieve(config["assets"]["url"], temporary)
        if digest(temporary) != config["assets"]["sha256"]:
            raise ValueError("Public assets archive hash differs from the pinned manifest")
        temporary.replace(archive)
    assets = cache / "assets"
    if not assets.is_dir():
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                target = (assets / item.filename).resolve()
                target.relative_to(assets.resolve())
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Symlinks are not accepted in the assets archive")
            source.extractall(assets)
    models = {}
    for name, spec in (("student", config["student"]), ("teacher", config["teachers"][arm])):
        print("GYM_STAGE=" + json.dumps({"phase": "download", "role": name, "repo": spec["repo"]}), flush=True)
        target = cache / "models" / spec["repo"].split("/")[-1]
        snapshot_download(repo_id=spec["repo"], revision=spec["revision"], token=False,
                          local_dir=target, max_workers=8,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "LICENSE", "tokenizer*"])
        models[name] = {"path": str(target), **spec}
    actor = Path("/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym")
    services = ("responses_api_models/vllm_model", "responses_api_agents/simple_agent", "resources_servers/general_qa")
    service_paths = []
    for name in services:
        prefix = cache / "gym_venvs" / name / ".venv"
        if not (prefix / "bin/python").is_file():
            venv.EnvBuilder(with_pip=False).create(prefix)
        target = next((prefix / "lib").glob("python*/site-packages")) / "xtoken-runtime.pth"
        actor_site = next((actor / "lib").glob("python*/site-packages"))
        target.write_text(str(site) + "\n" + str(actor_site) + "\n")
        service_paths.append(str(prefix))
    report = {"models": models, "assets": [str(p.relative_to(assets)) for p in assets.rglob("*") if p.is_file()],
              "services": service_paths, "supplement_packages": packages}
    (cache / ("stage-" + arm + ".json")).write_text(json.dumps(report, indent=2))
    print("GYM_STAGED=" + json.dumps({"models": models, "asset_files": len(report["assets"]),
                                      "services": service_paths, "supplement_packages": packages}), flush=True)


def imports(config, arm):
    """Use the real service probe, importing apps rather than only finding specs."""
    Path("/cache/imports-" + arm + ".json").write_text(json.dumps({"success": False, "state": "imports_pending"}))
    roots = [Path("/opt/nemo_rl_venv"), *Path("/opt/ray_venvs").glob("*")]
    for root in roots:
        if "dtensor_policy_worker_v2" not in str(root) and "vllm_worker_async" not in str(root) and "NemoGym" not in str(root) and root.name != "nemo_rl_venv":
            continue
        subprocess.run([str(root / "bin/python"), "-B", "-c",
                        "import ray,sys; print('GYM_ROLE_RAY='+str((sys.executable,ray.__version__)))"], check=True, timeout=30)
    for service, name in (("model", "responses_api_models/vllm_model"), ("agent", "responses_api_agents/simple_agent"),
                          ("qa", "resources_servers/general_qa")):
        executable = Path("/cache/gym_venvs") / name / ".venv/bin/python"
        subprocess.run([str(executable), "-B", "/work/source/tools/xtoken_all/gym_service_probe.py",
                        "--service", service, "--gym-root", "/work/gym"], check=True, timeout=60)
    actor = "/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python"
    actor_probe = "import faulthandler,runpy,sys; faulthandler.dump_traceback_later(45,repeat=True); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    subprocess.run([actor, "-B", "-c", actor_probe, "/work/source/tools/xtoken_all/runtime_probe.py", "--role", "gym_actor", "--gym-root", "/work/gym"],
                   check=True, timeout=config["canary"]["actor_import_timeout_s"])
    generation = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"
    subprocess.run([generation, "-B", "-c", "import importlib.metadata as m,json; from pathlib import Path; d=m.distribution('vllm'); print('GYM_VLLM_API_FILES='+json.dumps({p:Path(d.locate_file('vllm/'+p)).is_file() for p in ['entrypoints/openai/chat_completion/protocol.py','entrypoints/openai/chat_completion/serving.py','entrypoints/openai/models/serving.py','entrypoints/serve/tokenize/serving.py']}))"],
                   check=True, timeout=20)
    from transformers import AutoTokenizer
    from tools.xtoken_all.convert_general_qa import convert
    from nemo_rl.algorithms.x_token.gym_compat import verify_gym_source

    source_receipt = verify_gym_source("/work/gym", "/work/source/tools/xtoken_all/super-source-contract.json")
    stage = json.loads(Path("/cache/stage-" + arm + ".json").read_text())
    student = AutoTokenizer.from_pretrained(stage["models"]["student"]["path"], local_files_only=True)
    previous = list(Path("/cache/assets").rglob("tokenizers/qwen3-1p7b"))
    if len(previous) != 1:
        raise ValueError("Missing unique reviewed old student tokenizer for projection comparison")
    old = AutoTokenizer.from_pretrained(previous[0], local_files_only=True)
    if student.get_vocab() != old.get_vocab() or len(student) != len(old):
        raise ValueError("Projection student vocabulary changed; do not reuse the asset")
    if arm == "same":
        teacher = AutoTokenizer.from_pretrained(stage["models"]["teacher"]["path"], local_files_only=True)
        if teacher.get_vocab() != student.get_vocab():
            raise ValueError("Same-tokenizer teacher has a different actual token-to-ID mapping")
    data = list(Path("/cache/assets").rglob("on-policy-smoke-8.jsonl"))
    if len(data) != 1:
        raise ValueError("Missing unique reviewed 8-row DAPO smoke pool")
    rows = [json.loads(line) for line in data[0].read_text().splitlines()]
    if len(rows) != 8:
        raise ValueError("Expected exactly eight fixed smoke examples")
    for row in rows:
        ids = student.apply_chat_template([{"role": "user", "content": row["text"]}], tokenize=True,
                                          enable_thinking=False, add_generation_prompt=True, return_dict=False)
        if len(ids) > config["canary"]["max_prompt_tokens"]:
            raise ValueError("Actual Qwen4B nonthinking smoke prompt exceeds its configured cap")
    output = Path("/cache/data")
    conversion = convert(data[0], output, 8, "text", "ground_truth", "dapo") if not output.exists() else json.loads((output / "conversion.json").read_text())
    receipt = {"success": True, "supplement_packages": config["supplement"]["packages"],
               "projection_vocabulary_identical": True, "student_tokenizer_len": len(student),
               "source": source_receipt, "data": conversion}
    Path("/cache/imports-" + arm + ".json").write_text(json.dumps(receipt, indent=2))
    print("GYM_IMPORTS=" + json.dumps(receipt), flush=True)


def sanitized(text, key):
    text = text.replace(key, "[teacher key redacted]") if key else text
    return "\n".join(line if not any(value in line for value in ("env_vars", "MODAL_IDENTITY_TOKEN", "TOKEN_SECRET"))
                     else "[environment line omitted]" for line in text.splitlines()) + "\n"


def run_process(command, env, timeout, log_path, key):
    """Stream sanitized evidence; timeout terminates this owned process group."""
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, cwd="/work/source", stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        expired = threading.Event()

        def stop():
            expired.set()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(timeout, stop)
        timer.start()
        try:
            for line in process.stdout:
                clean = sanitized(line, key)
                log.write(clean)
                log.flush()
                if any(marker in clean for marker in ("XTOKEN_TRAIN_STEP=", "XTOKEN_TEACHER_SCORE=", "XTOKEN_ROLLOUT=",
                                                      "XTOKEN_REFIT_REQUEST=", "NRL_REFIT_BUFFER=", "XTK_TEACHER_", "Error", "Exception", "Traceback")):
                    print(clean, end="", flush=True)
            code = process.wait()
        finally:
            timer.cancel()
        return 124 if expired.is_set() else code


def render(config, arm, mode, run, teacher_url):
    """Reuse the production inherited Gym recipe and explicit algorithm adapters."""
    from omegaconf import OmegaConf
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from nemo_rl.algorithms.x_token.gym_compat import bind_teacher_endpoint

    stage = json.loads(Path("/cache/stage-" + arm + ".json").read_text())
    student, teacher = stage["models"]["student"]["path"], stage["models"]["teacher"]["path"]
    knobs = config["canary"]
    os.environ.update(XTOKEN_STUDENT_MODEL=student, XTOKEN_SOURCE_ROOT="/work/source", XTOKEN_GYM_SOURCE="/work/gym",
                      XTOKEN_GYM_ACTOR_PYTHON="/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python",
                      XTOKEN_GYM_VENV_DIR="/cache/gym_venvs", XTOKEN_RAY_ADDRESS="127.0.0.1:6379",
                      XTOKEN_RUN_DIR=str(run), XTOKEN_TRAIN_DATA="/cache/data/train.jsonl")
    register_omegaconf_resolvers()
    cfg = load_config("/work/source/examples/configs/xtoken_gym_qwen3_gemma4.yaml")
    cfg.cluster.gpus_per_node = knobs["gpu_count"]
    cfg.policy.train_global_batch_size = knobs["global_batch_size"]
    cfg.policy.train_micro_batch_size = knobs["micro_batch_size"]
    cfg.policy.max_total_sequence_length = knobs["context"]
    cfg.policy.dtensor_cfg.fsdp_mode = "hf_sharded" if knobs["gpu_count"] == 3 else "sharded"
    if knobs["gpu_count"] == 3:
        cfg.policy.dtensor_cfg.automodel_kwargs = {"force_hf": True, "use_liger_kernel": False}
    if knobs["global_batch_size"] % knobs["gpu_count"]:
        raise ValueError("Canary global batch must divide across the physical student GPU allocation")
    cfg.policy.generation.max_new_tokens = knobs["response_tokens"]
    cfg.policy.generation.vllm_cfg.max_model_len = knobs["context"]
    cfg.policy.generation.vllm_cfg.gpu_memory_utilization = 0.2
    cfg.policy.scheduler = {"name": "torch.optim.lr_scheduler.ConstantLR", "kwargs": {"factor": 1.0, "total_iters": knobs["max_num_steps"]}}
    cfg.distillation.num_prompts_per_step = knobs["global_batch_size"]
    cfg.distillation.max_num_steps = knobs["max_num_steps"]
    cfg.distillation.val_at_start = False
    cfg.distillation.val_at_end = False
    cfg.distillation.val_period = 0
    cfg.on_policy.max_prompt_tokens = knobs["max_prompt_tokens"]
    cfg.on_policy.teacher_max_total_sequence_length = knobs["teacher_context"]
    cfg.on_policy.refit_buffer_size_gb = knobs["refit_buffer_size_gb"]
    cfg.on_policy.nemo_gym.global_config.policy_model.responses_api_models.vllm_model.num_workers = knobs["gpu_count"]
    cfg.on_policy.nemo_gym.global_config.python_version = "3.13"
    cfg.logger.wandb_enabled = False
    cfg.logger.tensorboard_enabled = False
    cfg.logger.mlflow_enabled = False
    cfg.checkpointing.save_period = knobs["checkpoint_period"]
    cfg.checkpointing.keep_top_k = knobs["keep_checkpoints"]
    cfg.teachers[0].model_name = teacher
    cfg.teachers[0].tokenizer.name = teacher
    cfg.teachers[0].remote.url = teacher_url
    cfg.teachers[0].remote.model = config["teachers"][arm]["alias"]
    cfg.teachers[0].remote.model_vocab_size = config["teachers"][arm]["vocab"]
    cfg.teachers[0].remote.api_key_env = "XTOKEN_TEACHER_API_KEY"
    cfg.teachers[0].max_total_sequence_length = knobs["teacher_context"]
    cfg.teachers[0].train_micro_batch_size = knobs["micro_batch_size"]
    if arm == "same":
        cfg.teachers[0].tokenizer = deepcopy(cfg.policy.tokenizer)
        cfg.teachers[0].tokenizer.name = teacher
        cfg.teachers[0].projection_matrix_path = None
    else:
        projections = list(Path("/cache/assets").rglob("*.pt"))
        if len(projections) != 1:
            raise ValueError("Expected exactly one reviewed student-to-Gemma projection")
        cfg.teachers[0].projection_matrix_path = str(projections[0])
    if mode != "topk_tail_kl":
        cfg.loss_fn.variant = {"mode": mode, **config["variant"]}
        cfg.loss_fn.temperature = 1.0
        cfg.loss_fn.ce_loss_scale = 0.0
        cfg.loss_fn.dynamic_loss_scaling = False
    cfg = OmegaConf.to_container(cfg, resolve=True)
    bind_teacher_endpoint(cfg)
    (run / "effective.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def canaries(config, arm, teacher_url):
    """Two real updates per fresh mode, using unchanged training entrypoints."""
    import torch
    from tools.xtoken_all.check_result import assess

    key = os.environ["XTOKEN_TEACHER_API_KEY"]
    receipt = json.loads(Path("/cache/imports-" + arm + ".json").read_text())
    if "success" not in receipt or receipt["success"] is not True or receipt["supplement_packages"] != config["supplement"]["packages"]:
        raise ValueError("Missing, failed or stale runtime preparation receipt")
    if receipt["projection_vocabulary_identical"] is not True or receipt["data"]["output_rows"] != 8:
        raise ValueError("Real tokenizer/data/Gym import preparation has not passed")
    env = dict(os.environ, NEMO_RL_VENV_DIR="/opt/ray_venvs", UV_OFFLINE="1", HF_HUB_OFFLINE="1",
               RAY_ENABLE_UV_RUN_RUNTIME_ENV="0", RAY_DEDUP_LOGS="0", WANDB_MODE="disabled",
               RAY_worker_register_timeout_seconds="120")
    count = config["canary"]["gpu_count"]
    if torch.cuda.device_count() != count:
        raise ValueError("Actual CUDA allocation differs from canary GPU topology")
    hardware = {"count": count, "gpu_names": [torch.cuda.get_device_name(i) for i in range(count)],
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "source_payload_sha256": payload_hash()}
    print("GYM_HARDWARE=" + json.dumps(hardware), flush=True)
    reports = []
    persistent = Path("/cache/evidence") / arm
    persistent.mkdir(parents=True, exist_ok=True)
    suite_deadline = time.monotonic() + 1370
    for mode in config["modes"]:
        remaining = suite_deadline - time.monotonic()
        if remaining < 90:
            break
        run = Path("/tmp/xtoken-real") / arm / mode
        run.mkdir(parents=True, exist_ok=False)
        render(config, arm, mode, run, teacher_url)
        if not reports:
            evidence = {"teacher_url": teacher_url, "served_model": config["teachers"][arm]["alias"],
                        "logprobs_mode": "raw_logprobs", "verification_source": "operator_checked_server_args",
                        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "evidence_description": "Owned isolated Modal teacher launched with explicit --logprobs-mode raw_logprobs; unchanged server API"}
            (run / "server-mode.json").write_text(json.dumps(evidence))
            probe_code = run_process([sys.executable, "-B", "tools/xtoken_all/probe_teacher.py", "--config", str(run / "effective.yaml"),
                                      "--server-mode-evidence", str(run / "server-mode.json")], env, config["canary"]["teacher_probe_timeout_s"], run / "teacher-probe.log", key)
            if probe_code:
                raise RuntimeError("Real teacher scoring/tokenizer parity probe failed; see teacher-probe.log")
        subprocess.run([sys.executable, "-m", "ray.scripts.scripts", "start", "--head", "--node-ip-address=127.0.0.1",
                        "--port=6379", "--num-cpus=12", "--num-gpus=" + str(count), "--disable-usage-stats",
                        "--include-dashboard=false"], env=env, check=True, capture_output=True, text=True, timeout=90)
        print("GYM_CANARY=" + json.dumps({"arm": arm, "mode": mode, "state": "starting"}), flush=True)
        try:
            code = run_process([sys.executable, "-B", "examples/run_xtoken_on_policy_distillation.py", "--config", str(run / "effective.yaml")],
                               env, min(config["canary"]["mode_timeout_s"], max(1, suite_deadline - time.monotonic() - 65)), run / "driver.log", key)
            (run / "process-exit.json").write_text(json.dumps({"driver_exit_code": code, "tee_exit_code": 0}))
            report = assess(run, "on")
            report.update(mode=mode, arm=arm, hardware=hardware)
            (run / "result.json").write_text(json.dumps(report, indent=2))
        finally:
            subprocess.run([sys.executable, "-m", "ray.scripts.scripts", "stop", "--force"], env=env,
                           capture_output=True, text=True, timeout=60)
        reports.append(report)
        target = persistent / mode
        target.mkdir(exist_ok=True)
        for path in run.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".yaml", ".log") and "checkpoints" not in path.parts:
                relative = path.relative_to(run)
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(sanitized(path.read_text(errors="replace"), key))
        print("GYM_CANARY_RESULT=" + json.dumps(report), flush=True)
        # Only this temporary run's bulk tensors are removed after file-closure
        # evidence has been retained. No user checkpoint or volume is targeted.
        resolved = run.resolve()
        resolved.relative_to(Path("/tmp/xtoken-real").resolve())
        shutil.rmtree(resolved)
        if mode == "topk_tail_kl" and not report["success"]:
            break
    result = {"arm": arm, "reports": reports, "requested_modes": config["modes"],
              "success": len(reports) == len(config["modes"]) and all(item["success"] for item in reports)}
    (persistent / "summary.json").write_text(json.dumps(result, indent=2))
    print("GYM_SUITE_RESULT=" + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("stage", "imports", "canary"), required=True)
    parser.add_argument("--arm", choices=("cross", "same"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path("/work/source/tools/xtoken_all/gym_real.yaml").read_text())
    if args.phase == "stage":
        staging(config, args.arm)
        imports(config, args.arm)
    elif args.phase == "imports":
        imports(config, args.arm)
    else:
        canaries(config, args.arm, os.environ["XTOKEN_TEACHER_URL"])


if __name__ == "__main__":
    main()
