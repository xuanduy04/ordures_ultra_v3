"""Reuse A-on-11 assets and budgets in a new 200-step run, never resume it."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid

import yaml

from integrity import sha256


def rebase(value, old, new):
    if isinstance(value, dict):
        return {key: rebase(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase(item, old, new) for item in value]
    if isinstance(value, str) and (value == old or value.startswith(old + "/")):
        return new + value[len(old):]
    return value


def resolve_projection_reuse(baseline, spec, arm, load_tokenizer):
    """Reuse baseline matrix rows only after comparing the actual local vocabularies."""
    selected = spec["arms"][arm]
    if selected["tokenizer_mode"] != "cross" or "reuse_baseline_projection" not in selected:
        return spec, None
    if type(selected["reuse_baseline_projection"]) is not bool:
        raise ValueError("reuse_baseline_projection must be a boolean")
    if not selected["reuse_baseline_projection"]:
        return spec, None
    if len(baseline["teachers"]) != 1:
        raise ValueError("Projection reuse requires one baseline teacher")
    previous = baseline["teachers"][0]
    if selected["projection_path"] is not None or previous["projection_matrix_path"] is None:
        raise ValueError("Choose an explicit projection_path or reuse a non-null baseline projection")
    if selected["teacher_tokenizer"] is not None and selected["teacher_tokenizer"] != previous["tokenizer"]["name"]:
        raise ValueError("Baseline projection reuse requires the unchanged teacher tokenizer")
    tokenizer_config = deepcopy(baseline["policy"]["tokenizer"])
    if spec["student"]["model_path"] is not None:
        tokenizer_config["name"] = spec["student"]["model_path"]
    if spec["student"]["tokenizer_path"] is not None:
        tokenizer_config["name"] = spec["student"]["tokenizer_path"]
    before = load_tokenizer(baseline["policy"]["tokenizer"])
    after = load_tokenizer(tokenizer_config)
    vocab = after.get_vocab()
    if len(before) != len(after) or before.get_vocab() != vocab:
        raise ValueError("Student token-to-ID mapping changed; provide a matching projection_path")
    resolved = deepcopy(spec)
    resolved["arms"][arm]["projection_path"] = previous["projection_matrix_path"]
    receipt = {"student_tokenizer_before": baseline["policy"]["tokenizer"]["name"],
        "student_tokenizer_after": tokenizer_config["name"], "identical_token_to_id": True,
        "student_tokenizer_len": len(after),
        "student_vocab_sha256": hashlib.sha256(json.dumps(vocab, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "teacher_tokenizer": previous["tokenizer"]["name"],
        "projection_path": previous["projection_matrix_path"],
        "projection_sha256": sha256(previous["projection_matrix_path"])}
    return resolved, receipt


def pilot_config(baseline, spec, arm, old_run, run, old_source, source, ray_address):
    cfg = rebase(deepcopy(baseline), str(old_run), str(run))
    cfg = rebase(cfg, str(old_source), str(source))
    if len(cfg["teachers"]) != 1:
        raise ValueError("This plan requires one teacher per independent run")
    cfg["distillation"].update(spec["distillation"])
    cfg["policy"]["scheduler"] = deepcopy(spec["scheduler"])
    cfg["checkpointing"].update(spec["checkpointing"])
    cfg["checkpointing"]["enabled"] = True
    cfg["checkpointing"]["checkpoint_dir"] = str(run / "checkpoints")
    cfg["on_policy"]["nemo_gym"]["ray_address"] = ray_address
    cfg["logger"]["log_dir"] = str(run / "train-logs")
    cfg["on_policy"]["nemo_gym"]["global_config"]["nemo_gym_log_dir"] = str(run / "gym-logs")
    cfg["on_policy"]["nemo_gym"]["source_contract"] = str(source / "tools/xtoken_all/super-source-contract.json")
    if "variant" in spec:
        cfg["loss_fn"]["variant"] = deepcopy(spec["variant"])
        batch = spec["variant_batch"]
        cfg["policy"]["train_global_batch_size"] = batch["global_batch_size"]
        cfg["policy"]["train_micro_batch_size"] = batch["micro_batch_size"]
        cfg["distillation"]["num_prompts_per_step"] = batch["global_batch_size"]
        cfg["policy"]["dtensor_cfg"]["tensor_parallel_size"] = 1
        cfg["policy"]["dtensor_cfg"]["context_parallel_size"] = 1
        cfg["teachers"][0]["train_global_batch_size"] = batch["global_batch_size"]
        cfg["teachers"][0]["train_micro_batch_size"] = batch["micro_batch_size"]
        cfg["teachers"][0]["remote"]["score_mode"] = spec["variant"]["score_mode"]
        cfg["loss_fn"]["teacher_distribution_mode"] = "topk_tail_kl" if spec["variant"]["score_mode"] == "topk" else "full_logprobs"
        if spec["variant"]["score_mode"] == "full":
            cfg["loss_fn"]["vocab_topk"] = spec["arms"][arm]["teacher_vocab_size"]
    student = spec["student"]
    if student["model_path"] is not None:
        cfg["policy"]["model_name"] = student["model_path"]
        cfg["policy"]["tokenizer"]["name"] = student["model_path"]
    if student["tokenizer_path"] is not None:
        cfg["policy"]["tokenizer"]["name"] = student["tokenizer_path"]
    cfg["policy"]["tokenizer"]["chat_template_kwargs"] = deepcopy(student["chat_template_kwargs"])
    global_config = cfg["on_policy"]["nemo_gym"]["global_config"]
    for model in ("policy_model", "policy_model_reasoning_off"):
        global_config[model]["responses_api_models"]["vllm_model"]["chat_template_kwargs"] = deepcopy(student["chat_template_kwargs"])
    if spec["arms"][arm]["tokenizer_mode"] == "same":
        teacher = cfg["teachers"][0]
        teacher["tokenizer"] = deepcopy(cfg["policy"]["tokenizer"])
        tokenizer_path = spec["arms"][arm]["teacher_tokenizer"]
        if tokenizer_path is None:
            tokenizer_path = cfg["policy"]["tokenizer"]["name"]
        teacher["tokenizer"]["name"] = tokenizer_path
        teacher["model_name"] = tokenizer_path
        teacher["projection_matrix_path"] = None
        teacher["remote"].update(url=spec["arms"][arm]["teacher_url"],
            model=spec["arms"][arm]["teacher_model"], model_vocab_size=spec["arms"][arm]["teacher_vocab_size"],
            api_key_env=None)
    else:
        teacher = cfg["teachers"][0]
        arm_spec = spec["arms"][arm]
        tokenizer_changed = cfg["policy"]["tokenizer"]["name"] != baseline["policy"]["tokenizer"]["name"]
        tokenizer_changed |= arm_spec["teacher_tokenizer"] is not None and arm_spec["teacher_tokenizer"] != teacher["tokenizer"]["name"]
        if tokenizer_changed and arm_spec["projection_path"] is None:
            raise ValueError("Changed cross-tokenizer assets require an explicit matching projection_path")
        teacher["remote"]["url"] = arm_spec["teacher_url"]
        teacher["remote"]["model"] = arm_spec["teacher_model"]
        teacher["remote"]["model_vocab_size"] = arm_spec["teacher_vocab_size"]
        if arm_spec["teacher_tokenizer"] is not None:
            teacher["tokenizer"]["name"] = arm_spec["teacher_tokenizer"]
            teacher["model_name"] = arm_spec["teacher_tokenizer"]
        if arm_spec["projection_path"] is not None:
            teacher["projection_matrix_path"] = arm_spec["projection_path"]
    return cfg


def run_name(prefix, mode):
    if mode == "unique":
        return prefix + "-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    if mode == "fixed":
        return prefix
    raise ValueError("run_id_mode must be unique or fixed")


def allocation_command(source, spec, arm, address, output):
    indices = spec["runtime"]["gpu_indices"]
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    mapping = {int(line.split(",")[0]): line.split(",")[1].strip() for line in raw.splitlines()}
    observed = [mapping[index] for index in indices]
    expected = spec["arms"][arm]["gpu_uuids"]
    if expected is not None and expected != observed:
        raise ValueError("Selected GPU UUIDs differ from this arm's explicit hardware allocation")
    command = [sys.executable, "-B", str(source / "tools/xtoken_all/check_allocation.py"),
        "--node-label", socket.gethostname(), "--gpu-indices", *map(str, indices),
        "--expected-gpu-uuids", *observed, "--ray-address", address, "--auto-local", "--output", str(output),
        "--ray-executable", spec["runtime"]["ray_executable"], "--timeout-s", str(spec["runtime"]["ray_timeout_s"])]
    if spec["runtime"]["ray_start_if_missing"]:
        command.append("--start-if-missing")
    return command, observed


def setup_runtime(source, root, spec, arm, address):
    import fcntl

    runtime = spec["runtime"]
    os.environ.update(XTOKEN_ALL_ROOT=str(root), NRL_PYTHON=os.path.abspath(sys.executable),
        NEMO_RL_VENV_DIR=runtime["worker_venvs"], NEMO_GYM_VENV_DIR=runtime["gym_venvs"],
        CUDA_VISIBLE_DEVICES=",".join(map(str, runtime["gpu_indices"])))
    temporary = root / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    allocation = temporary / ("setup-allocation-" + uuid.uuid4().hex + ".json")
    command, uuids = allocation_command(source, spec, arm, address, allocation)
    locks = []
    try:
        for gpu_uuid in sorted(uuids):
            lock_name = hashlib.sha256(gpu_uuid.encode()).hexdigest()[:16]
            lock = (temporary / ("xtoken-gpu-" + lock_name + ".lock")).open("a")
            locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run(command, check=True)
    except BlockingIOError:
        for lock in locks:
            lock.close()
        raise RuntimeError("Another X-Token launcher holds these GPUs") from None
    except BaseException:
        for lock in locks:
            lock.close()
        raise
    return locks


def prepare_dataset(root, spec):
    """Optionally call the existing converter and verify its cached receipt."""
    source = Path(spec["data_path"]).resolve()
    source_hash = sha256(source)
    if spec["data_sha256"] is not None and source_hash != spec["data_sha256"]:
        raise ValueError("Dataset SHA256 differs from the selected YAML")
    conversion = spec["dataset_conversion"]
    if conversion is None:
        return source
    recipe_hash = hashlib.sha256(json.dumps([conversion, spec["expected_rows"]], sort_keys=True).encode()).hexdigest()[:12]
    output = root / "data" / ("converted-" + source_hash[:12] + "-" + recipe_hash)
    if not output.exists():
        options = conversion["options"]
        if conversion["kind"] == "math":
            if type(spec["expected_rows"]) is not int or spec["expected_rows"] <= 0:
                raise ValueError("Math conversion requires a positive expected_rows in the YAML")
            from convert_general_qa import convert
            convert(source, output, spec["expected_rows"], options["prompt_field"], options["answer_field"], options["dataset"])
        elif conversion["kind"] == "math_code":
            from convert_math_code import convert
            settings = {**options, "input": str(source), "output_dir": str(output)}
            convert(settings)
        else:
            raise ValueError("Use the existing math or math_code converter")
    receipt = json.loads((output / "conversion.json").read_text())
    if receipt["success"] is not True or receipt["source_sha256"] != source_hash or (output / "FAILED").exists():
        raise ValueError("Converted dataset has no valid successful provenance")
    converted = output / "train.jsonl"
    if sha256(converted) != receipt["output_sha256"]:
        raise ValueError("Converted dataset changed after conversion")
    return converted


def finish_pipeline(source, root, run, spec, arm, address, check_only):
    subprocess.run([sys.executable, "-B", str(source / "tools/xtoken_all/preflight.py"), "--run-dir", str(run)], check=True)
    probe = [sys.executable, "-B", str(source / "tools/xtoken_all/probe_teacher.py"), "--config", str(run / "effective.yaml")]
    evidence = spec["arms"][arm]["server_mode_evidence"]
    if evidence is not None:
        probe.extend(["--server-mode-evidence", evidence])
    subprocess.run(probe, check=True)
    allocation = run / "allocation.json"
    command, _ = allocation_command(source, spec, arm, address, allocation)
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, "-B", str(source / "tools/xtoken_all/ray_stream_probe.py"),
        "--run-dir", str(run), "--allocation-receipt", str(allocation)], check=True)
    if check_only:
        print("XTOKEN_READY=" + json.dumps({"run_dir": str(run), "training_started": False}), flush=True)
        return
    os.environ.update(XTOKEN_RUN_DIR=str(run), XTOKEN_ALLOCATION_RECEIPT=str(allocation))
    subprocess.run(["bash", str(source / "tools/xtoken_all/launch.sh")], check=True)
    subprocess.run([sys.executable, "-B", str(source / "tools/xtoken_all/check_pilot.py"), "--run-dir", str(run)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--ray-address")
    parser.add_argument("--run-id")
    parser.add_argument("--gym-actor-python", type=Path)
    parser.add_argument("--gym-source", type=Path)
    parser.add_argument("--gym-venvs", type=Path)
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-url")
    parser.add_argument("--teacher-tokenizer", type=Path)
    parser.add_argument("--teacher-api-key-env")
    parser.add_argument("--gpu-indices", type=int, nargs="+")
    args = parser.parse_args()
    if args.check_only and not args.run:
        parser.error("--check-only requires --run")
    source = Path(__file__).resolve().parents[2]
    spec_path = args.config if args.config is not None else source / "tools/xtoken_all/pilot_200.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    if args.arm not in spec["arms"]:
        parser.error("Unknown arm in the selected YAML")
    if args.gpu_indices is not None:
        old_indices = spec["runtime"]["gpu_indices"]
        if not args.gpu_indices or len(set(args.gpu_indices)) != len(args.gpu_indices) or not set(args.gpu_indices).issubset(old_indices):
            parser.error("GPU subset must be unique indices from this recipe's approved allocation")
        expected = spec["arms"][args.arm]["gpu_uuids"]
        if expected is not None:
            spec["arms"][args.arm]["gpu_uuids"] = [expected[old_indices.index(index)] for index in args.gpu_indices]
        spec["runtime"]["gpu_indices"] = args.gpu_indices
    if "environment" in spec["runtime"]:
        runtime_environment = spec["runtime"]["environment"]
        if set(runtime_environment) - {"VLLM_USE_DEEP_GEMM", "VLLM_MOE_USE_DEEP_GEMM"}:
            raise ValueError("Only reviewed optional DeepGEMM switches are supported in runtime.environment")
        if any(value not in ("0", "1") for value in runtime_environment.values()):
            raise ValueError("Runtime environment switches must be quoted strings 0 or 1")
        os.environ.update(runtime_environment)
        print("XTOKEN_VARIANT_RUNTIME=" + json.dumps(runtime_environment), flush=True)
    root = Path(spec["root"] if args.run or "XTOKEN_ALL_ROOT" not in os.environ else os.environ["XTOKEN_ALL_ROOT"]).resolve()
    source.relative_to(root)
    os.environ["XTOKEN_ALL_ROOT"] = str(root)
    address = args.ray_address
    if args.run and address is None:
        host = spec["runtime"]["ray_host"]
        if host is None:
            host = subprocess.check_output(["hostname", "-I"], text=True).split()[0]
        address = host + ":" + str(spec["runtime"]["ray_port"])
    if address is None:
        parser.error("--ray-address is required unless --run performs local setup")
    locks = setup_runtime(source, root, spec, args.arm, address) if args.run else []
    run_id = args.run_id if args.run_id is not None else run_name(spec["arms"][args.arm]["run_id"], spec["run_id_mode"])
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Run ID must be one component")
    old_run, run = root / "runs" / spec["baseline_run"], root / "runs" / run_id
    if run.exists():
        raise FileExistsError("Use a new run ID; no checkpoint/dataloader resume")
    baseline_path = old_run / "effective.yaml"
    baseline = yaml.safe_load(baseline_path.read_text())
    metadata = json.loads((old_run / "provenance.json").read_text())
    projection_reuse = None
    if spec["arms"][args.arm]["tokenizer_mode"] == "cross" and "reuse_baseline_projection" in spec["arms"][args.arm]:
        from nemo_rl.algorithms.utils import get_tokenizer

        spec, projection_reuse = resolve_projection_reuse(baseline, spec, args.arm, get_tokenizer)
    data = prepare_dataset(root, spec)
    with data.open(encoding="utf-8") as stream:
        rows = sum(1 for line in stream if line.strip())
    if spec["expected_rows"] is not None and rows != spec["expected_rows"]:
        raise ValueError("Dataset row count differs from the selected YAML")
    cfg = pilot_config(baseline, spec, args.arm, old_run, run,
                       Path(metadata["source_root"]), source, address)
    cfg["data"]["train"]["data_path"] = str(data.resolve())
    if args.run:
        cfg["cluster"]["gpus_per_node"] = len(spec["runtime"]["gpu_indices"])
        for name in ("policy_model", "policy_model_reasoning_off"):
            cfg["on_policy"]["nemo_gym"]["global_config"][name]["responses_api_models"]["vllm_model"]["num_workers"] = len(spec["runtime"]["gpu_indices"])
        for key, runtime_key in (("actor_python", "gym_actor_python"), ("source_root", "gym_source"), ("venv_dir", "gym_venvs")):
            cfg["on_policy"]["nemo_gym"][key] = spec["runtime"][runtime_key]
    if args.teacher_model is not None:
        cfg["teachers"][0]["remote"]["model"] = args.teacher_model
    if args.teacher_url is not None:
        cfg["teachers"][0]["remote"]["url"] = args.teacher_url
    if args.teacher_tokenizer is not None:
        cfg["teachers"][0]["tokenizer"]["name"] = str(args.teacher_tokenizer.resolve())
        cfg["teachers"][0]["model_name"] = str(args.teacher_tokenizer.resolve())
    if projection_reuse is not None and cfg["teachers"][0]["tokenizer"] != baseline["teachers"][0]["tokenizer"]:
        raise ValueError("Cannot override the teacher tokenizer after validating baseline projection reuse")
    gym = cfg["on_policy"]["nemo_gym"]
    for name, override in (("actor_python", args.gym_actor_python), ("source_root", args.gym_source), ("venv_dir", args.gym_venvs)):
        if override is not None:
            gym[name] = os.path.abspath(override)
    plans = root / "plans"
    plans.mkdir(exist_ok=True)
    plan_path = plans / (run_id + "-input.yaml")
    if plan_path.exists():
        if yaml.safe_load(plan_path.read_text()) != cfg:
            raise ValueError("A different input plan already exists; use a new run ID")
    else:
        with plan_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False, allow_unicode=True)
    teacher = cfg["teachers"][0]
    command = [sys.executable, "-B", str(source / "tools/xtoken_all/render.py"),
        "--config", str(plan_path), "--run-id", run_id, "--student", cfg["policy"]["model_name"],
        "--teacher-tokenizer", teacher["tokenizer"]["name"], "--data", str(data),
        "--teacher-url", teacher["remote"]["url"], "--teacher-model", teacher["remote"]["model"],
        "--gym-actor-python", gym["actor_python"], "--gym-source", gym["source_root"],
        "--gym-venvs", gym["venv_dir"], "--ray-address", address,
        "--student-tokenizer", cfg["policy"]["tokenizer"]["name"]]
    if spec["arms"][args.arm]["tokenizer_mode"] == "same":
        command.extend(["--same-tokenizer", "--teacher-vocab-size", str(teacher["remote"]["model_vocab_size"])])
    else:
        command.extend(["--projection", teacher["projection_matrix_path"]])
    key = args.teacher_api_key_env if args.teacher_api_key_env is not None else teacher["remote"]["api_key_env"]
    if key is not None:
        command.extend(["--teacher-api-key-env", key])
    subprocess.run(command, check=True)
    receipt = {"arm": args.arm, "baseline_run": spec["baseline_run"], "baseline_used_for": "assets/data/budgets only",
        "baseline_config_sha256": sha256(baseline_path), "pilot_spec_sha256": sha256(spec_path),
        "data_sha256": sha256(data), "rows": rows, "run_dir": str(run), "fresh_initialization": True,
        "steps": spec["distillation"]["max_num_steps"], "warmup": spec["scheduler"][0]["kwargs"]["total_iters"],
        "save_period": spec["checkpointing"]["save_period"], "ray_address": address,
        "keep_top_k": spec["checkpointing"]["keep_top_k"],
        "teacher_url": teacher["remote"]["url"], "teacher_model": teacher["remote"]["model"],
        "teacher_tokenizer": teacher["tokenizer"]["name"],
        "student_model": cfg["policy"]["model_name"], "student_tokenizer": cfg["policy"]["tokenizer"]["name"],
        "student_chat_template_kwargs": cfg["policy"]["tokenizer"]["chat_template_kwargs"],
        "projection_reuse": projection_reuse,
        "tokenizer_mode": spec["arms"][args.arm]["tokenizer_mode"],
        "baseline_schedule_preserved": False, "baseline_run_modified": False}
    (run / "pilot-plan.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if args.run:
        (run / "launch-input.yaml").write_text(spec_path.read_text())
        finish_pipeline(source, root, run, spec, args.arm, address, args.check_only)
    for lock in locks:
        lock.close()


if __name__ == "__main__":
    main()
