"""Reuse A-on-11 assets and budgets in a new 200-step run, never resume it."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

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
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("same", "cross", "same235"), required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--gym-actor-python", type=Path)
    parser.add_argument("--gym-source", type=Path)
    parser.add_argument("--gym-venvs", type=Path)
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-url")
    parser.add_argument("--teacher-tokenizer", type=Path)
    parser.add_argument("--teacher-api-key-env")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    root = Path(os.environ["XTOKEN_ALL_ROOT"]).resolve()
    source.relative_to(root)
    spec_path = source / "tools/xtoken_all/pilot_200.yaml"
    spec = yaml.safe_load(spec_path.read_text())
    expected_node = spec["arms"][args.arm]["node"]
    if socket.gethostname().split(".")[0] != expected_node:
        raise ValueError("Run this arm on " + expected_node + "; never attach to another node's Ray head")
    run_id = args.run_id if args.run_id is not None else spec["arms"][args.arm]["run_id"]
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("Run ID must be one component")
    old_run, run = root / "runs" / spec["baseline_run"], root / "runs" / run_id
    if run.exists():
        raise FileExistsError("Use a new run ID; no checkpoint/dataloader resume")
    baseline_path = old_run / "effective.yaml"
    baseline = yaml.safe_load(baseline_path.read_text())
    metadata = json.loads((old_run / "provenance.json").read_text())
    data = Path(spec["data_path"])
    if Path(baseline["data"]["train"]["data_path"]).resolve() != data.resolve():
        raise ValueError("A-on-11 no longer names the declared full math pool")
    if sha256(data) != metadata["data_sha256"]:
        raise ValueError("The active pool differs from A-on-11 provenance")
    with data.open(encoding="utf-8") as stream:
        rows = sum(1 for line in stream if line.strip())
    if rows != spec["expected_rows"]:
        raise ValueError("Wrong full math pool row count")
    cfg = pilot_config(baseline, spec, args.arm, old_run, run,
                       Path(metadata["source_root"]), source, args.ray_address)
    if args.teacher_model is not None:
        cfg["teachers"][0]["remote"]["model"] = args.teacher_model
    if args.teacher_url is not None:
        cfg["teachers"][0]["remote"]["url"] = args.teacher_url
    if args.teacher_tokenizer is not None:
        cfg["teachers"][0]["tokenizer"]["name"] = str(args.teacher_tokenizer.resolve())
        cfg["teachers"][0]["model_name"] = str(args.teacher_tokenizer.resolve())
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
        "--gym-venvs", gym["venv_dir"], "--ray-address", args.ray_address]
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
        "save_period": spec["checkpointing"]["save_period"], "ray_address": args.ray_address,
        "keep_top_k": spec["checkpointing"]["keep_top_k"],
        "teacher_url": teacher["remote"]["url"], "teacher_model": teacher["remote"]["model"],
        "teacher_tokenizer": teacher["tokenizer"]["name"],
        "tokenizer_mode": spec["arms"][args.arm]["tokenizer_mode"],
        "baseline_schedule_preserved": False, "baseline_run_modified": False}
    (run / "pilot-plan.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
