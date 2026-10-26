"""Check staged assets and imports; never start Ray, Gym servers or installs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from integrity import sha256, source_snapshot


SOURCE = Path(__file__).resolve().parents[2]


def gym_only(config_path):
    sys.path.insert(0, str(SOURCE))
    from nemo_gym import PARENT_DIR
    from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict
    from nemo_gym.cli_setup_command import setup_env_command
    from omegaconf import DictConfig, OmegaConf
    from nemo_rl.algorithms.x_token.gym_compat import (
        reject_inherited_gym_config, verify_gym_source, validate_resolved_gym_models,
    )

    reject_inherited_gym_config(os.environ)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    knobs = cfg["on_policy"]["nemo_gym"]
    verify_gym_source(knobs["source_root"], knobs["source_contract"])
    assert Path(PARENT_DIR).resolve() == Path(knobs["source_root"]).resolve(), "Gym import points to a different checkout"
    initial = dict(knobs["global_config"])
    initial.update(policy_model_name=cfg["policy"]["model_name"],
                   policy_base_url=["http://127.0.0.1:9/v1"], policy_api_key="unused-preflight",
                   ray_head_node_address="127.0.0.1:9", ray_namespace="xtoken-preflight",
                   ray_num_gpus_per_node=0, uv_venv_dir=knobs["venv_dir"])
    resolved = get_global_config_dict(global_config_dict_parser_config=GlobalConfigDictParserConfig(
        initial_global_config_dict=DictConfig(initial), skip_load_from_cli=True,
        skip_load_from_dotenv=True))
    validate_resolved_gym_models(OmegaConf.to_container(resolved, resolve=True),
                                 cfg["policy"]["tokenizer"]["chat_template_kwargs"])
    for top, kind, name in [("policy_model", "responses_api_models", "vllm_model"),
                             ("policy_model_reasoning_off", "responses_api_models", "vllm_model"),
                             ("general_qa", "resources_servers", "general_qa"),
                             ("general_qa_simple_agent", "responses_api_agents", "simple_agent"),
                             ("general_qa_simple_agent_reasoning_off", "responses_api_agents", "simple_agent")]:
        assert "entrypoint" in resolved[top][kind][name], top
        command = setup_env_command(Path(PARENT_DIR) / kind / name, resolved, top)
        assert "uv venv" not in command and "pip install" not in command and "uv sync" not in command, "Gym would provision dependencies: " + top
    for kind, name in (("responses_api_models", "vllm_model"),
                        ("resources_servers", "general_qa"), ("responses_api_agents", "simple_agent")):
        root = Path(knobs["venv_dir"])
        venv = root / kind / name / ".venv"
        if root.resolve() == Path(PARENT_DIR).resolve():
            venv = Path(PARENT_DIR) / kind / name / ".venv"
        subprocess.run([str(venv / "bin/python"), "-B", str(SOURCE / "tools/xtoken_all/runtime_probe.py"),
                        "--role", "gym", "--gym-root", knobs["source_root"]], timeout=60, check=True)
    print("XTOKEN_GYM_PREFLIGHT_OK", flush=True)


def check(run):
    config_path = run / "effective.yaml"
    config_hash = sha256(config_path)
    for name in ("assets-check", "source-check", "gym-preflight"):
        (run / (name + ".json")).write_text(json.dumps({
            "success": False, "state": "preflight_pending",
            "effective_config_sha256": config_hash}) + "\n")
    from omegaconf import OmegaConf

    sys.path.insert(0, str(SOURCE))
    import torch
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.algorithms.x_token.gym_protocol import prepare_example
    from nemo_rl.algorithms.x_token.loss_utils import get_sparse_projection_matrix
    from nemo_rl.algorithms.x_token.on_policy import validate_on_policy_config
    from nemo_rl.algorithms.x_token.gym_compat import reject_inherited_gym_config, verify_gym_source
    from nemo_rl.algorithms.xtoken_off_policy_distillation import MasterConfig

    reject_inherited_gym_config(os.environ)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    meta = json.loads((run / "provenance.json").read_text())
    config_hash = sha256(config_path)
    assert config_hash == meta["effective_config_sha256"]
    assert source_snapshot(SOURCE)["sha256"] == meta["source_sha256"], "Source changed after render"
    validate_on_policy_config(MasterConfig(**cfg))
    gym_contract = verify_gym_source(cfg["on_policy"]["nemo_gym"]["source_root"],
                                      cfg["on_policy"]["nemo_gym"]["source_contract"])
    project_metadata = {}
    for name in ("pyproject.toml", "uv.lock", "default_runtime_env.yaml", ".python-version"):
        path = SOURCE / name
        assert path.is_file(), "Preserve active project metadata before preflight: " + name
        project_metadata[name] = sha256(path)

    def receipt(name, **fields):
        path = run / (name + ".json")
        payload = {"success": True, "effective_config_sha256": config_hash,
                   "checked_at_utc": datetime.now(timezone.utc).isoformat(), **fields}
        path.write_text(json.dumps(payload, indent=2) + "\n")

    student_dir = Path(cfg["policy"]["model_name"])
    weights_index = student_dir / "model.safetensors.index.json"
    if weights_index.is_file():
        index = json.loads(weights_index.read_text())
        for shard in set(index["weight_map"].values()):
            path = (student_dir / shard).resolve()
            path.relative_to(student_dir.resolve())
            assert path.is_file() and path.stat().st_size > 0, shard
    else:
        assert (student_dir / "model.safetensors").is_file(), "Missing student weights"
    student = get_tokenizer(cfg["policy"]["tokenizer"])
    teacher = get_tokenizer(cfg["teachers"][0]["tokenizer"])
    head = json.loads((student_dir / "config.json").read_text())["vocab_size"]
    assert max(student.get_vocab().values()) < head
    projection_path = Path(cfg["teachers"][0]["projection_matrix_path"])
    assert sha256(projection_path) == meta["projection_sha256"]
    projection = get_sparse_projection_matrix(
        projection_path, torch.device("cpu"), student_vocab_size=len(student),
        teacher_vocab_size=len(teacher))
    assert projection.layout == torch.sparse_coo
    assert projection.shape == (len(student), len(teacher)), "Tokenizer/projection dimensions differ"
    assert torch.isfinite(projection.coalesce().values()).all()
    assert (projection.coalesce().values() >= 0).all()
    data_path = Path(cfg["data"]["train"]["data_path"])
    assert sha256(data_path) == meta["data_sha256"]
    counts = []
    with data_path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            _, prompt = prepare_example(json.loads(line), index, cfg["policy"]["generation"]["max_new_tokens"])
            ids = student.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
            assert 0 < len(ids) <= cfg["on_policy"]["max_prompt_tokens"], "Prompt exceeds explicit smoke budget"
            counts.append(len(ids))
    assert len(counts) >= 8, "Two global batches need at least eight rows"
    receipt("assets-check", rows=len(counts), max_prompt_tokens=max(counts),
            student_tokenizer_len=len(student), student_head=head,
            teacher_tokenizer_len=len(teacher), projection_shape=list(projection.shape))
    paths = [os.environ["NRL_PYTHON"]]
    store = Path(os.environ["NEMO_RL_VENV_DIR"])
    workers = ["nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2",
               "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker"]
    inherited_path = os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(part for part in (str(SOURCE), cfg["on_policy"]["nemo_gym"]["source_root"], inherited_path) if part), "PYTHONDONTWRITEBYTECODE": "1",
           "UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}
    runtime_apis = []
    for worker, role in zip(workers, ("policy", "generation")):
        executable = store / worker / "bin/python"
        assert executable.is_file(), worker
        completed = subprocess.run([str(executable), "-B", str(SOURCE / "tools/xtoken_all/runtime_probe.py"), "--role", role], env=env,
                                   capture_output=True, text=True, timeout=180, check=True)
        reports = [json.loads(line.split("=", 1)[1]) for line in completed.stdout.splitlines()
                   if line.startswith("XTOKEN_RUNTIME_PROBE=")]
        assert len(reports) == 1, "Worker did not return a structured API report"
        runtime_apis.append(reports[0])
        paths.append(str(executable))
    receipt("source-check", source_sha256=meta["source_sha256"], checked_interpreters=len(paths),
            runtime_apis=runtime_apis, project_metadata=project_metadata)
    subprocess.run([cfg["on_policy"]["nemo_gym"]["worker_python"], "-B", __file__,
                    "--gym-only", str(config_path)], env=env, timeout=180, check=True)
    receipt("gym-preflight", skip_load_from_dotenv=True, servers_started=False,
            judge_inference_enabled=False, reviewed_source=gym_contract,
            scope="Config resolution and source imports in all preprovisioned Gym sub-venvs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--gym-only", type=Path)
    args = parser.parse_args()
    if args.gym_only:
        gym_only(args.gym_only)
    else:
        check(args.run_dir.resolve())


if __name__ == "__main__":
    main()
