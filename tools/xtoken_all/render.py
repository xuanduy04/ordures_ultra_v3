"""Materialize an immutable Gym run YAML inside the user's xtoken_all root."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from integrity import sha256, source_snapshot
from tokenizer_contract import projection_digest


def within(path, root):
    path = Path(path).resolve()
    path.relative_to(root)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional training profile; default remains the two-update smoke recipe")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--student-tokenizer", type=Path)
    parser.add_argument("--teacher-tokenizer", type=Path, required=True)
    tokenizer_mode = parser.add_mutually_exclusive_group(required=True)
    tokenizer_mode.add_argument("--projection", type=Path)
    tokenizer_mode.add_argument("--same-tokenizer", action="store_true")
    parser.add_argument("--teacher-vocab-size", type=int)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher-url", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--gym-actor-python", type=Path, required=True,
                        help="Existing NemoGym actor venv, not a thin service .venv")
    parser.add_argument("--gym-venvs", type=Path, required=True)
    parser.add_argument("--gym-source", type=Path, required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--teacher-api-key-env")
    args = parser.parse_args()
    root = Path(os.environ["XTOKEN_ALL_ROOT"]).resolve()
    run = within(root / "runs" / args.run_id, root)
    if not args.run_id or Path(args.run_id).name != args.run_id or run.exists():
        raise ValueError("Use a new, single-component run ID")
    for path in (args.student, args.teacher_tokenizer, args.gym_venvs, args.gym_source):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (args.data, args.gym_actor_python):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.projection is not None:
        if not args.projection.is_file():
            raise FileNotFoundError(args.projection)
    if args.teacher_url.rstrip("/").endswith("/v1") or "@" in args.teacher_url or "?" in args.teacher_url:
        raise ValueError("Use teacher base URL without /v1 or embedded credentials")
    if args.teacher_api_key_env and args.teacher_api_key_env not in os.environ:
        raise ValueError("Teacher authentication variable is unavailable")
    source = Path(__file__).resolve().parents[2]
    within(source, root)
    sys.path.insert(0, str(source))
    from omegaconf import OmegaConf
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from nemo_rl.algorithms.xtoken_off_policy_distillation import MasterConfig
    from nemo_rl.algorithms.x_token.on_policy import validate_on_policy_config
    from nemo_rl.algorithms.x_token.gym_compat import gym_actor_runtime_env, bind_teacher_endpoint

    # Preserve the venv entrypoint; resolving its symlink selects base Python.
    os.environ.update(XTOKEN_GYM_ACTOR_PYTHON=os.path.abspath(args.gym_actor_python),
                      XTOKEN_GYM_VENV_DIR=str(args.gym_venvs.resolve()),
                      XTOKEN_GYM_SOURCE=str(args.gym_source.resolve()),
                      XTOKEN_SOURCE_ROOT=str(source), XTOKEN_RAY_ADDRESS=args.ray_address,
                      XTOKEN_RUN_DIR=str(run), XTOKEN_STUDENT_MODEL=str(args.student.resolve()),
                      XTOKEN_TRAIN_DATA=str(args.data.resolve()))
    register_omegaconf_resolvers()
    config_path = within(args.config, root) if args.config is not None else source / "examples/configs/xtoken_gym_qwen3_gemma4.yaml"
    cfg = load_config(config_path)
    cfg["policy"]["model_name"] = str(args.student.resolve())
    cfg["policy"]["tokenizer"]["name"] = str((args.student_tokenizer if args.student_tokenizer is not None else args.student).resolve())
    teacher = cfg["teachers"][0]
    teacher["model_name"] = str(args.teacher_tokenizer.resolve())
    teacher["tokenizer"]["name"] = str(args.teacher_tokenizer.resolve())
    teacher["projection_matrix_path"] = None if args.same_tokenizer else str(args.projection.resolve())
    if args.same_tokenizer:
        teacher["tokenizer"] = deepcopy(cfg["policy"]["tokenizer"])
        teacher["tokenizer"]["name"] = str(args.teacher_tokenizer.resolve())
        if args.teacher_vocab_size is None or args.teacher_vocab_size <= 0:
            raise ValueError("Same-tokenizer mode requires --teacher-vocab-size from model metadata")
        teacher["remote"]["model_vocab_size"] = args.teacher_vocab_size
    teacher["remote"]["url"] = args.teacher_url.rstrip("/")
    teacher["remote"]["model"] = args.teacher_model
    teacher["remote"]["api_key_env"] = args.teacher_api_key_env
    teacher["train_global_batch_size"] = cfg["policy"]["train_global_batch_size"]
    teacher["max_total_sequence_length"] = cfg["on_policy"]["teacher_max_total_sequence_length"]
    bind_teacher_endpoint(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    validate_on_policy_config(MasterConfig(**resolved))
    gym_actor_runtime_env(resolved["on_policy"]["nemo_gym"], source, os.environ)
    model = json.loads((args.student / "config.json").read_text())
    if not isinstance(model["model_type"], str) or type(model["vocab_size"]) is not int or model["vocab_size"] <= 0:
        raise ValueError("Student config must provide model_type and a positive vocab_size")
    run.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(resolved), run / "effective.yaml")
    provenance = {"mode": "on", "rollout_backend": "nemo_gym",
                  "reward_used_in_loss": False, "source_root": str(source),
                  "effective_config_sha256": sha256(run / "effective.yaml"),
                  "data_sha256": sha256(args.data), "projection_sha256": projection_digest(args.projection, sha256),
                  "tokenizer_mode": "same" if args.same_tokenizer else "cross",
                  "source_sha256": source_snapshot(source)["sha256"]}
    (run / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"run_dir": str(run), "steps": resolved["distillation"]["max_num_steps"], "mode": "on",
                      "backend": "nemo_gym", "gym_actor_python": resolved["on_policy"]["nemo_gym"]["actor_python"]}))


if __name__ == "__main__":
    main()
