"""Pilot budget, same-vocab identity, rendering and provenance regressions."""

from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/xtoken_all"
sys.path.insert(0, str(TOOLS))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load("pilot_config_test", TOOLS / "prepare_pilot.py")
identity = load("pilot_identity_test", TOOLS / "tokenizer_contract.py")


def spec():
    return yaml.safe_load((TOOLS / "pilot_200.yaml").read_text())


def baseline():
    return {"distillation": {"max_num_steps": 1000, "seed": 42, "num_prompts_per_step": 4},
        "policy": {"model_name": "/models/qwen-1.7", "tokenizer": {"name": "/models/qwen-1.7", "chat_template_kwargs": {"enable_thinking": False}},
            "scheduler": [], "max_total_sequence_length": 2048, "generation": {"max_new_tokens": 256},
            "make_sequence_length_divisible_by": 1, "train_global_batch_size": 4},
        "teachers": [{"projection_matrix_path": "/root/map.pt", "tokenizer": {"name": "/models/gemma", "from_pretrained_kwargs": {"extra_special_tokens": {}}},
            "remote": {"url": "http://gemma", "model": "gemma", "api_key_env": None, "model_vocab_size": 262144},
            "max_total_sequence_length": 4096, "make_sequence_length_divisible_by": 1}],
        "checkpointing": {"enabled": True, "checkpoint_dir": "/root/runs/A-on-11/checkpoints", "save_period": 10},
        "logger": {"log_dir": "/root/runs/A-on-11/train-logs"},
        "data": {"train": {"data_path": "/root/data/full-math.jsonl"}},
        "on_policy": {"max_prompt_tokens": 1792, "refit_buffer_size_gb": 4.0,
            "teacher_max_total_sequence_length": 4096,
            "nemo_gym": {"actor_python": "/venvs/gym/bin/python", "source_root": "/gym", "venv_dir": "/gym-venvs", "ray_address": "old-head:6379",
                "source_contract": "/old-source/tools/xtoken_all/super-source-contract.json", "global_config": {"nemo_gym_log_dir": "/root/runs/A-on-11/gym-logs"}}}}


@pytest.mark.parametrize("arm", ["same", "cross", "same235"])
def test_both_arms_preserve_actual_pilot_data_and_budgets(arm):
    old = baseline()
    original = deepcopy(old)
    cfg = pilot.pilot_config(old, spec(), arm, PurePosixPath("/root/runs/A-on-11"), PurePosixPath("/root/runs/new"),
                             PurePosixPath("/old-source"), PurePosixPath("/new-source"), "new-head:6379")
    assert old == original
    assert cfg["distillation"]["max_num_steps"] == 200
    assert cfg["policy"]["scheduler"][0]["kwargs"]["total_iters"] == 20
    assert cfg["policy"]["scheduler"][1]["kwargs"]["T_max"] == 180
    assert cfg["checkpointing"]["save_period"] == 10
    assert cfg["checkpointing"]["keep_top_k"] == 20
    assert cfg["checkpointing"]["checkpoint_dir"] == "/root/runs/new/checkpoints"
    assert cfg["data"] == old["data"] and cfg["policy"]["generation"] == old["policy"]["generation"]
    assert cfg["policy"]["max_total_sequence_length"] == old["policy"]["max_total_sequence_length"]
    assert cfg["on_policy"]["nemo_gym"]["ray_address"] == "new-head:6379"
    assert cfg["on_policy"]["nemo_gym"]["source_contract"].startswith("/new-source/")
    if spec()["arms"][arm]["tokenizer_mode"] == "same":
        assert cfg["teachers"][0]["projection_matrix_path"] is None
        assert "from_pretrained_kwargs" not in cfg["teachers"][0]["tokenizer"]
        assert cfg["teachers"][0]["remote"]["model_vocab_size"] == 151936
        assert cfg["teachers"][0]["remote"]["model"] == spec()["arms"][arm]["teacher_model"]
        if arm == "same235":
            assert cfg["teachers"][0]["tokenizer"]["name"] == old["policy"]["tokenizer"]["name"]
    else:
        assert cfg["teachers"] == old["teachers"]


def test_yaml_profiles_share_the_approved_budget():
    train = yaml.safe_load((ROOT / "examples/configs/xtoken_gym_qwen3_gemma4_train.yaml").read_text())
    same = yaml.safe_load((ROOT / "examples/configs/xtoken_gym_qwen3_same_tokenizer_train.yaml").read_text())
    assert same["defaults"] == "xtoken_gym_qwen3_gemma4_train.yaml"
    assert same["teachers"][0]["projection_matrix_path"] is None
    assert train["distillation"]["max_num_steps"] == spec()["distillation"]["max_num_steps"] == 200
    assert train["policy"]["scheduler"] == spec()["scheduler"]
    assert train["checkpointing"]["save_period"] == 10
    assert train["checkpointing"]["keep_top_k"] == spec()["checkpointing"]["keep_top_k"] == 20
    largest = yaml.safe_load((ROOT / "examples/configs/xtoken_gym_qwen3_235b_same_tokenizer_train.yaml").read_text())
    assert largest["defaults"] == "xtoken_gym_qwen3_same_tokenizer_train.yaml"
    assert len(largest["teachers"]) == 1
    assert largest["teachers"][0]["projection_matrix_path"] is None
    assert largest["teachers"][0]["remote"]["model"] == spec()["arms"]["same235"]["teacher_model"]


def test_independent_arms_reject_a_multi_teacher_baseline():
    old = baseline()
    old["teachers"].append(deepcopy(old["teachers"][0]))
    with pytest.raises(ValueError, match="one teacher per independent run"):
        pilot.pilot_config(old, spec(), "same235", PurePosixPath("/root/runs/old"), PurePosixPath("/root/runs/new"),
                           PurePosixPath("/old-source"), PurePosixPath("/new-source"), "host:6379")


def test_actual_super_retention_keeps_every_pilot_checkpoint():
    tree = ast.parse((ROOT / "nemo_rl/utils/checkpoint.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "remove_old_checkpoints")
    history = [(step, "/checkpoints/step_" + str(step), {}) for step in range(10, 201, 10)]
    remove = MagicMock()
    namespace = {"_load_checkpoint_history": lambda _: history, "shutil": SimpleNamespace(rmtree=remove)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_super_retention", "exec"), namespace)
    manager = SimpleNamespace(checkpoint_dir="/checkpoints", keep_top_k=spec()["checkpointing"]["keep_top_k"],
        ft_keep_latest_k=None, save_period=10, metric_name=None)
    namespace["remove_old_checkpoints"](manager)
    remove.assert_not_called()


class Tokenizer:
    chat_template = "same"

    def __len__(self):
        return 2

    def get_vocab(self):
        return {"a": 0, "b": 1}


@pytest.mark.parametrize("mismatch", [None, "ids", "template", "kwargs", "context", "grid"])
def test_identity_requires_more_than_vocab_size(mismatch):
    student, teacher = Tokenizer(), Tokenizer()
    cfg = baseline()
    tc = cfg["teachers"][0]
    tc["tokenizer"] = deepcopy(cfg["policy"]["tokenizer"])
    if mismatch == "ids":
        teacher.get_vocab = lambda: {"a": 1, "b": 0}
    elif mismatch == "template":
        teacher.chat_template = "different"
    elif mismatch == "kwargs":
        tc["tokenizer"]["chat_template_kwargs"] = {}
    elif mismatch == "context":
        tc["max_total_sequence_length"] = 1
    elif mismatch == "grid":
        tc["make_sequence_length_divisible_by"] = 2
    if mismatch is None:
        identity.validate_identity(student, teacher, cfg["policy"], tc)
    else:
        with pytest.raises(ValueError):
            identity.validate_identity(student, teacher, cfg["policy"], tc)


def test_null_projection_never_hashes_or_opens_a_fake_file():
    digest = MagicMock()
    assert identity.projection_digest(None, digest) is None
    digest.assert_not_called()
    identity.projection_digest("map.pt", digest)
    digest.assert_called_once_with("map.pt")


def test_actual_renderer_supports_null_projection(tmp_path, monkeypatch):
    renderer = load("pilot_render_test", TOOLS / "render.py")
    cfg = baseline()
    source = ROOT
    run = tmp_path / "runs/B-test"
    for name in ("student", "teacher", "gym", "services"):
        (tmp_path / name).mkdir()
    actor = tmp_path / "actor-python"
    actor.write_text("fixture")
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    (tmp_path / "student/config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 2048, "num_hidden_layers": 28, "tie_word_embeddings": True}))
    omega = SimpleNamespace(to_container=lambda value, **kwargs: deepcopy(value), create=lambda value: value,
                            save=lambda value, path: path.write_text(yaml.safe_dump(value)))
    monkeypatch.setitem(sys.modules, "omegaconf", SimpleNamespace(OmegaConf=omega))
    monkeypatch.setitem(sys.modules, "nemo_rl.utils.config", SimpleNamespace(load_config=lambda _: cfg, register_omegaconf_resolvers=lambda: None))
    monkeypatch.setitem(sys.modules, "nemo_rl.algorithms.xtoken_off_policy_distillation", SimpleNamespace(MasterConfig=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "nemo_rl.algorithms.x_token.on_policy", SimpleNamespace(validate_on_policy_config=lambda _: None))
    monkeypatch.setitem(sys.modules, "nemo_rl.algorithms.x_token.gym_compat", SimpleNamespace(gym_actor_runtime_env=lambda *args: None))
    monkeypatch.setattr(renderer, "source_snapshot", lambda _: {"sha256": "fixture-source"})
    monkeypatch.setattr(renderer, "within", lambda path, root: Path(path).resolve())
    monkeypatch.setenv("XTOKEN_ALL_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["render.py", "--run-id", "B-test", "--student", str(tmp_path / "student"),
        "--teacher-tokenizer", str(tmp_path / "teacher"), "--same-tokenizer", "--teacher-vocab-size", "151936",
        "--data", str(data), "--teacher-url", "http://teacher", "--teacher-model", "Qwen3-32B",
        "--gym-actor-python", str(actor), "--gym-source", str(tmp_path / "gym"), "--gym-venvs", str(tmp_path / "services"), "--ray-address", "host:6379"])
    renderer.main()
    effective = yaml.safe_load((run / "effective.yaml").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    assert effective["teachers"][0]["projection_matrix_path"] is None
    assert "from_pretrained_kwargs" not in effective["teachers"][0]["tokenizer"]
    assert provenance["projection_sha256"] is None and provenance["tokenizer_mode"] == "same"


@pytest.mark.parametrize("failure", [None, "stale", "missing-step", "nonfinite", "exit", "checkpoint"])
def test_pilot_checker_is_not_the_two_step_smoke_checker(tmp_path, failure):
    checker = load("pilot_checker_test", TOOLS / "check_pilot.py")
    (tmp_path / "effective.yaml").write_text(yaml.safe_dump({"distillation": {"max_num_steps": 200},
        "checkpointing": {"checkpoint_dir": str(tmp_path / "checkpoints")}}))
    lines = []
    for index in range(200):
        if failure == "missing-step" and index == 99:
            continue
        lines.append("XTOKEN_TRAIN_STEP=" + json.dumps({"step": index + 1, "loss": float("nan") if failure == "nonfinite" else 1.0, "grad_norm": 1.0}))
        lines.append("XTOKEN_ROLLOUT=" + json.dumps({"training": True, "policy_version": 0 if failure == "stale" else index,
            "backend": "nemo_gym", "response_tokens": 4, "reward_used_in_loss": False}))
    lines.extend(["XTOKEN_TEACHER_SCORE=" + json.dumps({"representation": "topk_logprobs", "top_k": 20}),
                  "Max steps reached, stopping training."])
    (tmp_path / "driver.log").write_text("\n".join(lines))
    (tmp_path / "process-exit.json").write_text(json.dumps({"driver_exit_code": 1 if failure == "exit" else 0, "tee_exit_code": 0}))
    checkpoint = tmp_path / "checkpoints/step_200"
    checkpoint.mkdir(parents=True)
    (checkpoint / "training_info.json").write_text('{"total_steps":200}')
    (checkpoint / "config.yaml").write_text("checkpointing: {}\n")
    (checkpoint / "train_dataloader.pt").write_bytes(b"fixture")
    for name in ("weights", "optimizer", "tokenizer"):
        folder = checkpoint / "policy" / name
        folder.mkdir(parents=True)
        (folder / "fixture.bin").write_bytes(b"not-a-real-tensor")
    if failure == "checkpoint":
        (checkpoint / "train_dataloader.pt").unlink()
    result = checker.assess(tmp_path)
    assert result["success"] is (failure is None)
    assert result["steps_expected"] == 200
