"""Actor/service separation regressions; GPU and full SDK imports run on node."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import MagicMock

from pydantic import BaseModel
import pytest


ROOT = Path(__file__).resolve().parents[3]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = load("gym_runtime_boundary", "nemo_rl/algorithms/x_token/gym_compat.py")


@pytest.fixture
def teacher_config_class():
    """Use the production Pydantic class without importing the GPU training stack."""
    path = ROOT / "nemo_rl/algorithms/xtoken_off_policy_distillation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    schema = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TeacherConfig")
    namespace = {"BaseModel": BaseModel, "Any": Any, "Optional": Optional,
                 "cast": cast, "PolicyConfig": dict}
    exec(compile(ast.Module(body=[schema], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["TeacherConfig"]


def endpoint_config(teacher_config_class, shape):
    raw = {"teachers": [{"weight": 1.0, "remote": {"url": "http://teacher/", "model": "teacher"}}],
           "on_policy": {"nemo_gym": {"global_config": {"general_qa": {"resources_servers": {
               "general_qa": {"judge_server_url": "http://teacher", "judge_model": "teacher"}}}}}}}
    if shape != "raw":
        raw["teachers"] = [teacher_config_class.model_validate(raw["teachers"][0])]
    return SimpleNamespace(**raw) if shape == "driver" else raw


@pytest.mark.parametrize("shape", ["raw", "mixed", "driver"])
@pytest.mark.parametrize("changed", [None, "url", "model", "missing-url"])
def test_endpoint_binding_accepts_real_teacher_config_and_keeps_required_fields(teacher_config_class, shape, changed):
    config = endpoint_config(teacher_config_class, shape)
    fields = vars(config) if shape == "driver" else config
    teacher = fields["teachers"][0]
    remote = teacher["remote"] if shape == "raw" else teacher.remote
    qa = fields["on_policy"]["nemo_gym"]["global_config"]["general_qa"]["resources_servers"]["general_qa"]
    compat.validate_teacher_endpoint(config)
    if changed == "missing-url":
        del remote["url"]
        with pytest.raises(KeyError, match="url"):
            compat.validate_teacher_endpoint(config)
        with pytest.raises(KeyError, match="url"):
            compat.bind_teacher_endpoint(config)
    elif changed is not None:
        remote[changed] = "http://replacement/" if changed == "url" else "replacement"
        with pytest.raises(ValueError, match="Stale general_qa"):
            compat.validate_teacher_endpoint(config)
        compat.bind_teacher_endpoint(config)
        compat.validate_teacher_endpoint(config)
        assert qa["judge_server_url"] == remote["url"].rstrip("/")
        assert qa["judge_model"] == remote["model"]


@pytest.mark.parametrize("stale", [False, True])
def test_early_on_policy_validation_checks_normalized_teacher_before_model_setup(teacher_config_class, monkeypatch, stale):
    config = endpoint_config(teacher_config_class, "driver")
    config.on_policy.update(rollout_backend="nemo_gym", max_prompt_tokens=256,
        include_eos_loss=False, refit_buffer_size_gb=4, prompt_source="nemo_gym")
    config.on_policy["nemo_gym"].update(actor_python="/existing/actor/bin/python", ray_address="local:6379")
    config.policy = {"train_global_batch_size": 4, "max_total_sequence_length": 512, "generation": {
        "backend": "vllm", "colocated": {"enabled": True}, "temperature": 1.0, "top_p": 1.0,
        "top_k": None, "max_new_tokens": 256, "vllm_cfg": {
            "async_engine": True, "expose_http_server": True, "skip_tokenizer_init": False}}}
    config.distillation = {"num_prompts_per_step": 4, "val_period": 0, "val_at_start": False, "val_at_end": False}
    config.loss_fn = {"dynamic_loss_scaling": False, "ce_loss_scale": 0}
    config.data = {"train": {"dataset_name": "NemoGymDataset", "processor": "nemo_gym_data_processor"}}
    if stale:
        config.teachers[0].remote["url"] = "http://new-teacher"
    path = ROOT / "nemo_rl/algorithms/x_token/on_policy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    validate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_on_policy_config")
    monkeypatch.setitem(sys.modules, "nemo_rl.algorithms.x_token.gym_compat", compat)
    namespace = {}
    exec(compile(ast.Module(body=[validate], type_ignores=[]), str(path), "exec"), namespace)
    if stale:
        with pytest.raises(ValueError, match="Stale general_qa"):
            namespace[validate.name](config)
    else:
        namespace[validate.name](config)
    setup = ast.parse((ROOT / "nemo_rl/algorithms/xtoken_off_policy_distillation.py").read_text(encoding="utf-8"))
    setup_fn = next(node for node in setup.body if isinstance(node, ast.FunctionDef) and node.name == "setup")
    calls = [(node.func.id, node.lineno) for node in ast.walk(setup_fn)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    validation_line = next(line for name, line in calls if name == "validate_on_policy_config")
    policy_lines = [line for name, line in calls if name == "Policy"]
    assert policy_lines and all(validation_line < line for line in policy_lines)


@pytest.fixture
def knobs(tmp_path):
    actor = tmp_path / "ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python"
    actor.parent.mkdir(parents=True)
    actor.write_text("fixture executable")
    services = tmp_path / "gym_venvs"
    services.mkdir()
    sdk = tmp_path / "Gym"
    sdk.mkdir()
    return {"actor_python": str(actor), "venv_dir": str(services), "source_root": str(sdk)}


def actor_report(runtime):
    return {"role": "gym_actor", "python_executable": runtime["py_executable"],
            "python_prefix": str(Path(runtime["py_executable"]).parent.parent),
            "module": str(ROOT / "nemo_rl/environments/nemo_gym.py"),
            "gym_source_root": runtime["env_vars"]["PYTHONPATH"].split(os.pathsep)[1],
            "actor_import_completed": True, "cuda_operations_requested": False,
            "checked_actor_apis": ["_spinup", "run_rollouts", "shutdown"]}


def test_runtime_preserves_super_environment_merge_and_separate_service_store(knobs):
    inherited = {"PYTHONPATH": "existing-dependency-source", "NLTK_DATA": "existing-nltk",
                 "VIRTUAL_ENV": "old-env", "UV_PROJECT_ENVIRONMENT": "old-env",
                 "CUSTOM_FLAG": "kept"}
    runtime = compat.gym_actor_runtime_env(knobs, ROOT, inherited)
    assert runtime["py_executable"] == knobs["actor_python"]
    assert runtime["env_vars"]["CUSTOM_FLAG"] == "kept"
    assert runtime["env_vars"]["NLTK_DATA"] == "existing-nltk"
    assert runtime["env_vars"]["PYTHONPATH"].split(os.pathsep) == [str(ROOT), knobs["source_root"], "existing-dependency-source"]
    assert runtime["env_vars"]["NEMO_GYM_VENV_DIR"] == knobs["venv_dir"]
    assert runtime["env_vars"]["VIRTUAL_ENV"] == str(Path(knobs["actor_python"]).parent.parent)
    assert runtime["env_vars"]["UV_PROJECT_ENVIRONMENT"] == runtime["env_vars"]["VIRTUAL_ENV"]
    assert inherited["VIRTUAL_ENV"] == "old-env"
    compat.validate_gym_actor_report(actor_report(runtime), runtime)


@pytest.mark.parametrize("service", ["responses_api_models/vllm_model", "responses_api_agents/simple_agent", "resources_servers/general_qa"])
def test_thin_service_interpreter_is_rejected_for_actor(knobs, service):
    executable = Path(knobs["venv_dir"]) / service / ".venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("thin fixture")
    knobs["actor_python"] = str(executable)
    with pytest.raises(ValueError, match="thin Gym service"):
        compat.gym_actor_runtime_env(knobs, ROOT, {})


def test_old_config_is_rejected_instead_of_silently_reinterpreted(knobs):
    knobs["worker_python"] = knobs.pop("actor_python")
    with pytest.raises(ValueError, match="Re-render r7"):
        compat.gym_actor_runtime_env(knobs, ROOT, {})


def test_missing_actor_venv_never_falls_back_or_provisions(knobs):
    Path(knobs["actor_python"]).unlink()
    with pytest.raises(FileNotFoundError):
        compat.gym_actor_runtime_env(knobs, ROOT, {})


@pytest.mark.parametrize("key,value", [("role", "gym"), ("python_executable", "another/bin/python"),
    ("python_prefix", "another-env"), ("actor_import_completed", False),
    ("cuda_operations_requested", True), ("checked_actor_apis", []),
    ("module", "another/nemo_gym.py"), ("gym_source_root", "another/Gym")])
def test_receipt_cannot_replace_actual_actor_imports(knobs, key, value):
    runtime = compat.gym_actor_runtime_env(knobs, ROOT, {})
    report = actor_report(runtime)
    report[key] = value
    with pytest.raises(ValueError):
        compat.validate_gym_actor_report(report, runtime)


@pytest.mark.parametrize("typed_teacher", [False, True])
def test_actual_factory_passes_same_runtime_plan_to_nemogym(knobs, monkeypatch, teacher_config_class, typed_teacher):
    path = ROOT / "nemo_rl/algorithms/x_token/gym_rollout.py"
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_gym_environment")
    nemo_gym = MagicMock()
    monkeypatch.setitem(sys.modules, "nemo_rl.environments.nemo_gym",
                        SimpleNamespace(NemoGym=nemo_gym, NemoGymConfig=dict))
    ray = MagicMock()
    ray.get_runtime_context.return_value.namespace = "existing-namespace"
    knobs.update(global_config={"config_paths": [], "general_qa": {"resources_servers": {"general_qa": {
        "judge_server_url": "http://teacher", "judge_model": "teacher"}}}}, source_contract="fixture-contract",
                 skip_load_from_dotenv=True, retain_truncated_token_ids=True, result_timeout_s=5)
    namespace = {"Path": Path, "os": os, "ray": ray, "__file__": str(path),
                 "reject_inherited_gym_config": compat.reject_inherited_gym_config,
                 "validate_teacher_endpoint": compat.validate_teacher_endpoint,
                 "verify_gym_source": MagicMock(), "gym_actor_runtime_env": compat.gym_actor_runtime_env}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    env = {"PYTHONPATH": "existing-source", "NRL_IGNORE_VERSION_MISMATCH": "1"}
    monkeypatch.setattr(os, "environ", env)
    teacher = {"weight": 1.0, "remote": {"url": "http://teacher", "model": "teacher"}}
    config = SimpleNamespace(on_policy={"nemo_gym": knobs}, teachers=[
        teacher_config_class.model_validate(teacher) if typed_teacher else teacher])
    namespace[fn.name](config,
        SimpleNamespace(cfg={"model_name": "student"}, dp_openai_server_base_urls=["http://student"]))
    assert nemo_gym.options.call_args.kwargs["runtime_env"] == compat.gym_actor_runtime_env(knobs, ROOT, env)
    actor_config = nemo_gym.options.return_value.remote.call_args.args[0]
    assert actor_config["initial_global_config_dict"]["uv_venv_dir"] == knobs["venv_dir"]
    assert actor_config["base_urls"] == ["http://student"]
    nemo_gym.reset_mock()
    knobs["global_config"]["general_qa"]["resources_servers"]["general_qa"]["judge_server_url"] = "http://dead-teacher"
    with pytest.raises(ValueError, match="Stale general_qa"):
        namespace[fn.name](config, SimpleNamespace())
    nemo_gym.options.assert_not_called()


@pytest.mark.parametrize("service,module,class_name", [
    ("model", "responses_api_models.vllm_model.app", "VLLMModel"),
    ("agent", "responses_api_agents.simple_agent.app", "SimpleAgent"),
    ("qa", "resources_servers.general_qa.app", "GeneralQAResourcesServer"),
])
def test_service_probe_subprocess_never_imports_training_stack(tmp_path, service, module, class_name):
    sdk = tmp_path / "Gym"
    (sdk / "nemo_gym").mkdir(parents=True)
    (sdk / "nemo_gym/__init__.py").write_text("from pathlib import Path\nPARENT_DIR=Path(__file__).parent.parent\n")
    app = sdk / (module.replace(".", "/") + ".py")
    app.parent.mkdir(parents=True)
    app.write_text("class " + class_name + ":\n    pass\n")
    command = '''
import importlib.abc, runpy, sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in ('torch', 'vllm', 'transformers', 'nemo_rl'):
            raise AssertionError('Training stack imported in a thin service: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
sys.argv = [sys.argv[1], '--service', sys.argv[2], '--gym-root', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    completed = subprocess.run([sys.executable, "-B", "-c", command,
        str(ROOT / "tools/xtoken_all/gym_service_probe.py"), service, str(sdk)],
        capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.split("=", 1)[1])
    assert report["app_import_completed"] is True
    assert report["servers_started"] is False


def test_actor_probe_uses_real_package_import_and_does_not_hide_missing_torch():
    command = '''
import importlib.abc, runpy, sys, types
class MissingTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch':
            raise ModuleNotFoundError('actor-venv-missing-torch')
sys.meta_path.insert(0, MissingTorch())
sys.modules['ray'] = types.ModuleType('ray')
sys.path.insert(0, sys.argv[2])
sys.argv = [sys.argv[1], '--role', 'gym_actor', '--gym-root', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    completed = subprocess.run([sys.executable, "-B", "-c", command,
        str(ROOT / "tools/xtoken_all/runtime_probe.py"), str(ROOT)],
        capture_output=True, text=True, timeout=15)
    assert completed.returncode != 0
    assert "actor-venv-missing-torch" in completed.stderr
    assert "nemo_rl/environments/nemo_gym.py" in completed.stderr.replace("\\", "/")
