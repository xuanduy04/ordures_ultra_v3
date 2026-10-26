"""Execute reviewed SDK boundary methods without installing the GPU stack."""

import ast
import asyncio
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import MagicMock
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = load_file("gym_compat_boundaries", "nemo_rl/algorithms/x_token/gym_compat.py")
protocol = load_file("gym_protocol_boundaries", "nemo_rl/algorithms/x_token/gym_protocol.py")


def method(source, class_name, name, namespace):
    tree = ast.parse(source)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in owner.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ast.fix_missing_locations(future)
    exec(compile(ast.Module(body=[future, node], type_ignores=[]), "reviewed_source.py", "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def sdk():
    if "XTOKEN_REVIEWED_GYM_ZIP" not in os.environ:
        pytest.skip("Set the reviewed SDK ZIP path to run archive-bound tests")
    with zipfile.ZipFile(os.environ["XTOKEN_REVIEWED_GYM_ZIP"]) as archive:
        yield {name: archive.read(name) for name in archive.namelist()}


@pytest.mark.parametrize("key", ["NEMO_GYM_CONFIG_DICT", "NEMO_GYM_CONFIG_PATH"])
def test_foreign_gym_environment_is_rejected_without_echoing_value(key):
    with pytest.raises(ValueError) as error:
        compat.reject_inherited_gym_config({key: "private-value-do-not-echo"})
    assert "private-value" not in str(error.value)
    compat.reject_inherited_gym_config({key: ""})


def test_pipeline_cannot_rebuild_the_provisioned_runtime():
    with pytest.raises(ValueError, match="NRL_FORCE_REBUILD_VENVS"):
        compat.reject_inherited_gym_config({"NRL_FORCE_REBUILD_VENVS": "true"})


@pytest.mark.parametrize("address", ["auto", "local", "", "ray://host:10001", "host:0", "host:65536", ":8000"])
def test_no_ray_autostart_address(address):
    with pytest.raises(ValueError):
        compat.validate_ray_address(address)


def test_ray_connect_failure_never_falls_back_to_another_cluster():
    cfg = SimpleNamespace(on_policy={"nemo_gym": {"ray_address": "localhost:6767"}},
                          cluster={"num_nodes": 1, "gpus_per_node": 4})
    ray = MagicMock()
    ray.is_initialized.return_value = False
    ray.init.side_effect = ConnectionError("head unavailable")
    with pytest.raises(ConnectionError):
        compat.init_existing_gym_ray(cfg, ray, {})
    ray.init.assert_called_once()
    assert ray.init.call_args.kwargs["address"] == "localhost:6767"


def test_ray_topology_mismatch_disconnects_this_driver():
    cfg = SimpleNamespace(on_policy={"nemo_gym": {"ray_address": "localhost:6767"}},
                          cluster={"num_nodes": 1, "gpus_per_node": 4})
    ray = MagicMock()
    ray.is_initialized.return_value = False
    ray.nodes.return_value = [{"Alive": True}]
    ray.cluster_resources.return_value = {"GPU": 8}
    with pytest.raises(RuntimeError):
        compat.init_existing_gym_ray(cfg, ray, {})
    ray.shutdown.assert_called_once()


def test_archive_closure_matches_committed_contract(sdk, tmp_path):
    contract = ROOT / "tools/xtoken_all/super-source-contract.json"
    for name, data in sdk.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    report = compat.verify_gym_source(tmp_path, contract)
    assert report["files_checked"] == 60
    (tmp_path / "nemo_gym/global_config.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source differs"):
        compat.verify_gym_source(tmp_path, contract)


def test_actual_sdk_forwards_unfiltered_sampling_and_template(sdk):
    fn = method(sdk["responses_api_models/vllm_model/app.py"], "VLLMModel",
                "_preprocess_chat_completion_create_params", {"deepcopy": deepcopy, "json": json})
    model = SimpleNamespace(config=SimpleNamespace(replace_developer_role_with_system=False,
        model="student", chat_template_kwargs={"enable_thinking": False},
        extra_body=None, return_token_id_information=True, uses_reasoning_parser=False))
    body = {"messages": [{"role": "user", "content": "2+3?"}], "temperature": 1.0, "top_p": 1.0,
            "max_tokens": 32}
    result = fn(model, None, body)
    assert result["temperature"] == result["top_p"] == 1.0
    assert result["max_tokens"] == 32
    assert result["chat_template_kwargs"] == {"enable_thinking": False}
    assert result["return_tokens_as_token_ids"] is True and result["logprobs"] is True
    assert "top_k" not in result


def test_actual_sdk_judge_is_not_invoked_for_g1(sdk):
    fn = method(sdk["resources_servers/general_qa/app.py"], "GeneralQAResourcesServer", "_verify_answer", {})

    async def unexpected_judge(*args):
        raise AssertionError("G1 must not call an LLM judge")

    server = SimpleNamespace(config=SimpleNamespace(should_use_judge=False),
        _verify_answer_deterministically=lambda *args: (0.0, "partial"),
        _verify_answer_with_judge=unexpected_judge)
    assert asyncio.run(fn(server, "2+3?", "5", "partial", False)) == (0.0, "partial", 0.0, None)


def test_actual_sdk_uses_preprovisioned_environment_without_install(sdk, tmp_path):
    tree = ast.parse(sdk["nemo_gym/cli_setup_command.py"])
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setup_env_command")
    namespace = {"Path": Path, "PARENT_DIR": tmp_path / "Gym", "DictConfig": dict}
    for name in ("HEAD_SERVER_DEPS", "UV_VENV_DIR", "PYTHON_VERSION", "SKIP_VENV_IF_PRESENT",
                 "UV_PIP_SET_PYTHON", "PIP_INSTALL_VERBOSE"):
        namespace[name + "_KEY_NAME"] = name.lower()
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "sdk_setup.py", "exec"), namespace)
    root = tmp_path / "venvs"
    directory = tmp_path / "Gym/resources_servers/general_qa"
    bins = root / "resources_servers/general_qa/.venv/bin"
    bins.mkdir(parents=True)
    (bins / "python").write_text("placeholder")
    (bins / "activate").write_text("placeholder")
    cfg = {"head_server_deps": [], "uv_venv_dir": str(root), "python_version": "3.12",
           "skip_venv_if_present": True}
    command = namespace[fn.name](directory, cfg, "general_qa")
    assert "source " in command and "uv " not in command and "pip install" not in command


@pytest.mark.parametrize("failure", ["submit", "wait", "refit"])
def test_actual_rollout_failure_releases_generation_without_tensor_import(failure):
    source = (ROOT / "nemo_rl/algorithms/x_token/gym_rollout.py").read_text()
    ray = MagicMock()
    ray.wait.return_value = ([], [])
    fn = method(source, "NemoGymOnPolicyRollout", "generate", {
        "prepare_example": protocol.prepare_example, "time": time, "ray": ray})
    generation = MagicMock()
    generation.finish_generation.return_value = True
    environment = MagicMock()
    if failure == "submit":
        environment.run_rollouts.remote.side_effect = RuntimeError("submit failed")
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [3, 4])
    state = SimpleNamespace(pending_rollout=None, generation=generation,
        environment=environment, tokenizer=tokenizer, knobs={"max_prompt_tokens": 128, "refit_buffer_size_gb": 1},
        generation_config={"max_new_tokens": 32}, synced_version=None, policy_version=0,
        refit=MagicMock(), policy=None, gym_config={"result_timeout_s": 2})
    if failure == "refit":
        state.refit.side_effect = RuntimeError("refit failed")
    batch = {"gym_examples": [{"agent_ref": {"name": "general_qa_simple_agent"},
        "responses_create_params": {"input": [{"role": "user", "content": "2+3?"}]},
        "question": "2+3?", "expected_answer": "5", "should_use_judge": False}]}
    with pytest.raises(TimeoutError if failure == "wait" else RuntimeError):
        fn(state, batch, training=True)
    generation.finish_generation.assert_called_once()
    assert state.pending_rollout is None
    assert ray.cancel.call_count == (1 if failure == "wait" else 0)


def test_failed_preflight_invalidates_all_previous_successes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools/xtoken_all"))
    preflight = load_file("gym_preflight_transactions", "tools/xtoken_all/preflight.py")
    (tmp_path / "effective.yaml").write_text("policy: {}\n")
    for name in ("assets-check", "source-check", "gym-preflight"):
        (tmp_path / (name + ".json")).write_text('{"success":true}')
    monkeypatch.setitem(__import__("sys").modules, "omegaconf", None)
    with pytest.raises(ImportError):
        preflight.check(tmp_path)
    assert all(json.loads((tmp_path / (name + ".json")).read_text())["success"] is False
               for name in ("assets-check", "source-check", "gym-preflight"))
