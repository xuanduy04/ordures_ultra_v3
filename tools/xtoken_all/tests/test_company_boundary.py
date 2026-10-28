"""Legacy source protection and explicit X-Token integration boundaries."""

import ast
from copy import deepcopy
from functools import partial
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
BASELINE = "d92dc0967696a8982c6f226b97cd74957fb5c83d"
PROTECTED = {"grpo.py", "reward_functions.py", "async_utils.py", "opd.py", "run_grpo_nemo_gym.py",
             "rollouts.py", "filter_overlong_token.py", "use_custom_opt.sh"}


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


integrity = load("company_source_integrity", "tools/xtoken_all/integrity.py")


def test_legacy_grpo_rewards_and_original_opd_match_pinned_company_source():
    result = integrity.verify_company_source(ROOT)
    assert result == {"commit": BASELINE, "files_checked": 8}
    contract = json.loads((ROOT / "tools/xtoken_all/super-runtime-lineage.json").read_text())
    assert {Path(path).name for path in contract["company_baseline"]["normalized_lf_sha256"]} == PROTECTED


def test_source_capture_rejects_modified_company_module(tmp_path):
    lineage = ROOT / "tools/xtoken_all/super-runtime-lineage.json"
    target = tmp_path / lineage.relative_to(ROOT)
    target.parent.mkdir(parents=True)
    target.write_bytes(lineage.read_bytes())
    contract = json.loads(lineage.read_text())["company_baseline"]
    for relative in contract["normalized_lf_sha256"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert integrity.verify_company_source(tmp_path)["files_checked"] == 8
    changed = tmp_path / "nemo_rl/algorithms/reward_functions.py"
    with changed.open("ab") as stream:
        stream.write(b"\n# Unreviewed company change\n")
    with pytest.raises(ValueError, match="reward_functions.py"):
        integrity.source_snapshot(tmp_path)


def test_xtoken_rollout_does_not_import_the_grpo_training_module():
    for relative in ["nemo_rl/algorithms/x_token/on_policy.py", "nemo_rl/algorithms/x_token/refit.py"]:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        assert not any(isinstance(node, ast.ImportFrom) and node.module == "nemo_rl.algorithms.grpo"
                       for node in ast.walk(tree))


@pytest.mark.parametrize("colocated", [True, False])
def test_isolated_refit_orders_transfer_and_preserves_explicit_buffer(monkeypatch, colocated):
    events = []

    def record(name, result=None):
        def call(*args, **kwargs):
            events.append((name, kwargs))
            return result
        return call

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda result: result))
    module = load("isolated_xtoken_refit", "nemo_rl/algorithms/x_token/refit.py")
    module._malloc_trim = lambda: None
    module._log_rss = lambda tag: None
    policy = SimpleNamespace(offload_before_refit=record("offload"),
        offload_after_refit=record("offload_after"), prepare_for_training=record("train"),
        stream_weights_via_ipc_zmq=record("stream", [True]),
        broadcast_weights_for_collective=record("broadcast", [True]))
    generation = SimpleNamespace(prepare_for_generation=record("prepare", True),
        update_weights_via_ipc_zmq=record("update", [True, None]),
        update_weights_from_collective=record("collective", [True]))
    module.refit_policy_generation(policy, generation, colocated, _refit_buffer_size_gb=4,
                                   kv_scales={"layer": 1.0})
    if colocated:
        assert events == [("offload", {}), ("prepare", {"tags": ["weights"]}),
                          ("stream", {"buffer_size_bytes": 4294967296}), ("update", {}),
                          ("offload_after", {}), ("prepare", {"tags": ["kv_cache"]})]
    else:
        assert events == [("offload", {}), ("broadcast", {"kv_scales": {"layer": 1.0}}),
                          ("collective", {}), ("train", {})]


def test_isolated_refit_rejects_failed_weight_transfer(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda result: result))
    module = load("failed_xtoken_refit", "nemo_rl/algorithms/x_token/refit.py")
    module._malloc_trim = lambda: None
    policy = SimpleNamespace(offload_before_refit=lambda: None,
                             stream_weights_via_ipc_zmq=lambda **kwargs: [True])
    generation = SimpleNamespace(prepare_for_generation=lambda **kwargs: True,
                                  update_weights_via_ipc_zmq=lambda: [False])
    with pytest.raises(RuntimeError, match="Updating weights"):
        module.refit_policy_generation(policy, generation, True, _refit_buffer_size_gb=4)


def setup_data_fixture():
    tree = ast.parse((ROOT / "nemo_rl/data/utils.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setup_response_data")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    marker = object()

    def loaded(config):
        return SimpleNamespace(dataset=[0, 1, 2], processor=marker, task_name=config["name"], task_spec=None)

    namespace = {"load_response_dataset": loaded,
                 "concatenate_datasets": lambda parts: [item for part in parts for item in part],
                 "AllTaskProcessedDataset": lambda data, *args, **kwargs: data}
    exec(compile(module, "actual_setup_response_data", "exec"), namespace)
    return namespace["setup_response_data"], marker


def test_company_data_path_does_not_filter_without_explicit_transform():
    setup, _ = setup_data_fixture()
    config = {"train": {"name": "train"}, "validation": {"name": "val"}, "max_input_seq_length": 1}
    assert setup(None, deepcopy(config)) == ([0, 1, 2], [0, 1, 2])


@pytest.mark.parametrize("maximum", [None, 1])
def test_xtoken_explicit_transform_retains_its_train_and_validation_filter(monkeypatch, maximum):
    setup, marker = setup_data_fixture()
    calls = []

    def filtered(data, tokenizer, limit):
        calls.append((tokenizer, limit))
        return data[:limit]

    monkeypatch.setitem(sys.modules, "nemo_rl.data.processors", SimpleNamespace(
        nemo_gym_data_processor=marker, filter_overlong_nemo_gym_dataset=filtered))
    adapter = load("xtoken_dataset_transform", "nemo_rl/algorithms/x_token/data.py")
    config = {"train": {"name": "train"}, "validation": {"name": "val"}, "max_input_seq_length": maximum}
    result = setup("tokenizer", deepcopy(config), dataset_transform=adapter.dataset_transform("tokenizer", config))
    assert result == (([0], [0]) if maximum == 1 else ([0, 1, 2], [0, 1, 2]))
    assert calls == ([("tokenizer", 1), ("tokenizer", 1)] if maximum == 1 else [])


@pytest.mark.parametrize("processor", [False, True])
def test_tokenizer_loading_kwargs_are_owned_by_the_new_entrypoint(monkeypatch, processor):
    calls = []
    token = SimpleNamespace(pad_token="pad", eos_token="eos", bos_token="bos", pad_token_id=0,
                            eos_token_id=1, bos_token_id=2, name_or_path="local",
                            apply_chat_template=lambda messages, **kwargs: kwargs)

    def constructor(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(tokenizer=token) if processor else token

    factory = SimpleNamespace(from_pretrained=constructor)
    tree = ast.parse((ROOT / "nemo_rl/algorithms/utils.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_tokenizer")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    namespace = {"maybe_patch_fastokens": lambda: None, "AutoTokenizer": factory,
                 "AutoProcessor": factory, "partial": partial}
    exec(compile(module, "actual_company_tokenizer", "exec"), namespace)
    config = {"name": "local", "from_pretrained_kwargs": {"extra_special_tokens": {}},
              "chat_template_kwargs": {"enable_thinking": False}}
    namespace["get_tokenizer"](config, get_processor=processor)
    old_options = {"trust_remote_code": True, **({"use_fast": True} if processor else {})}
    assert calls == [("local", old_options)]
    calls.clear()
    monkeypatch.setitem(sys.modules, "nemo_rl.data.processors", SimpleNamespace(
        nemo_gym_data_processor=object(), filter_overlong_nemo_gym_dataset=None))
    monkeypatch.setitem(sys.modules, "nemo_rl.algorithms.utils", SimpleNamespace(get_tokenizer=namespace["get_tokenizer"]))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=factory, AutoProcessor=factory))
    adapter = load("xtoken_tokenizer_options", "nemo_rl/algorithms/x_token/data.py")
    adapter.get_tokenizer(config, get_processor=processor)
    assert calls == [("local", {**old_options, "extra_special_tokens": {}})]
    assert token.apply_chat_template([])["enable_thinking"] is False
    with pytest.raises(ValueError, match="trust_remote_code"):
        adapter.get_tokenizer({"name": "local", "from_pretrained_kwargs": {"trust_remote_code": True}})
