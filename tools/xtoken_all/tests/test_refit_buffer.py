"""Exercise the actual refit size check with GPU/socket doubles, no allocations."""

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
EMBEDDING_BYTES = 151936 * 2048 * 2


def extract(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, relative, "exec"), namespace)


@pytest.mark.parametrize("total_gib,passes", [(1.0, False), (2.0, True), (4.0, True), (4, True)])
def test_real_streamer_halves_total_and_logs_actual_capacity(total_gib, passes, capsys):
    torch = SimpleNamespace(Tensor=object, uint8="uint8", cuda=MagicMock(),
                            empty=MagicMock(side_effect=lambda *args, **kwargs: MagicMock()))
    socket = MagicMock()
    namespace = {"torch": torch, "json": json, "gc": MagicMock(),
                 "zmq": SimpleNamespace(Again=type("Again", (Exception,), {}),
                                        ZMQError=type("ZMQError", (Exception,), {})),
                 "IPCProtocol": SimpleNamespace(COMPLETE="COMPLETE"),
                 "get_handle_from_tensor": lambda buffer: "fixture-handle"}
    extract("nemo_rl/models/policy/utils.py",
            ["calculate_aligned_size", "stream_weights_via_ipc_zmq_impl"], namespace)
    tensor = SimpleNamespace(nbytes=EMBEDDING_BYTES, device="fixture-cuda", data=MagicMock())
    params = iter([("model.embed_tokens.weight", tensor)])
    if passes:
        namespace["stream_weights_via_ipc_zmq_impl"](params, total_gib * (1024**3), socket, 1, "fixture-worker")
        assert socket.send_pyobj.call_args.args == ("COMPLETE",)
    else:
        with pytest.raises(AssertionError, match="622329856 > 536870912"):
            namespace["stream_weights_via_ipc_zmq_impl"](params, total_gib * (1024**3), socket, 1, "fixture-worker")
        socket.send_pyobj.assert_not_called()
    lines = capsys.readouterr().out.splitlines()
    report = json.loads(next(line.split("=", 1)[1] for line in lines if line.startswith("NRL_REFIT_BUFFER=")))
    expected = int(total_gib * (1024**3))
    assert report["total_buffer_bytes"] == expected
    assert report["per_buffer_capacity_bytes"] == expected // 2
    assert report["per_buffer_allocation_bytes"] == expected // 2
    assert report["buffer_count"] == 2 and report["rank"] == 1
    assert torch.empty.call_count == 2
    assert [call.args[0] for call in torch.empty.call_args_list] == [expected // 2] * 2
    torch.cuda.empty_cache.assert_called_once()


@pytest.mark.parametrize("gym", [True, False])
def test_driver_logs_resolved_knob_before_calling_refit(gym, capsys):
    path = "nemo_rl/algorithms/x_token/" + ("gym_rollout.py" if gym else "on_policy.py")
    tree = ast.parse((ROOT / path).read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == ("NemoGymOnPolicyRollout" if gym else "OnPolicyRollout"))
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    spec = importlib.util.spec_from_file_location("refit_test_protocol", ROOT / "nemo_rl/algorithms/x_token/gym_protocol.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    namespace = {"json": json, "prepare_example": protocol.prepare_example}
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec"), namespace)
    generation = MagicMock()
    generation.finish_generation.return_value = True
    refit = MagicMock(side_effect=RuntimeError("stop after request log"))
    state = SimpleNamespace(pending_rollout=None, synced_version=None, policy_version=0,
        tokenizer=SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [3, 4]),
        generation_config={"max_new_tokens": 32}, knobs={"max_prompt_tokens": 128, "refit_buffer_size_gb": 4.0},
        refit=refit, policy="policy", generation=generation)
    batch = {"gym_examples": [{"agent_ref": {"name": "general_qa_simple_agent"},
        "responses_create_params": {"input": [{"role": "user", "content": "2+3?"}]},
        "question": "2+3?", "expected_answer": "5", "should_use_judge": False}]} if gym else {
        "prompts": [[{"role": "user", "content": "2+3?"}]]}
    with pytest.raises(RuntimeError, match="stop after request log"):
        namespace["generate"](state, batch, training=True)
    report = json.loads(next(line.split("=", 1)[1] for line in capsys.readouterr().out.splitlines()
                             if line.startswith("XTOKEN_REFIT_REQUEST=")))
    assert report == {"policy_version": 0, "refit_buffer_size_gb": 4.0, "total_buffer_bytes": 4294967296}
    assert refit.call_args.kwargs["_refit_buffer_size_gb"] == 4.0


def test_qwen_gemma_and_gym_recipes_explicitly_select_four_gib():
    for name in ("xtoken_qwen3_gemma4_on_policy_distillation.yaml", "xtoken_gym_qwen3_gemma4.yaml"):
        cfg = yaml.safe_load((ROOT / "examples/configs" / name).read_text())
        assert cfg["on_policy"]["refit_buffer_size_gb"] == 4.0
