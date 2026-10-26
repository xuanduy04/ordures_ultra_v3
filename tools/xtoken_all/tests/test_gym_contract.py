"""Offline checks of real converter and Gym boundary; no tensor/GPU imports."""

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol = load_file("gym_protocol_isolated", "nemo_rl/algorithms/x_token/gym_protocol.py")
converter = load_file("gym_converter_isolated", "tools/xtoken_all/convert_general_qa.py")


def example():
    return {"agent_ref": {"name": "general_qa_simple_agent_reasoning_off"},
            "responses_create_params": {"input": [{"role": "user", "content": "2+3?"}]},
            "question": "2+3?", "expected_answer": "5", "should_use_judge": False}


def result(prompt=None, response=None, reward=1.0):
    return {"full_result": {"reward": reward}, "message_log": [
        {"role": "user", "token_ids": [3, 4] if prompt is None else prompt},
        {"role": "assistant", "token_ids": [7, 8] if response is None else response}]}


def test_preparation_copies_and_never_leaks_ground_truth():
    row = example()
    original = deepcopy(row)
    prepared, prompt = protocol.prepare_example(row, 9, 32)
    assert row == original
    assert prepared["_rowidx"] == 9
    assert prompt == [{"role": "user", "content": "2+3?"}]
    assert prepared["responses_create_params"]["max_output_tokens"] == 32
    assert "expected_answer" not in prepared["responses_create_params"]


@pytest.mark.parametrize("key,value", [("tools", []), ("instructions", "system"),
                                       ("extra_body", {"top_k": 1}), ("stop", ["5"])])
def test_rejects_hidden_generation_changes(key, value):
    row = example()
    row["responses_create_params"][key] = value
    with pytest.raises(ValueError):
        protocol.prepare_example(row, 0, 32)


@pytest.mark.parametrize("change", ["judge", "multiturn", "agent", "empty_answer"])
def test_unsupported_task_is_explicit(change):
    row = example()
    if change == "judge":
        row["should_use_judge"] = True
    elif change == "multiturn":
        row["responses_create_params"]["input"].append({"role": "assistant", "content": "5"})
    elif change == "agent":
        row["agent_ref"]["name"] = "code_gen_simple_agent"
    else:
        row["expected_answer"] = " "
    with pytest.raises(ValueError):
        protocol.prepare_example(row, 0, 32)


def test_uses_sampled_ids_and_preserves_order():
    payload = result()
    assert protocol.unpack_result(payload, [3, 4]) == ([3, 4, 7, 8], 1.0, 1.0)
    assert protocol.restore_order([(1, "b"), (0, "a")], 2) == ["a", "b"]


@pytest.mark.parametrize("pairs", [[(0, "a"), (0, "b")], [(2, "x")], [(0, "a")], [(True, "x")]])
def test_missing_or_duplicate_results_fail(pairs):
    with pytest.raises(ValueError):
        protocol.restore_order(pairs, 2)


@pytest.mark.parametrize("payload", [result(prompt=[9]), result(response=[True]),
                                      result(response=[-1]), result(reward=float("nan"))])
def test_bad_rollout_cannot_enter_loss(payload):
    with pytest.raises(ValueError):
        protocol.unpack_result(payload, [3, 4])


def test_masked_or_empty_samples_have_no_loss():
    payload = result()
    payload["full_result"]["instance_config"] = {"mask_sample": True}
    assert protocol.unpack_result(payload, [3, 4]) == ([3, 4], 0.0, None)
    assert protocol.unpack_result(result(response=[]), [3, 4])[1] == 0.0


def test_actual_company_converter_preserves_prompts_order_and_metadata(tmp_path):
    rows = [{"id": "b", "prompt": [{"role": "user", "content": "Tính 2+3?"}],
             "reward_model": {"ground_truth": "5"}, "data_source": "dapo"},
            {"id": "a", "prompt": [{"role": "user", "content": "3+4?"}],
             "reward_model": {"ground_truth": "7"}, "data_source": "dapo"}]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "converted"
    receipt = converter.convert(source, output, 2, "prompt", "reward_model.ground_truth", "dapo")
    converted = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    metadata = [json.loads(line) for line in (output / "source-metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["original_row"] for row in metadata] == rows
    assert receipt["output_rows"] == 2 and receipt["resampled"] is False
    assert [row["expected_answer"] for row in converted] == ["5", "7"]
    for row in converted:
        protocol.prepare_example(row, 0, 32)
    with pytest.raises(FileExistsError):
        converter.convert(source, output, 2, "prompt", "reward_model.ground_truth", "dapo")


def test_parquet_uses_same_company_converter(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist([{"question": "1+1?", "answer": 2}]), source)
    receipt = converter.convert(source, tmp_path / "out", 1, "question", "answer", "math")
    assert receipt["source_rows"] == receipt["output_rows"] == 1


@pytest.mark.parametrize("bad", [{"data_source": "CodeForces"}, {"answer": {"tests": []}},
                                 {"prompt": [{"role": "system", "content": "x"}]}])
def test_invalid_pool_cannot_be_reported_as_success(tmp_path, bad):
    row = {"prompt": "2+3?", "answer": "5", **bad}
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        converter.convert(source, output, 1, "prompt", "answer", "dapo")
    assert (output / "FAILED").exists()
    assert not (output / "conversion.json").exists()


def test_row_count_mismatch_is_not_silent_filtering(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('{"prompt":"2+3?","answer":"5"}\n')
    with pytest.raises(ValueError):
        converter.convert(source, tmp_path / "out", 2, "prompt", "answer", "dapo")


def test_gym_yaml_explicitly_separates_reward_and_distillation():
    cfg = yaml.safe_load((ROOT / "examples/configs/xtoken_gym_qwen3_gemma4.yaml").read_text())
    assert cfg["on_policy"]["nemo_gym"]["skip_load_from_dotenv"] is True
    assert cfg["on_policy"]["rollout_backend"] == "nemo_gym"
    assert cfg["data"]["train"]["processor"] == "nemo_gym_data_processor"
    qa = cfg["on_policy"]["nemo_gym"]["global_config"]["general_qa"]["resources_servers"]["general_qa"]
    assert qa["should_use_judge"] is False and qa["judge_model"]
    assert "on_policy_distillation" not in cfg


def test_rollout_version_lease_requires_a_real_update():
    tree = ast.parse((ROOT / "nemo_rl/algorithms/x_token/on_policy.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OnPolicyRollout")
    namespace = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "on_policy.py", "exec"), namespace)
    state = namespace["OnPolicyRollout"](None, None, None, None, {}, {}, 0, None)
    with pytest.raises(RuntimeError):
        state.after_update()
    state.pending_rollout = (0, 0)
    state.after_update()
    assert state.policy_version == 1 and state.pending_rollout is None
    with pytest.raises(RuntimeError):
        state.after_update()


def test_source_fingerprint_detects_code_edits(tmp_path):
    integrity = load_file("xtoken_integrity_isolated", "tools/xtoken_all/integrity.py")
    path = tmp_path / "nemo_rl/example.py"
    path.parent.mkdir()
    path.write_text("value = 1\n")
    before = integrity.source_snapshot(tmp_path)
    path.write_text("value = 2\n")
    after = integrity.source_snapshot(tmp_path)
    assert before["sha256"] != after["sha256"]
    assert list(after["files"]) == ["nemo_rl/example.py"]


@pytest.mark.parametrize("retain,has_ids", [(False, True), (True, True), (True, False)])
def test_actual_gym_postprocessor_keeps_bounded_prefix_only_when_requested(retain, has_ids):
    tree = ast.parse((ROOT / "nemo_rl/environments/nemo_gym.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NemoGym")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_postprocess_nemo_gym_to_nemo_rl_result")

    class IdList(list):
        def tolist(self):
            return list(self)

    # Only this method's list/ID control flow is under test, not Torch kernels.
    tensor_lists = SimpleNamespace(tensor=lambda values, **kwargs: IdList(values),
                                   cat=lambda parts: IdList(item for part in parts for item in part),
                                   equal=lambda left, right: left == right,
                                   int64="int64", long="long", float32="float32")
    namespace = {"torch": tensor_lists, "PreTrainedTokenizerBase": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "nemo_gym.py", "exec"), namespace)
    tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids: "text",
                                batch_decode=lambda rows: ["text"] * len(rows),
                                apply_chat_template=lambda *args, **kwargs: [3, 4])
    output = {"type": "message", "role": "assistant", "content": [{"text": "partial"}]}
    if has_ids:
        output.update(prompt_token_ids=[3, 4], generation_token_ids=[7, 8], generation_log_probs=[-1.0, -1.0])
    payload = {"reward": 0, "response": {"output": [output],
                                        "incomplete_details": {"reason": "max_output_tokens"}},
               "responses_create_params": {"input": [{"role": "user", "content": "2+3?"}]}}
    env = SimpleNamespace(cfg={"retain_truncated_token_ids": retain})
    parsed = namespace[method.name](env, payload, tokenizer)
    if retain and has_ids:
        assert parsed["message_log"][1]["token_ids"].tolist() == [7, 8]
    else:
        assert len(parsed["message_log"]) == 1
        assert parsed["message_log"][0]["token_ids"].tolist() == [0]


@pytest.mark.parametrize("failure", [None, "zero_gradient", "missing_step", "wrong_backend", "stale_version"])
def test_result_checker_rejects_incomplete_or_wrong_pipeline(tmp_path, failure):
    checker = load_file("xtoken_result_checker_isolated", "tools/xtoken_all/check_result.py")
    events = []
    for index in range(2):
        rollout = {"policy_version": index, "training": True, "response_tokens": 2,
                   "backend": "nemo_gym", "reward_used_in_loss": False}
        step = {"step": index + 1, "loss": 0.4, "grad_norm": 0.3}
        if failure == "zero_gradient":
            step["grad_norm"] = 0
        elif failure == "wrong_backend":
            rollout["backend"] = "direct"
        elif failure == "stale_version":
            rollout["policy_version"] = 0
        events.append("XTOKEN_ROLLOUT=" + json.dumps(rollout))
        if failure != "missing_step" or index == 0:
            events.append("XTOKEN_TRAIN_STEP=" + json.dumps(step))
        events.append('XTOKEN_TEACHER_SCORE={"representation":"topk_logprobs","top_k":20}')
    events.append("Max steps reached, stopping training.")
    (tmp_path / "driver.log").write_text("\n".join(events))
    (tmp_path / "process-exit.json").write_text('{"driver_exit_code":0,"tee_exit_code":0}')
    (tmp_path / "effective.yaml").write_text("checkpointing:\n  enabled: false\n")
    assert checker.assess(tmp_path, "on")["success"] is (failure is None)
