"""Mixed-pool conversion, provenance and code QA metric exclusions."""

from copy import deepcopy
import ast
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.xtoken_all.convert_math_code import convert, inspect_pool, task_type

spec = importlib.util.spec_from_file_location("mixed_gym_protocol", ROOT / "nemo_rl/algorithms/x_token/gym_protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


@pytest.fixture
def pool(tmp_path):
    config = yaml.safe_load((ROOT / "tools/xtoken_all/math_code_20k.yaml").read_text())
    assert config["expected_task_counts"] == {"math": 10000, "code": 10000}
    config.update(input=str(tmp_path / "pool.jsonl"), output_dir=str(tmp_path / "out"),
                  expected_task_counts={"math": 1, "code": 1})
    rows = [{"id": "m", "ability": "math", "data_source": "dapo",
             "prompt": [{"role": "user", "content": "  Tính 2+3?\n"}],
             "reward_model": {"ground_truth": "5"}},
            {"id": "c", "ability": "code", "data_source": "codeforces",
             "prompt": [{"role": "user", "content": "Implement sum(a, b)."}],
             "reward_model": {"ground_truth": {"inputs": ["2 3"], "outputs": ["5"]}}}]
    write_rows(config, rows)
    return config, rows


def write_rows(config, rows):
    Path(config["input"]).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_preserves_labels_order_and_generation_without_answers(pool):
    config, rows = pool
    receipt = convert(config)
    output = Path(config["output_dir"])
    examples = read_rows(output / "train.jsonl")
    metadata = read_rows(output / "source-metadata.jsonl")
    assert [row["original_row"] for row in metadata] == rows
    assert [row["source_row_index"] for row in metadata] == [0, 1]
    assert receipt["task_counts"] == {"math": 1, "code": 1}
    assert receipt["code_execution_verified"] is False
    assert json.loads(examples[1]["expected_answer"]) == rows[1]["reward_model"]["ground_truth"]
    for index, example in enumerate(examples):
        prepared, prompt = protocol.prepare_example(example, index, 256)
        assert prompt == rows[index]["prompt"]
        assert "expected_answer" not in prepared["responses_create_params"]
    assert protocol.reward_metric_valid(examples[0]) is True
    assert protocol.reward_metric_valid(examples[1]) is False
    with pytest.raises(FileExistsError):
        convert(config)


def test_inspect_is_read_only_and_counts_unmapped_labels(pool):
    config, rows = pool
    rows[1]["data_source"] = "unknown-code-source"
    write_rows(config, rows)
    result = inspect_pool(config)
    assert result["rows"] == 2 and result["classification_errors"]
    assert not Path(config["output_dir"]).exists()


@pytest.mark.parametrize("case", ["conflict", "unknown", "missing", "count", "empty-code", "missing-answer", "nan", "multiturn", "tools"])
def test_bad_pool_never_gets_success_receipt(pool, case):
    config, rows = pool
    if case == "conflict":
        rows[1]["ability"] = "math"
    elif case == "unknown":
        rows[1]["data_source"] = "unknown"
    elif case == "missing":
        del rows[1]["ability"], rows[1]["data_source"]
    elif case == "count":
        config["expected_task_counts"]["code"] = 2
    elif case == "empty-code":
        rows[1]["reward_model"]["ground_truth"] = {}
    elif case == "missing-answer":
        rows[1]["reward_model"]["ground_truth"] = None
    elif case == "nan":
        rows[0]["reward_model"]["ground_truth"] = float("nan")
    elif case == "multiturn":
        rows[1]["prompt"].append({"role": "assistant", "content": "answer"})
    else:
        rows[1]["tools"] = []
    write_rows(config, rows)
    with pytest.raises((ValueError, KeyError)):
        convert(config)
    output = Path(config["output_dir"])
    assert (output / "FAILED").exists() and not (output / "conversion.json").exists()


def test_parquet_and_jsonl_produce_identical_examples(pool, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    config, rows = pool
    for row in rows:
        row["reward_model"]["ground_truth"] = json.dumps(row["reward_model"]["ground_truth"])
    write_rows(config, rows)
    convert(config)
    parquet_config = deepcopy(config)
    parquet_config.update(input=str(tmp_path / "pool.parquet"), output_dir=str(tmp_path / "parquet-out"))
    pq.write_table(pa.Table.from_pylist(rows), parquet_config["input"])
    convert(parquet_config)
    assert (Path(config["output_dir"]) / "train.jsonl").read_bytes() == (Path(parquet_config["output_dir"]) / "train.jsonl").read_bytes()


def test_source_mapping_is_explicit_and_conflicts_fail(pool):
    config, rows = pool
    del rows[1]["ability"]
    assert task_type(rows[1], config) == "code"
    rows[1]["ability"] = "math"
    with pytest.raises(ValueError):
        task_type(rows[1], config)


@pytest.mark.parametrize("metadata", [{"task_type": "code", "reward_metric_valid": True},
                                      {"task_type": "math", "reward_metric_valid": False},
                                      {"task_type": "code", "reward_metric_valid": 0}])
def test_code_cannot_claim_qa_metric(metadata):
    with pytest.raises(ValueError):
        protocol.reward_metric_valid({"xtoken_data": metadata})


def test_excluding_code_reward_does_not_mask_kd(pool):
    config, _ = pool
    convert(config)
    example = read_rows(Path(config["output_dir"]) / "train.jsonl")[1]
    payload = {"full_result": {"reward": 0.0}, "message_log": [
        {"role": "user", "token_ids": [1, 2]}, {"role": "assistant", "token_ids": [3]}]}
    assert protocol.reward_metric_valid(example) is False
    assert protocol.unpack_result(payload, [1, 2]) == ([1, 2, 3], 1.0, 0.0)


@pytest.mark.parametrize("code_only", [False, True])
def test_actual_rollout_logs_exclude_code_scores(pool, capsys, code_only):
    """Execute real generate control flow; tensor/Ray I/O are test doubles."""
    config, _ = pool
    convert(config)
    examples = read_rows(Path(config["output_dir"]) / "train.jsonl")
    if code_only:
        examples = examples[1:]
    tree = ast.parse((ROOT / "nemo_rl/algorithms/x_token/gym_rollout.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NemoGymOnPolicyRollout")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
    ray, torch, environment = MagicMock(), MagicMock(), MagicMock()
    ray.wait.side_effect = lambda refs, **kwargs: (refs, [])
    ray.get.side_effect = lambda ref, **kwargs: ref
    payload = {"full_result": {"reward": 0.75}, "message_log": [
        {"role": "user", "token_ids": [1, 2]}, {"role": "assistant", "token_ids": [3]}]}
    environment.run_rollouts.remote.return_value = iter((i, deepcopy(payload), None) for i in range(len(examples)))
    token_mask = MagicMock()
    token_mask.sum.return_value = len(examples)
    state = SimpleNamespace(pending_rollout=None, synced_version=0, policy_version=0,
        generation=MagicMock(), environment=environment, rollout_counter=0,
        tokenizer=SimpleNamespace(pad_token_id=0, apply_chat_template=lambda *args, **kwargs: [1, 2]),
        knobs={"max_prompt_tokens": 128}, generation_config={"max_new_tokens": 32},
        gym_config={"result_timeout_s": 10}, response_collator=MagicMock(return_value={"token_mask": token_mask}))
    class Mask(list):
        def clone(self):
            return Mask(self)

        def __gt__(self, value):
            return Mask(item > value for item in self)

        def sum(self):
            return sum(self)

    batch = {"gym_examples": examples, "sample_mask": Mask([1.0] * len(examples)), "idx": list(range(len(examples)))}
    namespace = {"ray": ray, "torch": torch, "time": time, "json": json, "BatchedDataDict": dict,
        "prepare_example": protocol.prepare_example, "restore_order": protocol.restore_order,
        "unpack_result": protocol.unpack_result, "reward_metric_valid": protocol.reward_metric_valid}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "gym_rollout.py", "exec"), namespace)
    namespace["generate"](state, batch, training=True)
    report = json.loads(capsys.readouterr().out.split("XTOKEN_ROLLOUT=", 1)[1])
    assert report["gym_reward_sample_count"] == (0 if code_only else 1)
    assert report["gym_reward_mean"] == (None if code_only else 0.75)
    assert report["gym_code_qa_scores_excluded"] == 1
    assert report["reward_used_in_loss"] is False
    assert state.pending_rollout == (0, 1)
    assert state.response_collator.call_args.args[0]["sample_mask"] == [1.0] * len(examples)
    assert state.response_collator.call_args.args[1]["unpadded_sequence_lengths"] is not None
