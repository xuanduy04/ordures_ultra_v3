"""Preserve a fixed math/code pool for Gym KD; code QA scores are not metrics."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml

from tools.xtoken_all.convert_general_qa import digest, field, source_rows


def task_type(row, config):
    """Require explicit task labels or an operator-reviewed source mapping."""
    candidates = []
    for field_key, mapping_key in (("task_field", "task_values"), ("source_field", "source_tasks")):
        try:
            value = field(row, config[field_key])
        except KeyError:
            continue
        label = str(value).strip().lower()
        if label not in config[mapping_key]:
            raise ValueError(f"Unmapped {config[field_key]}={value!r}; review the recipe")
        candidates.append(config[mapping_key][label])
    if not candidates or len(set(candidates)) != 1 or candidates[0] not in ("math", "code"):
        raise ValueError("Missing, conflicting, or unsupported task classification")
    return candidates[0]


def inspect_pool(config):
    """Read only: report fields and counts before creating any converted files."""
    source = Path(config["input"])
    counts, sources, columns, errors = Counter(), Counter(), set(), Counter()
    rows = 0
    for row in source_rows(source):
        columns.update(row)
        rows += 1
        try:
            counts[task_type(row, config)] += 1
        except ValueError as error:
            errors[str(error)] += 1
        try:
            sources[str(field(row, config["source_field"]))] += 1
        except KeyError:
            sources["<missing>"] += 1
    return {"source_sha256": digest(source), "rows": rows, "columns": sorted(columns),
            "task_counts": dict(counts), "source_counts": dict(sources),
            "classification_errors": dict(errors), "conversion_performed": False}


def convert(config):
    from custom_utils.utils_data.convert_to_general_qa import _convert_entry

    expected = config["expected_task_counts"]
    if set(expected) != {"math", "code"} or any(type(n) is not int or n <= 0 for n in expected.values()):
        raise ValueError("Specify positive exact counts for both math and code")
    source, output = Path(config["input"]).resolve(), Path(config["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError("Output must be a new directory; never overwrite a pool")
    before = digest(source)
    output.mkdir(parents=True, exist_ok=False)
    counts, origins = Counter(), Counter()
    try:
        with (output / "train.jsonl").open("x", encoding="utf-8", newline="\n") as target, (output / "source-metadata.jsonl").open("x", encoding="utf-8", newline="\n") as metadata:
            for index, row in enumerate(source_rows(source)):
                task = task_type(row, config)
                prompt, answer = field(row, config["prompt_field"]), field(row, config["answer_field"])
                if isinstance(prompt, str):
                    prompt = [{"role": "user", "content": prompt}]
                if not isinstance(prompt, list) or len(prompt) != 1 or prompt[0]["role"] != "user":
                    raise ValueError(f"Row {index}: require exactly one user turn")
                text = prompt[0]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Row {index}: empty/non-text prompt")
                if any(key in row for key in ("tools", "tool_calls")) or any(key in prompt[0] for key in ("tools", "tool_calls")):
                    raise ValueError("Tool tasks need a separate reviewed adapter")
                if answer is None or isinstance(answer, bool) or not isinstance(answer, (str, int, float, dict, list)):
                    raise ValueError(f"Row {index}: missing/unsupported label")
                if task == "math" and isinstance(answer, (dict, list)):
                    raise ValueError(f"Row {index}: math answer must be scalar")
                if isinstance(answer, (dict, list)) and not answer:
                    raise ValueError(f"Row {index}: empty structured code label")
                json.dumps(answer, allow_nan=False)
                qa_answer = json.dumps(answer, ensure_ascii=False, allow_nan=False) if isinstance(answer, (dict, list)) else str(answer)
                if not qa_answer.strip():
                    raise ValueError(f"Row {index}: empty label")
                converted = _convert_entry({"messages": prompt, "expected_answer": qa_answer},
                    "messages", "expected_answer", False, True, "tools", "", config["dataset"])
                converted["xtoken_data"] = {"task_type": task, "reward_metric_valid": task == "math"}
                target.write(json.dumps(converted, ensure_ascii=False, allow_nan=False) + "\n")
                metadata.write(json.dumps({"source_row_index": index, "task_type": task,
                    "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "original_row": row}, ensure_ascii=False, allow_nan=False) + "\n")
                counts[task] += 1
                if counts[task] > expected[task]:
                    raise ValueError("Task count exceeds declared pool; no sampling or dropping allowed")
                try:
                    origins[str(field(row, config["source_field"]))] += 1
                except KeyError:
                    origins["<missing>"] += 1
        if dict(counts) != expected or digest(source) != before:
            raise ValueError(f"Wrong task counts {dict(counts)} or source changed")
        result = {"success": True, "source_sha256": before, "source_rows": sum(counts.values()),
            "output_rows": sum(counts.values()), "task_counts": dict(counts), "source_counts": dict(origins),
            "source_order_preserved": True, "resampled": False, "filtered": False,
            "answers_appended_to_input": False, "reward_used_in_loss": False,
            "code_execution_verified": False, "code_qa_reward_metric_valid": False,
            "output_sha256": digest(output / "train.jsonl"),
            "metadata_sha256": digest(output / "source-metadata.jsonl"), "recipe": config}
        (output / "conversion.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    except Exception:
        (output / "FAILED").write_text("Partial conversion; do not train from this directory.\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(inspect_pool(config) if args.inspect else convert(config), indent=2))


if __name__ == "__main__":
    main()
