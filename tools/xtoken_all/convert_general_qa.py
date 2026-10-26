"""Convert a fixed local pool with the company's existing general_qa converter."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


CODE_SOURCES = ("codeforces", "code_contests", "codecontests", "taco", "apps")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def field(row, dotted):
    for key in dotted.split("."):
        row = row[key]
    return row


def source_rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError("Use a staged JSONL or Parquet input")


def convert(source, output, expected_rows, prompt_field, answer_field, dataset):
    from custom_utils.utils_data.convert_to_general_qa import _convert_entry

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source == output:
        raise FileExistsError("Conversion output must be a new directory")
    before = digest(source)
    output.mkdir(parents=True, exist_ok=False)
    rows, origins = 0, Counter()
    try:
        with (output / "train.jsonl").open("x", encoding="utf-8", newline="\n") as target, (output / "source-metadata.jsonl").open("x", encoding="utf-8", newline="\n") as metadata:
            for row in source_rows(source):
                origin = str(row["data_source"]).lower() if "data_source" in row else ""
                if any(name in origin for name in CODE_SOURCES) or (
                    "ability" in row and str(row["ability"]).lower() == "code"
                ):
                    raise ValueError("Code tasks require an execution verifier; use a math-only pool")
                prompt, answer = field(row, prompt_field), field(row, answer_field)
                if isinstance(prompt, list):
                    if len(prompt) != 1 or prompt[0]["role"] != "user":
                        raise ValueError("Initial Gym X-Token pool must contain one user turn")
                    text = prompt[0]["content"]
                else:
                    text = prompt
                    prompt = [{"role": "user", "content": text}]
                if not isinstance(text, str) or not text.strip() or not isinstance(answer, (str, int, float)) or isinstance(answer, bool):
                    raise ValueError("Empty prompt or missing answer; no rows are silently dropped")
                if isinstance(answer, float) and not math.isfinite(answer):
                    raise ValueError("Non-finite numeric answer")
                if any(key in row for key in ("tools", "tool_calls")):
                    raise ValueError("Tool datasets require a separate execution/verifier adapter")
                converted = _convert_entry(
                    {"messages": prompt, "expected_answer": answer},
                    "messages", "expected_answer", False, True, "tools", "", dataset,
                )
                target.write(json.dumps(converted, ensure_ascii=False) + "\n")
                metadata.write(json.dumps({"source_row_index": rows,
                    "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "original_row": row}, ensure_ascii=False) + "\n")
                if "data_source" in row:
                    origins[str(row["data_source"])] += 1
                rows += 1
        if rows != expected_rows or digest(source) != before:
            raise ValueError("Unexpected row count or source changed during conversion")
        result = {"success": True, "source_sha256": before,
                  "source_rows": rows, "output_rows": rows,
                  "output_sha256": digest(output / "train.jsonl"),
                  "metadata_sha256": digest(output / "source-metadata.jsonl"),
                  "source_order_preserved": True, "resampled": False,
                  "filtered": False, "answers_appended_to_input": False,
                  "reward_used_in_loss": False, "data_source_counts": dict(origins),
                  "converter": "custom_utils/utils_data/convert_to_general_qa.py::_convert_entry"}
        (output / "conversion.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception:
        (output / "FAILED").write_text("Partial conversion; do not use this directory.\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--prompt-field", required=True)
    parser.add_argument("--answer-field", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.output_dir, args.expected_rows,
                             args.prompt_field, args.answer_field, args.dataset), indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
