from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _preprocess_underscore_args(argv: list[str]) -> list[str]:
    """Replace underscores with hyphens in ``--arg_name`` / ``--arg_name=val`` prefixes.

    argparse normalises ``-`` and ``_`` internally, so ``--output-path`` and
    ``--output_path`` would normally map to different destinations.  This
    function rewrites ``_`` to ``-`` in the leading ``--key`` portion so that
    argparse treats both spellings identically.
    """
    out: list[str] = []
    for arg in argv:
        if arg.startswith("--") and "=" in arg:
            key, _, val = arg.partition("=")
            arg = key.replace("_", "-") + "=" + val
        elif arg.startswith("--"):
            arg = arg.replace("_", "-")
        out.append(arg)
    return out


def _coerce_tools(tools):
    """Coerce *tools* into a list of dicts, json-loading string layers before validating.

    The 'tools' field may arrive as a list of JSON strings, a JSON string of a
    list, or a plain list of dicts.
    """
    if tools is None:
        return None

    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as exc:
            raise ValueError("Entry has invalid 'tools' field: not valid JSON") from exc

    if not isinstance(tools, list):
        raise ValueError("Entry has invalid 'tools' field: expected a list")

    coerced = []
    for idx, tool in enumerate(tools):
        if isinstance(tool, str):
            try:
                tool = json.loads(tool)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Entry has invalid 'tools' entry at index {idx}: not valid JSON") from exc
        if not isinstance(tool, dict):
            raise ValueError(f"Entry has invalid 'tools' entry at index {idx}: expected a dict")
        coerced.append(tool)

    return coerced


def _validate_output_path(output_path: Path) -> None:
    """Ensure *output_path* ends with ``.jsonl``, parent dir exists, and file does not already exist."""
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Output path must end with .jsonl, got: {output_path}")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)


def _convert_pass(input_path: Path, output_path: Path, skip_on_error: bool) -> int:
    """Convert SFT-format records from *input_path* into RL-style question/answer records."""
    processed = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for idx, line in enumerate(tqdm(infile, desc="Converting", unit="line")):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                messages = data.get("messages", [])

                if not messages:
                    continue

                tools = _coerce_tools(data.get("tools"))

                # Split: all turns except last -> question, last turn -> ground_truth
                question = messages[:-1]
                last_message = messages[-1]

                if not isinstance(last_message, dict):
                    continue

                tool_calls = last_message.get("tool_calls") or []
                if tool_calls:
                    if not tools:
                        raise ValueError("Entry has a tool-call answer but no 'tools' field")
                    if len(tool_calls) > 1:
                        raise ValueError("Entry has a tool-call answer with multiple tool calls: expected exactly 1")
                    tool_call = tool_calls[0]
                    if not isinstance(tool_call, dict):
                        raise ValueError("Entry has invalid tool call: expected a dict")
                    function = tool_call.get("function", tool_call)
                    if not isinstance(function, dict):
                        raise ValueError("Entry has invalid tool call: expected a dict")
                    name = function.get("name")
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("Entry has invalid tool call 'name': expected a non-empty string")
                    arguments = function.get("arguments")
                    if arguments is None:
                        raise ValueError("Entry has invalid tool call: missing 'arguments' field")
                    if not isinstance(arguments, (str, dict)):
                        raise ValueError("Entry has invalid tool call 'arguments': expected a string or a dict")
                    ground_truth = {"name": name, "arguments": arguments}
                    answer_is_tool_call = True
                else:
                    ground_truth = last_message.get("content", "").strip()
                    answer_is_tool_call = False

                if not question or not ground_truth:
                    continue

                transformed_data = {
                    "question": question,
                    "ground_truth": ground_truth,
                }
                if answer_is_tool_call:
                    transformed_data["answer_is_tool_call"] = True
                if tools:
                    transformed_data["tools"] = tools

                json.dump(transformed_data, outfile, ensure_ascii=False)
                outfile.write("\n")
                processed += 1

            except Exception as exc:
                if skip_on_error:
                    logger.warning(f"Skipping line {idx+1} (numbered from 1): {exc}")
                    skipped += 1
                    continue
                raise

    if skipped:
        logger.info(f"Skipped {skipped} invalid entries.")

    return processed


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Convert an SFT-format JSONL file into RL-style question/answer records.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input JSONL file (single file, no glob).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output JSONL file (default: '<input stem>-RL_QA.jsonl').",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        default=False,
        help="Skip invalid entries with a warning instead of aborting (default: false).",
    )

    args = parser.parse_args()

    input_path: Path = Path(args.input).resolve()
    output_path: Path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(input_path.stem + "-RL_QA.jsonl")
    )
    skip_on_error: bool = args.skip_on_error

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    if output_path == input_path:
        parser.error(f"Input and output paths must differ, got: {output_path}")

    try:
        _validate_output_path(output_path)
    except FileExistsError:
        if not args.yes:
            response = input(f"{output_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Existing output file will not be overridden; exiting.")
                sys.exit(0)

    logger.info(f"Converting {input_path} -> {output_path}")

    processed = _convert_pass(input_path, output_path, skip_on_error)

    logger.info(f"Done - wrote {processed} entries to {output_path}.")


if __name__ == "__main__":
    main()
