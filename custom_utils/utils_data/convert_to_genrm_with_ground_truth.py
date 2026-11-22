from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AGENT_TYPE = "responses_api_agents"
AGENT_NAME = "genrm_with_ground_truth_simple_agent"


def _preprocess_underscore_args(argv: list[str]) -> list[str]:
    """Replace underscores with hyphens in ``--arg_name`` / ``--arg_name=val`` prefixes.

    argparse normalises ``-`` and ``_`` internally, so ``--skip-on-error`` and ``--skip_on_error`` would normally map to
    different destinations.  This function rewrites ``_`` to ``-`` in the leading ``--key`` portion so that argparse
    treats both spellings identically.
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


def _validate_output_path(output_path: Path) -> None:
    """Ensure *output_path* ends with ``.jsonl``, parent dir exists, and file does not already exist."""
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Output path must end with .jsonl, got: {output_path}")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)


def _convert_chat_template_to_nemo_gym(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for turn in turns:
        role = turn["role"]
        content = turn.get("content")

        if role == "tool":
            out.append({"type": "function_call_output", "call_id": turn["tool_call_id"], "output": content})
        elif role == "assistant":
            out.append({"type": "message", "role": role, "content": content or ""})
            for tc in turn.get("tool_calls") or []:
                out.append(
                    {
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": json.dumps(tc["function"]["arguments"]),
                    }
                )
        elif role in ("user", "system"):
            out.append({"type": "message", "role": role, "content": content})
    return out


_OPENAI_FUNCTION_TOOL_KEYS = frozenset({"type", "name", "description", "parameters", "strict"})


def _convert_tools_to_nemo_gym(tools: list[dict], allow_unknown_tool_keys: bool) -> list[dict]:
    """Normalize tool definitions to the NeMo-Gym Responses API shape.

    Accepts both the chat-template form (``{"type": "function", "function": {...}}``) and the already-unwrapped form
    (``{"name": ..., "parameters": ...}``). OpenAI 2.7.2 (pinned by NeMo Gym) requires ``type: "function"`` and
    ``strict`` on function tools, so both are added when missing.

    NeMo Gym validates the whole row with ``extra="forbid"``, so any remaining top-level key outside the OpenAI
    function-tool schema (``type``, ``name``, ``description``, ``parameters``, ``strict``) would make the row fail with
    a 422 at rollout time. A top-level ``required`` is relocated into ``parameters["required"]`` when possible; other
    unknown keys raise a ``ValueError`` unless *allow_unknown_tool_keys* is set, in which case they are dropped. A
    ``parameters`` given as a JSON string is parsed into a dict. The contents of ``parameters`` are otherwise left
    untouched.
    """
    out = []
    for idx, tool in enumerate(tools):
        if "function" in tool:
            tool = tool["function"]
        else:
            tool = tool.copy()
        tool.setdefault("type", "function")
        if tool.get("type") == "function":
            tool.setdefault("strict", True)

            parameters = tool.get("parameters")
            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Entry has invalid tool 'parameters' field at index {idx}: expected valid JSON when a string"
                    ) from exc
                if not isinstance(parameters, dict):
                    raise ValueError(
                        f"Entry has invalid tool 'parameters' field at index {idx}: expected a dict after JSON parsing"
                    )
                tool["parameters"] = parameters

            required = tool.pop("required", None)
            if required is not None:
                parameters = tool.get("parameters")
                if isinstance(parameters, dict):
                    parameters.setdefault("required", required)
                elif not allow_unknown_tool_keys:
                    raise ValueError(
                        f"Entry has a top-level 'required' tool key at index {idx} "
                        "but 'parameters' is not a dict to relocate it into "
                        "(use --allow-unknown-tool-keys to drop it)"
                    )

            unknown_keys = sorted(key for key in tool if key not in _OPENAI_FUNCTION_TOOL_KEYS)
            if unknown_keys:
                if allow_unknown_tool_keys:
                    for key in unknown_keys:
                        tool.pop(key, None)
                else:
                    raise ValueError(
                        f"Entry has unknown tool key(s) at index {idx}: {unknown_keys}: "
                        "not allowed by the OpenAI Responses API function-tool schema "
                        "(use --allow-unknown-tool-keys to drop them)"
                    )
        out.append(tool)
    return out


def _convert_entry(
    entry: dict,
    prompt_field: str,
    ground_truth_field: str,
    tools_field: str,
    dataset: str,
    allow_unknown_tool_keys: bool,
) -> dict:
    """Convert one raw entry to the genrm_with_ground_truth JSONL schema."""
    prompt = entry.get(prompt_field, "")

    if not prompt or not str(prompt).strip():
        raise ValueError(f"Entry has empty or missing '{prompt_field}' field")

    if isinstance(prompt, dict) or isinstance(prompt, list):
        if isinstance(prompt, dict):
            prompt = [prompt]
        if any("role" not in prompt[i] for i in range(len(prompt))):
            raise ValueError("Entry has invalid chat template format: missing 'role' field in (at least) 1 turn")
        if any("content" not in prompt[i] for i in range(len(prompt))):
            raise ValueError("Entry has invalid chat template format: missing 'content' field in (at least) 1 turn")
        _input = _convert_chat_template_to_nemo_gym(prompt)
    else:
        _input = [{"role": "user", "content": str(prompt).strip()}]

    raw_ground_truth = entry.get(ground_truth_field)
    if raw_ground_truth is None or (isinstance(raw_ground_truth, str) and not raw_ground_truth.strip()):
        raise ValueError(f"Entry has empty or missing '{ground_truth_field}' field")
    if not isinstance(raw_ground_truth, str):
        raise ValueError(
            f"Entry has invalid '{ground_truth_field}' field: expected a non-empty string, "
            f"got {type(raw_ground_truth).__name__}"
        )
    ground_truth = raw_ground_truth.strip()

    responses_create_params: dict = {"input": _input}
    tools = entry.get(tools_field)
    if tools:
        if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
            raise ValueError(f"Entry has invalid '{tools_field}' field: expected a list of dicts")
        responses_create_params["tools"] = _convert_tools_to_nemo_gym(tools, allow_unknown_tool_keys)
    responses_create_params["parallel_tool_calls"] = False

    out: dict = {
        "responses_create_params": responses_create_params,
        "ground_truth": ground_truth,
        "agent_ref": {"type": AGENT_TYPE, "name": AGENT_NAME},
    }
    if dataset:
        out["dataset"] = dataset

    return out


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(description="Convert a JSON/JSONL dataset to genrm_with_ground_truth format.")
    parser.add_argument("input", type=Path, help="Path to the input data file (must be a .jsonl).")
    parser.add_argument("--output", type=Path, help="Path to the output JSONL file (must end with .jsonl).")
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )
    parser.add_argument(
        "--prompt-field",
        default="messages",
        help="Field name for the prompt (a string or a chat-template list of turns) (default: %(default)s).",
    )
    parser.add_argument("--tools-field", default="tools", help="Field name for the tools list (default: %(default)s).")
    parser.add_argument(
        "--ground-truth-field",
        default="ground_truth",
        help="Field name for the ground-truth answer text (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name stamped on each output row; an empty value omits the field entirely.",
    )
    parser.add_argument(
        "--allow-unknown-tool-keys",
        action="store_true",
        default=False,
        help="Drop unknown top-level keys in function tool definitions instead of raising an error (default: raise).",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        default=False,
        help="Skip invalid entries with a warning instead of aborting (default: false).",
    )

    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = (
        args.output
        if args.output is not None
        else args.input.with_name(args.input.stem + "-genrm_with_ground_truth.jsonl")
    )
    prompt_field: str = args.prompt_field
    tools_field: str = args.tools_field
    ground_truth_field: str = args.ground_truth_field
    dataset: str = args.dataset.strip()
    skip_on_error: bool = args.skip_on_error
    allow_unknown_tool_keys: bool = args.allow_unknown_tool_keys

    if not dataset:
        logger.info("--dataset is empty; the 'dataset' field will be omitted from all output rows.")

    input_path = input_path.resolve()
    output_path = output_path.resolve()

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if input_path == output_path:
        parser.error(f"Input and output paths must differ, got: {input_path}")

    try:
        _validate_output_path(output_path)
    except FileExistsError:
        if not args.yes:
            response = input(f"{output_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Existing output file will not be overridden; exiting.")
                sys.exit(0)

    logger.info(f"Converting {input_path} -> {output_path}")

    written = 0
    skipped = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(tqdm(fin, desc="Converting", unit="line"), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
                converted_entry = _convert_entry(
                    entry, prompt_field, ground_truth_field, tools_field, dataset, allow_unknown_tool_keys
                )
            except Exception as exc:
                if skip_on_error:
                    logger.warning(f"Skipping line {lineno}: {exc}")
                    skipped += 1
                    continue
                raise
            json.dump(converted_entry, fout, ensure_ascii=False)
            fout.write("\n")
            written += 1

    if written == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("No valid entries found - nothing to write.")

    if skipped:
        logger.info(f"Skipped {skipped} invalid entries.")

    logger.info(f"Done - wrote {written} entries to {output_path}")


if __name__ == "__main__":
    main()
