"""Convert a JSON/JSONL dataset to genrm_compare format.

Each input entry must have a prompt field (a plain string or a chat-template
list of turns). The output is a JSONL file with entries containing
``agent_ref`` (with ``responses_api_agents`` type and a ``genrm_simple_agent``
or ``genrm_simple_agent_reasoning_off`` name), ``responses_create_params``
(with ``input``, ``tools`` when present, and ``parallel_tool_calls: false``),
and an optional ``dataset`` field.
"""

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
AGENT_NAME = "genrm_simple_agent"
AGENT_NAME_REASONING_OFF = "genrm_simple_agent_reasoning_off"


def _preprocess_underscore_args(argv: list[str]) -> list[str]:
    """Replace underscores with hyphens in ``--arg_name`` / ``--arg_name=val`` prefixes.

    argparse normalises ``-`` and ``_`` internally, so ``--skip-on-error`` and
    ``--skip_on_error`` would normally map to different destinations.  This
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


def _convert_chat_template_to_nemo_gym(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for turn in turns:
        role = turn["role"]
        content = turn.get("content")

        if role == "tool":
            out.append({
                "type": "function_call_output",
                "call_id": turn["tool_call_id"],
                "output": content,
            })
        elif role == "assistant":
            out.append({
                "type": "message",
                "role": role,
                "content": content or "",
            })
            for tc in turn.get("tool_calls") or []:
                out.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": json.dumps(tc["function"]["arguments"]),
                })
        elif role in ("user", "system"):
            out.append({
                "type": "message",
                "role": role,
                "content": content,
            })
    return out


def _convert_tools_to_nemo_gym(tools: list[dict]) -> list[dict]:
    """Normalize tool definitions to the NeMo-Gym Responses API shape.

    Accepts both the chat-template form (``{"type": "function", "function": {...}}``)
    and the already-unwrapped form (``{"name": ..., "parameters": ...}``).
    OpenAI 2.7.2 (pinned by NeMo Gym) requires ``type: "function"`` and ``strict``
    on function tools, so both are added when missing.
    """
    out = []
    for tool in tools:
        if "function" in tool:
            tool = tool["function"]
        else:
            tool = tool.copy()
        tool.setdefault("type", "function")
        if tool.get("type") == "function":
            tool.setdefault("strict", True)
        out.append(tool)
    return out


def _convert_entry(entry: dict, prompt_field: str, tools_field: str, no_reasoning: bool, dataset: str) -> dict:
    """Convert one raw entry to the genrm_compare JSONL schema."""
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

    responses_create_params: dict = {
        "input": _input,
    }
    tools = entry.get(tools_field)
    if tools:
        if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
            raise ValueError(f"Entry has invalid '{tools_field}' field: expected a list of dicts")
        responses_create_params["tools"] = _convert_tools_to_nemo_gym(tools)
    responses_create_params["parallel_tool_calls"] = False

    out: dict = {
        "responses_create_params": responses_create_params,
        "agent_ref": {
            "type": AGENT_TYPE, 
            "name": AGENT_NAME_REASONING_OFF if no_reasoning else AGENT_NAME
        },
    }
    if dataset:
        out["dataset"] = dataset

    return out


def _validate_output_path(output_path: Path) -> None:
    """Ensure *output_path* ends with ``.jsonl``, parent dir exists, and file does not already exist."""
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Output path must end with .jsonl, got: {output_path}")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Convert a JSON/JSONL dataset to genrm_compare format.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input data file (must be a .jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output JSONL file (must end with .jsonl).",
    )
    parser.add_argument(
        "--prompt-field",
        default="messages",
        help="Field name for the prompt (a string or a chat-template list of turns) (default: %(default)s).",
    )
    parser.add_argument(
        "--tools-field",
        default="tools",
        help="Field name for the tools list (default: %(default)s).",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        default=False,
        help="Use 'genrm_simple_agent_reasoning_off' as agent_ref.name instead of 'genrm_simple_agent' (default: false).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name stamped on each output row; an empty value omits the field entirely.",
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
        else args.input.with_name(args.input.stem + "_genrm_compare.jsonl")
    )
    prompt_field: str = args.prompt_field
    tools_field: str = args.tools_field
    no_reasoning: bool = args.no_reasoning
    dataset: str = args.dataset.strip()
    skip_on_error: bool = args.skip_on_error

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
                converted_entry = _convert_entry(entry, prompt_field, tools_field, no_reasoning, dataset)
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

    logger.info(f"Done - wrote {written} entries to {output_path}.")


if __name__ == "__main__":
    main()
