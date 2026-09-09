from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
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


def _convert_entry(entry: dict, prompt_field: str, tools_field: str, no_reasoning: bool, dataset: str, principle: str) -> dict:
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
    if principle:
        out["principle"] = principle

    return out


def _load_principle_script(script_path: Path) -> ModuleType:
    """Import *script_path* as a module and validate its ``get_principle`` contract.

    Raises:
        RuntimeError: If the module cannot be imported or does not define a
            callable ``get_principle``.
    """
    try:
        spec = importlib.util.spec_from_file_location("genrm_principle_script", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create a module spec for '{script_path}'")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"Failed to import principle script '{script_path}': {exc}") from exc

    if not callable(getattr(module, "get_principle", None)):
        raise RuntimeError(
            f"Principle script '{script_path}' must define a callable 'get_principle(entry: dict) -> str'"
        )

    return module


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
    principle_group = parser.add_mutually_exclusive_group()
    principle_group.add_argument(
        "--principle-string",
        default=None,
        help="Literal principle stamped on each output row (used by the GenRM judge when use_principle is enabled).",
    )
    principle_group.add_argument(
        "--principle-file",
        type=Path,
        default=None,
        help="Path to a text file whose content is used as the principle stamped on each output row.",
    )
    principle_group.add_argument(
        "--principle-script",
        type=Path,
        default=None,
        help="Path to a .py principle script. The script must define a callable "
        "'get_principle(entry: dict) -> str'. It is imported once and get_principle is called for "
        "every input entry with the parsed JSON entry as its only argument; the returned string "
        "(after .strip()) becomes that row's principle. A missing file, an import error, or a "
        "missing get_principle always aborts the run; exceptions or invalid/empty return values "
        "per entry are treated as row failures (honored by --skip-on-error).",
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
        else args.input.with_name(args.input.stem + "-genrm_compare.jsonl")
    )
    prompt_field: str = args.prompt_field
    tools_field: str = args.tools_field
    no_reasoning: bool = args.no_reasoning
    dataset: str = args.dataset.strip()
    skip_on_error: bool = args.skip_on_error

    principle: str = ""
    principle_getter: Any = None
    principle_script_path: Path | None = None
    if args.principle_string is not None:
        principle = args.principle_string.strip()
        if not principle:
            parser.error("--principle-string is empty after stripping")
    elif args.principle_file is not None:
        principle_path = args.principle_file
        if not principle_path.is_file():
            parser.error(f"--principle-file must be an existing file, got: {principle_path}")
        principle = principle_path.read_text(encoding="utf-8").strip()
        if not principle:
            parser.error("--principle-file is empty after stripping")
    elif args.principle_script is not None:
        principle_script_path = args.principle_script
        if not principle_script_path.is_file():
            parser.error(f"--principle-script must be an existing file, got: {principle_script_path}")
        if principle_script_path.suffix != ".py":
            parser.error(f"--principle-script must end with .py, got: {principle_script_path}")
        if not principle_script_path.read_text(encoding="utf-8").strip():
            parser.error(f"--principle-script is empty: {principle_script_path}")
        principle_getter = _load_principle_script(principle_script_path).get_principle

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
                row_principle = principle
                if principle_getter is not None:
                    result = principle_getter(entry)
                    if not isinstance(result, str):
                        raise ValueError(
                            f"Principle script '{principle_script_path}' get_principle() returned "
                            f"{type(result).__name__}, expected str"
                        )
                    row_principle = result.strip()
                    if not row_principle:
                        raise ValueError(
                            f"Principle script '{principle_script_path}' returned an empty principle"
                        )
                converted_entry = _convert_entry(entry, prompt_field, tools_field, no_reasoning, dataset, row_principle)
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
