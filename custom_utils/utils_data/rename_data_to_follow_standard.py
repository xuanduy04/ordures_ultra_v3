from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESOURCES_SERVERS_DIR = (
    Path(__file__).resolve().parents[2] / "3rdparty" / "Gym-workspace" / "Gym" / "resources_servers"
)

LEN_FIELD_PATTERN = re.compile(r"len(\d+)k")
SAM_FIELD_PATTERN = re.compile(r"sam(\d+)k")

CHUNK_SIZE = 256
INFLIGHT_TASKS_PER_WORKER = 4

DATA_TYPE_SFT = "SFT"
DATA_TYPE_RL_QA = "RL_QA"

G_TOKENIZER: Any = None
G_DATA_TYPE: str = ""


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


def _load_valid_data_types() -> set[str] | None:
    """Collect the valid data type names.

    The fixed types ``SFT``, ``RL_QA``, and ``nemo_gym`` are always valid.
    NeMo-Gym data containing exactly 1 environment uses that environment's
    name, so every subdirectory of the Gym ``resources_servers`` directory is
    accepted as well.  If that directory is unavailable, ``None`` is returned
    to signal that any trailing filename field is accepted as a data type.
    """
    valid_data_types = {"SFT", "RL_QA", "nemo_gym"}
    if not RESOURCES_SERVERS_DIR.is_dir():
        logger.warning(
            f"Gym resources_servers directory not found at {RESOURCES_SERVERS_DIR}; "
            "accepting any trailing filename field as the data type."
        )
        return None
    valid_data_types.update(
        entry.name for entry in RESOURCES_SERVERS_DIR.iterdir() if entry.is_dir()
    )
    return valid_data_types


def _consume_metadata_fields(
    fields: list[str], idx: int, len_field: str | None
) -> tuple[int, str | None]:
    """Walk left from *idx*, consuming ``len``/``sam`` metadata fields.

    Returns the index of the first non-metadata field (or ``-1``) and the
    rightmost ``len`` field encountered, normalised to ``len<value>k``.
    """
    while idx >= 0:
        len_match = LEN_FIELD_PATTERN.fullmatch(fields[idx])
        if len_match is not None:
            if len_field is None:
                len_field = f"len{max(1, int(len_match.group(1)))}k"
            idx -= 1
            continue
        if SAM_FIELD_PATTERN.fullmatch(fields[idx]) is not None:
            idx -= 1
            continue
        break
    return idx, len_field


def _parse_filename_right_to_left(
    stem: str, valid_data_types: set[str] | None
) -> tuple[str, str | None, str | None]:
    """Parse a filename stem right-to-left into ``(content, len_field, data_type)``.

    ``sam`` metadata is always discarded (the script recomputes it) and ``len``
    metadata is kept (rightmost wins).  The rightmost non-metadata field is the
    data type when it is in *valid_data_types* (``None`` accepts anything);
    otherwise it is treated as part of the content.  ``data_type`` is ``None``
    when the filename carries no valid data type.
    """
    fields = stem.split("-")

    idx, len_field = _consume_metadata_fields(fields, len(fields) - 1, None)
    if idx < 0:
        return "", len_field, None

    dtype_field = fields[idx]
    if valid_data_types is not None and dtype_field not in valid_data_types:
        return "-".join(fields[: idx + 1]), len_field, None

    idx, len_field = _consume_metadata_fields(fields, idx - 1, len_field)
    if idx < 0:
        return "", len_field, dtype_field

    return "-".join(fields[: idx + 1]), len_field, dtype_field


def _count_lines_python(input_path: Path) -> int:
    """Fallback line counter used when the native `wc` command is unavailable."""
    # Buffer size (in bytes) for the fallback line-counting pass when `wc` is unavailable.
    # As this is already empirically optimal and does not require user input, 
    # we place it here 
    BUF_SIZE = 1024 * 1024

    lines = 0
    with input_path.open("rb") as f:
        while buf := f.read(BUF_SIZE):
            lines += buf.count(b"\n")
    return lines


def _count_lines_wc(input_path: Path) -> int:
    """Count newline characters using the native `wc -l` implementation."""
    with input_path.open("rb") as f:
        result = subprocess.run(
            ["wc", "-l"],
            stdin=f,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

    return int(result.stdout.strip())


def _count_lines(input_path: Path) -> int:
    """First pass: count the number of lines in the input file."""
    if shutil.which("wc") is None:
        logger.warning("`wc` was not found on PATH; falling back to Python line counting.")
        return _count_lines_python(input_path)

    try:
        return _count_lines_wc(input_path)
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        logger.warning(f"Failed to count '{input_path}' with `wc -l` ({exc}); falling back to Python.")
        return _count_lines_python(input_path)


def _extract_main_field(entry: dict, data_type: str) -> Any:
    """Return the main field measured for *data_type*: ``messages``, ``question``, or the gym input."""
    try:
        if data_type == DATA_TYPE_SFT:
            return entry["messages"]
        if data_type == DATA_TYPE_RL_QA:
            return entry["question"]
        return entry["responses_create_params"]["input"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing or invalid main field for data type {data_type}: {exc!r}") from exc


def _init_worker(tokenizer_dir: str, data_type: str) -> None:
    """Load the tokenizer once per worker process and store the counting settings."""
    global G_TOKENIZER, G_DATA_TYPE
    G_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_dir)
    G_DATA_TYPE = data_type


def _count_string_tokens(texts: list[str]) -> list[int]:
    """Return the token count of each plain string in *texts*."""
    encoded = G_TOKENIZER(texts, add_special_tokens=False)
    return [len(ids) for ids in encoded["input_ids"]]


def _count_conversation_tokens(conversations: list[list[dict]]) -> list[int]:
    """Return the chat-template token count of each conversation in *conversations*.

    A failed batch is retried one conversation at a time so that a single bad
    row does not discard its whole chunk; failed rows are reported as ``-1``.
    """
    try:
        encoded = G_TOKENIZER.apply_chat_template(conversations, tokenize=True, return_dict=True)
        return [len(ids) for ids in encoded["input_ids"]]
    except Exception as exc:
        logger.warning(f"Batched chat-template tokenization failed ({exc}); retrying per conversation.")
        counts: list[int] = []
        for conversation in conversations:
            try:
                ids = G_TOKENIZER.apply_chat_template(conversation, tokenize=True, return_dict=True)["input_ids"]
                counts.append(len(ids))
            except Exception:
                counts.append(-1)
        return counts


def _process_chunk(chunk: tuple[int, list[str]]) -> tuple[int, int, int]:
    """Count the main-field tokens of one chunk of raw JSONL lines.

    Returns the number of lines examined, the maximum token count seen, and
    the number of unreadable entries skipped.
    """
    start_lineno, lines = chunk
    string_positions: list[int] = []
    string_fields: list[str] = []
    conversation_positions: list[int] = []
    conversation_fields: list[list[dict]] = []
    n_errors = 0
    max_tokens = 0

    for offset, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        lineno = start_lineno + offset
        try:
            entry = json.loads(stripped)
            field = _extract_main_field(entry, G_DATA_TYPE)
        except Exception as exc:
            n_errors += 1
            logger.warning(f"Skipping unreadable entry at line {lineno}: {exc}")
            continue
        if isinstance(field, str):
            string_positions.append(offset)
            string_fields.append(field)
        elif isinstance(field, (list, dict)):
            conversation_positions.append(offset)
            conversation_fields.append(field if isinstance(field, list) else [field])
        else:
            n_errors += 1
            logger.warning(f"Skipping entry at line {lineno}: main field has unsupported type {type(field).__name__}.")

    if string_fields:
        try:
            string_counts = _count_string_tokens(string_fields)
        except Exception as exc:
            string_counts = [-1] * len(string_fields)
            logger.warning(f"Failed to tokenize {len(string_fields)} plain-text field(s) near line {start_lineno}: {exc}")
        for offset, count in zip(string_positions, string_counts):
            if count < 0:
                n_errors += 1
                logger.warning(f"Skipping entry at line {start_lineno + offset}: tokenization failed.")
            else:
                max_tokens = max(max_tokens, count)

    if conversation_fields:
        conversation_counts = _count_conversation_tokens(conversation_fields)
        for offset, count in zip(conversation_positions, conversation_counts):
            if count < 0:
                n_errors += 1
                logger.warning(f"Skipping entry at line {start_lineno + offset}: chat-template tokenization failed.")
            else:
                max_tokens = max(max_tokens, count)

    return len(lines), max_tokens, n_errors


def _resolve_worker_count() -> int:
    """Return the number of worker processes to use, honouring CPU affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _iter_chunks(input_path: Path, chunk_size: int) -> Iterator[tuple[int, list[str]]]:
    """Yield ``(start_lineno, lines)`` chunks lazily from *input_path*."""
    with input_path.open("r", encoding="utf-8") as fin:
        start_lineno = 1
        chunk: list[str] = []
        for line in fin:
            chunk.append(line)
            if len(chunk) == chunk_size:
                yield start_lineno, chunk
                start_lineno += len(chunk)
                chunk = []
        if chunk:
            yield start_lineno, chunk


def _max_token_pass(
    input_path: Path,
    total_lines: int | None,
    tokenizer_dir: Path,
    data_type: str,
) -> tuple[int, int]:
    """Run the multiprocessing token-counting pass and return the maximum main-field token count.

    Chunks are submitted lazily and at most
    ``workers * INFLIGHT_TASKS_PER_WORKER`` tasks stay in flight, keeping memory
    bounded on large inputs.
    """
    worker_count = 64
    inflight_limit = worker_count * INFLIGHT_TASKS_PER_WORKER
    chunks = _iter_chunks(input_path, CHUNK_SIZE)

    max_tokens = 0
    n_errors = 0

    pool = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_init_worker,
        initargs=(str(tokenizer_dir), data_type),
    )

    with pool, tqdm(total=total_lines, desc="Counting tokens", unit="line") as pbar:
        pending: set[Future] = set()
        exhausted = False
        while pending or not exhausted:
            if not exhausted and len(pending) < inflight_limit:
                try:
                    pending.add(pool.submit(_process_chunk, next(chunks)))
                    continue
                except StopIteration:
                    exhausted = True
                    continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                n_lines, chunk_max, chunk_errors = future.result()
                max_tokens = max(max_tokens, chunk_max)
                n_errors += chunk_errors
                pbar.update(n_lines)

    return max_tokens, n_errors


def _round_to_thousands(value: int, round_up: bool = False) -> int:
    """Round *value* to the nearest thousand, representing values below 1k as 1k."""
    if round_up:
        return max(1, (value + 999) // 1000)
    return max(1, (value + 500) // 1000)


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Rename a JSONL file inplace to follow the dataset filename standard.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input JSONL file (must end with .jsonl).",
    )
    parser.add_argument(
        "--data-type",
        default=None,
        help="Data type for the renamed file (e.g. SFT, RL_QA, nemo_gym, or a NeMo-Gym "
        "environment name). Overrides the data type parsed from the filename, and is "
        "required when the filename has none (default: parse from the filename).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Path to a local HuggingFace tokenizer directory. Only used to compute "
        "the len metadata when the filename has none (default: omit `len` metadata).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the target file (default: false).",
    )

    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    data_type_arg: str | None = args.data_type

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    if input_path.suffix.lower() != ".jsonl":
        parser.error(f"Input path must end with .jsonl, got: {input_path}")

    if data_type_arg is not None:
        data_type_arg = data_type_arg.strip()
        if not data_type_arg:
            parser.error("--data-type is empty after stripping")
        if "-" in data_type_arg:
            parser.error(f"--data-type must not contain '-', got: {data_type_arg}")

    valid_data_types = _load_valid_data_types()
    content, len_field, filename_data_type = _parse_filename_right_to_left(
        input_path.stem, valid_data_types
    )

    if not content:
        parser.error(
            f"Could not determine a content segment from filename '{input_path.name}'; "
            "expected '<content>[-<metadata>]-<data type>'."
        )

    if data_type_arg is not None:
        data_type = data_type_arg
    elif filename_data_type is not None:
        data_type = filename_data_type
    else:
        parser.error(
            f"Could not determine the data type from filename '{input_path.name}'; "
            "pass --data-type to specify it (e.g. --data-type SFT)."
        )

    logger.info(
        f"Parsed '{input_path.name}': content='{content}', "
        f"len={len_field if len_field is not None else 'none'}, data type='{data_type}'."
    )

    line_count = _count_lines(input_path)
    if line_count == 0:
        parser.error("Empty input file.")

    sample_count = _round_to_thousands(line_count)
    logger.info(f"Counted {line_count} line(s) -> sam{sample_count}k.")

    if len_field is None:
        if args.tokenizer is not None:
            tokenizer_dir: Path = args.tokenizer.resolve()
            if not tokenizer_dir.is_dir():
                parser.error(f"Tokenizer directory does not exist: {tokenizer_dir}")
            try:
                AutoTokenizer.from_pretrained(str(tokenizer_dir))
            except Exception as exc:
                parser.error(f"Failed to load tokenizer from {tokenizer_dir}: {exc}")

            max_tokens, n_errors = _max_token_pass(input_path, line_count, tokenizer_dir, data_type)
            if max_tokens == 0:
                parser.error(
                    f"Could not tokenize any entry in '{input_path.name}' ({n_errors} error(s))."
                )

            len_field = f"len{_round_to_thousands(max_tokens, round_up=True)}k"
            logger.info(f"Counted max main-field token length {max_tokens} -> {len_field}.")
        else:
            logger.warning(
                f"'{input_path.name}' has no len metadata and no --tokenizer was given; omitting len."
            )

    stem_parts = [content]
    if len_field is not None:
        stem_parts.append(len_field)
    stem_parts.append(f"sam{sample_count}k")
    stem_parts.append(data_type)
    target_path = input_path.with_name("-".join(stem_parts) + input_path.suffix)

    if target_path == input_path:
        logger.info(f"'{input_path.name}' already follows the filename standard; nothing to do.")
        return

    if target_path.exists() and not args.yes:
        response = input(f"{target_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
        if response not in ("y", "yes"):
            logger.info("Existing target file will not be overridden; exiting.")
            sys.exit(0)

    input_path.rename(target_path)
    logger.info(f"Done - renamed '{input_path.name}' -> '{target_path.name}'.")


if __name__ == "__main__":
    main()
