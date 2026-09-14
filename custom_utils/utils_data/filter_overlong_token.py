from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BUF_SIZE = 1024 * 1024

CHUNK_SIZE = 256
INFLIGHT_TASKS_PER_WORKER = 4

DATA_TYPE_SFT = "SFT"
DATA_TYPE_RL_QA = "RL_QA"
DATA_TYPE_NEMO_GYM = "nemo_gym"
DATA_TYPES = [DATA_TYPE_SFT, DATA_TYPE_RL_QA, DATA_TYPE_NEMO_GYM]

G_TOKENIZER: Any = None
G_DATA_TYPE: str = ""
G_MAX_TOKEN: int = 0


def _preprocess_underscore_args(argv: list[str]) -> list[str]:
    """Replace underscores with hyphens in ``--arg_name`` / ``--arg_name=val`` prefixes.

    argparse normalises ``-`` and ``_`` internally, so ``--max-token`` and
    ``--max_token`` would normally map to different destinations.  This
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


def _resolve_max_token(value: int) -> int:
    """Resolve the ``--max-token`` shorthand into an absolute token count.

    Values divisible by 1000 are used directly; any other value is interpreted
    as thousands of tokens (e.g. ``6`` becomes ``6000``).
    """
    if value <= 0:
        raise ValueError(f"--max-token must be a positive integer, got: {value}")
    return value if value % 1000 == 0 else value * 1000


def _extract_main_field(entry: dict, data_type: str) -> Any:
    """Return the main field measured for *data_type*: ``messages``, ``question``, or the gym input."""
    try:
        if data_type == DATA_TYPE_SFT:
            return entry["messages"]
        if data_type == DATA_TYPE_RL_QA:
            return entry["question"]
        if data_type == DATA_TYPE_NEMO_GYM:
            return entry["responses_create_params"]["input"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing or invalid main field for data type {data_type}: {exc!r}") from exc
    raise ValueError(f"Unsupported data type: {data_type}")


def _init_worker(tokenizer_dir: str, data_type: str, max_token: int) -> None:
    """Load the tokenizer once per worker process and store the filter settings."""
    global G_TOKENIZER, G_DATA_TYPE, G_MAX_TOKEN
    G_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_dir)
    G_DATA_TYPE = data_type
    G_MAX_TOKEN = max_token


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


def _process_chunk(chunk: tuple[int, list[str]]) -> tuple[list[str], int, int, int]:
    """Filter one chunk of raw JSONL lines.

    Returns the kept lines along with the total number of lines examined, the
    number of overlong entries dropped, and the number of unreadable entries
    skipped.
    """
    start_lineno, lines = chunk
    keep = [False] * len(lines)
    string_positions: list[int] = []
    string_fields: list[str] = []
    conversation_positions: list[int] = []
    conversation_fields: list[list[dict]] = []
    n_errors = 0

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
                keep[offset] = count <= G_MAX_TOKEN

    if conversation_fields:
        conversation_counts = _count_conversation_tokens(conversation_fields)
        for offset, count in zip(conversation_positions, conversation_counts):
            if count < 0:
                n_errors += 1
                logger.warning(f"Skipping entry at line {start_lineno + offset}: chat-template tokenization failed.")
            else:
                keep[offset] = count <= G_MAX_TOKEN

    kept_lines = [lines[i] for i, keep_line in enumerate(keep) if keep_line]
    n_dropped = len(string_positions) + len(conversation_positions) - len(kept_lines)
    return kept_lines, len(lines), n_dropped, n_errors


def _resolve_worker_count() -> int:
    """Return the number of worker processes to use, honouring CPU affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _count_lines(input_path: Path) -> int:
    """Fast pass: count the number of lines in *input_path*."""
    with input_path.open("rb") as f:
        lines = 0
        last_byte = b""
        read_f = f.raw.read
        # Wrap in tqdm; Note: total is unknown initially, so it will show iterations/sec
        with tqdm(unit=" line", desc=f"Counting {input_path.name}") as pbar:
            buf = read_f(BUF_SIZE)
            while buf:
                newlines = buf.count(b"\n")
                lines += newlines
                last_byte = buf[-1:]
                pbar.update(newlines)
                buf = read_f(BUF_SIZE)
        if last_byte and last_byte != b"\n":
            lines += 1
    return lines


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


def _filter_pass(
    input_path: Path,
    output_path: Path,
    total_lines: int,
    tokenizer_dir: Path,
    data_type: str,
    max_token: int,
) -> dict[str, int]:
    """Run the multiprocessing filter pass and write kept lines to *output_path*.

    Chunks are submitted lazily and at most
    ``workers * INFLIGHT_TASKS_PER_WORKER`` tasks stay in flight, keeping memory
    bounded on large inputs.  Kept lines are written in completion order, which
    shuffles the dataset.
    """
    worker_count = _resolve_worker_count()
    inflight_limit = worker_count * INFLIGHT_TASKS_PER_WORKER
    chunks = _iter_chunks(input_path, CHUNK_SIZE)

    n_kept = 0
    n_dropped = 0
    n_errors = 0

    pool = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_init_worker,
        initargs=(str(tokenizer_dir), data_type, max_token),
    )

    with pool, output_path.open("w", encoding="utf-8") as fout, tqdm(
        total=total_lines, desc="Filtering", unit="line"
    ) as pbar:
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
                kept_lines, n_lines, n_chunk_dropped, n_chunk_errors = future.result()
                for line in kept_lines:
                    fout.write(line.strip() + "\n")
                n_kept += len(kept_lines)
                n_dropped += n_chunk_dropped
                n_errors += n_chunk_errors
                pbar.update(n_lines)

    return {"kept": n_kept, "dropped_overlong": n_dropped, "skipped_unreadable": n_errors}


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Filter JSONL rows whose main field exceeds a maximum token length.",
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
        help="Path to the output JSONL file (default: '<input stem>-len<max token in thousands>k.jsonl').",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )
    parser.add_argument(
        "--max-token",
        type=int,
        required=True,
        help="Maximum token length. Values not divisible by 1000 are interpreted as thousands (e.g. 6 -> 6000).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to a local HuggingFace tokenizer directory.",
    )
    parser.add_argument(
        "--data-type",
        choices=DATA_TYPES,
        required=True,
        help="Data type of the input rows, selecting the main field to measure.",
    )

    args = parser.parse_args()

    input_path: Path = Path(args.input).resolve()
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    tokenizer_dir: Path = args.tokenizer.resolve()
    if not tokenizer_dir.is_dir():
        parser.error(f"Tokenizer directory does not exist: {tokenizer_dir}")
    try:
        AutoTokenizer.from_pretrained(str(tokenizer_dir))
    except Exception as exc:
        parser.error(f"Failed to load tokenizer from {tokenizer_dir}: {exc}")

    try:
        max_token = _resolve_max_token(args.max_token)
    except ValueError as exc:
        parser.error(str(exc))

    output_path: Path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}-len{max_token // 1000}k.jsonl")
    )

    if output_path == input_path:
        parser.error(f"Input and output paths must differ, got: {output_path}")

    try:
        _validate_output_path(output_path)
    except ValueError as exc:
        parser.error(str(exc))
    except FileExistsError:
        if not args.yes:
            try:
                response = input(f"{output_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
            except EOFError:
                response = ""
            if response not in ("y", "yes"):
                logger.info("Existing output file will not be overridden; exiting.")
                sys.exit(0)

    logger.info(f"Filtering {input_path} to at most {max_token} tokens ({args.data_type}) -> {output_path}")

    total_lines = _count_lines(input_path)
    if total_lines == 0:
        parser.error("Input dataset is empty.")
    logger.info(f"Found {total_lines} line(s) in '{input_path.name}'.")

    stats = _filter_pass(input_path, output_path, total_lines, tokenizer_dir, args.data_type, max_token)

    logger.info(
        f"Done - kept {stats['kept']} of {total_lines} entries, dropped {stats['dropped_overlong']} overlong, "
        f"skipped {stats['skipped_unreadable']} unreadable; wrote to {output_path}."
    )


if __name__ == "__main__":
    main()
