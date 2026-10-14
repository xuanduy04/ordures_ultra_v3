from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import multiprocessing
import os
import pprint
import struct
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_U64 = struct.Struct("<Q")


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


def _validate_output_path(output_path: Path) -> None:
    """Ensure *output_path* ends with ``.jsonl``, parent dir exists, and file does not already exist."""
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Output path must end with .jsonl, got: {output_path}")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_input_files(patterns: list[str]) -> list[Path]:
    """Resolve glob *patterns* to a sorted, deduplicated list of files."""
    # Expand all globs and deduplicate using a set, then convert to Path objects
    candidates = [Path(p) for pattern in patterns for p in glob.glob(pattern)]
    files = sorted({p for p in candidates if p.is_file()})
    if not files:
        raise FileNotFoundError(f"No input files matched patterns: {patterns}")
    return files


def _resolve_input_dirs(patterns: list[str]) -> list[Path]:
    """Resolve glob *patterns* to a sorted, deduplicated list of directories."""
    candidates = [Path(p) for pattern in patterns for p in glob.glob(pattern)]
    dirs = sorted({p for p in candidates if p.is_dir()})
    if not dirs:
        raise FileNotFoundError(f"No input directories matched patterns: {patterns}")
    return dirs


def _resolve_worker_count() -> int:
    """Return the number of worker processes to use, honouring CPU affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def hash_list_dicts(obj: list[dict[str, str]]) -> int:
    h = hashlib.blake2b(digest_size=64)

    update = h.update
    pack_u64 = _U64.pack

    update(pack_u64(len(obj)))

    for d in obj:
        update(pack_u64(len(d)))

        for key in sorted(d):
            kb = key.strip().encode("utf-8")
            value = d[key]
            vb = value.strip().encode("utf-8") if isinstance(value, str) else b""

            update(pack_u64(len(kb)))
            update(kb)
            update(pack_u64(len(vb)))
            update(vb)

    return int.from_bytes(h.digest(), "little")


def get_user_turns(obj: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the non-empty ``user`` turns of input conversation."""
    return [
        {
            "role": turn["role"].strip(),
            "content": turn["content"].strip()
        } for turn in obj
        if isinstance(turn.get("content"), str)
        and turn["content"].strip()
        and isinstance(turn.get("role"), str)
        and turn["role"].strip() == "user"
    ]


def _get_input_hashes(input_turns: list[dict]) -> tuple[int, int]:
    """Return the ``(full conversation, user turns only)`` hash pair of *input_turns*.

    Mirrors the two signatures written by the hashed response logging update:
    the full hash covers every turn as logged, the user-only hash only the
    non-empty ``user`` turns.
    """
    return hash_list_dicts(input_turns), hash_list_dicts(get_user_turns(input_turns))


def _get_trained_row_hashes(data: dict) -> tuple[list[int], list[int]]:
    """Return the ``(full conversation, user turns only)`` hash pairs logged for *data*.

    Rows written with hashed response logging carry ``full_hash``/``user_hash``
    directly; rows written without it (older commits, or a run with
    ``env.should_log_nemo_gym_responses`` but without
    ``env.should_log_hashed_nemo_gym_responses``) still carry the raw
    conversation, which is hashed on the fly with the same two signatures.
    """
    try:
        full_hashes: list[int] = []
        user_hashes: list[int] = []
        for full_result in data["full_result"]:
            if "full_hash" in full_result:
                full_hashes.append(full_result["full_hash"])
                user_hashes.append(full_result["user_hash"])
            else:
                full_hash, user_hash = _get_input_hashes(
                    full_result["responses_create_params"]["input"]
                )
                full_hashes.append(full_hash)
                user_hashes.append(user_hash)
        return full_hashes, user_hashes
    except Exception:
        pprint.pprint(data)
        raise


def _worker_process_file(
    file_path: str,
    preprocess_fn: Callable[[dict], tuple[list[int], list[int]]],
) -> list[tuple[list[int], list[int]]]:
    """Open a single JSONL file and hash each of its valid rows."""
    results: list[tuple[list[int], list[int]]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON on line {line_num} in {file_path}")
                continue

            results.append(preprocess_fn(data))
    return results


def _extract_trained_hashes_pass(
    training_step_folders: list[Path],
    step_index_range: Sequence[int] | None = None,
) -> tuple[set[int], set[int]]:
    """Collect the full-conversation and user-only signatures of every logged row of the ``train_data_step*.jsonl`` files in *training_step_folders*.

    Rows carrying the ``full_hash``/``user_hash`` keys of the hashed response
    logging update are read directly; rows still carrying the raw conversation
    are hashed on the fly with those same two signatures.

    Args:
        training_step_folders:
            Folders containing files named `train_data_step{N}.jsonl`.

        step_index_range:
            Optional sequence of step indices to process. If omitted,
            all steps from 1 through the maximum discovered step are
            considered.

    Raises:
        FileNotFoundError:
            If no training-step files are found.

        ValueError:
            If a ``train_data_step<N>.jsonl`` filename cannot be parsed.
    """
    files: list[Path] = []
    for folder in training_step_folders:
        if step_index_range is None:
            files.extend(
                sorted(
                    folder.glob("train_data_step*.jsonl"),
                    key=lambda p: int(p.stem.removeprefix("train_data_step")),
                )
            )
        else:
            files.extend(folder / f"train_data_step{i}.jsonl" for i in step_index_range)

    if not files:
        raise FileNotFoundError(f"No train_data_step*.jsonl files found in: {training_step_folders}")

    logger.info(f"Found {len(files)} training step file(s). Starting parallel processing, this may take a few minutes...")
    worker_with_args = partial(_worker_process_file, preprocess_fn=_get_trained_row_hashes)

    full_hashes: set[int] = set()
    user_hashes: set[int] = set()
    logged_row_count = 0
    with ProcessPoolExecutor(
        max_workers=_resolve_worker_count(),
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        iterator = executor.map(worker_with_args, files, chunksize=1)

        for file_results in tqdm(iterator, total=len(files), desc="Processing files"):
            logged_row_count += len(file_results)
            for row_full_hashes, row_user_hashes in file_results:
                full_hashes.update(row_full_hashes)
                user_hashes.update(row_user_hashes)

    logger.info(f"Found {len(full_hashes)} full / {len(user_hashes)} user unique trained-sample signature(s) across {logged_row_count} logged row(s).")

    return full_hashes, user_hashes


def _dataset_set_minus_pass(
    A_files: list[Path],
    full_hashes: set[int],
    user_hashes: set[int],
    output_path: Path,
) -> dict[str, int]:
    r"""Perform a set-minus operation (A \\ B) on JSONL datasets.

    Rows are compared by their full-conversation hash and their user-only hash
    (sample semantics), so rows that are the same sample serialized differently
    are treated as one and deduplicated. Invalid JSON lines in A are warned
    about and skipped.
    """
    logger.info(f"Loaded {len(full_hashes)} full / {len(user_hashes)} user unique exclusion signatures. Processing dataset(s) (A)...")

    seen_full_hashes: set[int] = set()
    seen_user_hashes: set[int] = set()
    kept_count = 0
    excluded_count = 0
    internal_dupes = 0
    skipped_count = 0

    with output_path.open("w", encoding="utf-8") as outfile:
        for a_path in tqdm(A_files, desc="A files processed", unit="file"):
            with a_path.open("r", encoding="utf-8") as infile:
                for line_num, line in tqdm(
                    enumerate(infile, 1), desc=f"Processing {a_path.name}", unit="line", leave=False
                ):
                    line_str = line.strip()
                    if not line_str:
                        continue

                    try:
                        line_dict = json.loads(line_str)
                    except json.JSONDecodeError:
                        skipped_count += 1
                        logger.warning(f"Skipping invalid JSON on line {line_num} in {a_path}. It was not copied.")
                        continue

                    line_full_hash, line_user_hash = _get_input_hashes(
                        line_dict["responses_create_params"]["input"]
                    )
                    if line_full_hash in full_hashes or line_user_hash in user_hashes:
                        excluded_count += 1
                        continue
                    if line_full_hash in seen_full_hashes or line_user_hash in seen_user_hashes:
                        internal_dupes += 1
                        continue

                    seen_full_hashes.add(line_full_hash)
                    seen_user_hashes.add(line_user_hash)
                    outfile.write(line_str + "\n")
                    kept_count += 1

    return {
        "unique_lines_kept": kept_count,
        "excluded": excluded_count,
        "internal_dupes": internal_dupes,
        "invalid_json_skipped": skipped_count,
    }


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Extract the untrained subset of dataset(s) A by excluding samples seen in training-log folder(s) B.",
    )
    parser.add_argument(
        "--a",
        nargs="+",
        required=True,
        type=str,
        help="Input JSONL file(s) or glob pattern(s) for dataset A.",
    )
    parser.add_argument(
        "--b",
        nargs="+",
        required=True,
        type=str,
        help="Input folder(s) or glob pattern(s) for training log B, each containing train_data_step<N>.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/get_untrained_data.jsonl"),
        help="Path to the output JSONL file (default: %(default)s).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )

    args = parser.parse_args()
    A_patterns: list[str] = args.a
    B_patterns: list[str] = args.b
    output_path: Path = args.output.resolve()

    try:
        A_files = _resolve_input_files(A_patterns)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    try:
        B_folders = _resolve_input_dirs(B_patterns)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if output_path in [f.resolve() for f in A_files]:
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
                logger.info("Existing output file will not be overwritten; exiting.")
                sys.exit(0)

    try:
        full_hashes, user_hashes = _extract_trained_hashes_pass(B_folders)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    logger.info(f"Computing A \\ B from:\n\t{len(A_files):>6} A file(s) and\n\t{len(full_hashes):>6} full / {len(user_hashes):>6} user B trained sample(s)\n\t-> {output_path}")

    stats = _dataset_set_minus_pass(A_files, full_hashes, user_hashes, output_path)

    logger.info(f"Done - set-minus complete:\n{pprint.pformat(stats)}.")
    logger.info(f"Resulting dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
