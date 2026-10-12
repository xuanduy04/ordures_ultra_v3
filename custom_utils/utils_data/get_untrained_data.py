from __future__ import annotations

import argparse
import glob
import json
import logging
import multiprocessing
import os
import pprint
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable

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


def _canonical_json_key(obj: object) -> str:
    """Canonical JSON serialization used to compare hashes order-insensitively."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _get_input_hash(input_turns: list[dict]) -> list[dict]:
    """Return the non-empty ``user`` turns of *input_turns*, used as a sample's identity."""
    return  [
        {
            "role": turn["role"].strip(),
            "content": turn["content"].strip()
        } for turn in input_turns
        if len(turn.get("content", "")) > 0 and turn.get("role", "N/A").strip() in ["user"]
    ]


def _get_trained_dataset_hash(data: dict) -> str:
    # # data.keys()=dict_keys(['agent_ref', 'content', 'rewards', 'input_lengths', 'token_ids', 'token_loss_mask', 'sample_loss_mask', 'advantages', 'generation_logprobs', 'prev_logprobs', 'full_result', 'idx'])
    # print(data["agent_ref"])
    # print(data["full_result"][0]["responses_create_params"]['input'])
    # exit(0)
    try:
        return _canonical_json_key({
            "agent_ref": {"name": data["agent_ref"][0]["name"]},
            "responses_create_params": {
                "input": _get_input_hash(data["full_result"][0]["responses_create_params"]["input"]),
                # "tools": data["full_result"][0]["responses_create_params"].get("tools", []),
            }
        })
    except Exception:
        pprint.pprint(data["full_result"][0])
        raise


def _get_nemo_gym_dataset_hash(data: dict) -> str:
    try:
        return _canonical_json_key({
            "agent_ref": {"name": data["agent_ref"]["name"]},
            "responses_create_params": {
                "input": _get_input_hash(data["responses_create_params"]["input"]),
                # "tools": data["responses_create_params"].get("tools", []),
            },
        })
    except Exception:
        pprint.pprint(data)
        raise


def _worker_process_file(file_path: str, preprocess_fn: Callable[[dict], str]) -> list[str]:
    """Open a single JSONL file and hash each of its valid rows."""
    results: list[str] = []
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
) -> set[str]:
    """Hash every logged row of the ``train_data_step*.jsonl`` files in *training_step_folders*.

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
    worker_with_args = partial(_worker_process_file, preprocess_fn=_get_trained_dataset_hash)

    all_hashes: list[str] = []
    with ProcessPoolExecutor(
        max_workers=_resolve_worker_count(),
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        iterator = executor.map(worker_with_args, files, chunksize=1)

        for file_results in tqdm(iterator, total=len(files), desc="Processing files"):
            all_hashes.extend(file_results)

    unique_hashes = set(all_hashes)
    logger.info(f"Found {len(unique_hashes)} unique trained-sample signature(s) across {len(all_hashes)} logged row(s).")

    return unique_hashes


def _dataset_set_minus_pass(A_files: list[Path], exclude_data: set[str], output_path: Path) -> dict[str, int]:
    r"""Perform a set-minus operation (A \\ B) on JSONL datasets.

    Rows are compared by their NeMo-Gym dataset hash (sample semantics), so
    rows that are the same sample serialized differently are treated as one
    and deduplicated. Invalid JSON lines in A are warned about and skipped.
    """
    logger.info(f"Loaded {len(exclude_data)} unique exclusion signatures. Processing dataset(s) (A)...")

    seen_in_A: set[str] = set()
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

                    line_hash = _get_nemo_gym_dataset_hash(line_dict)
                    if line_hash in exclude_data:
                        excluded_count += 1
                        continue
                    if line_hash in seen_in_A:
                        internal_dupes += 1
                        continue

                    seen_in_A.add(line_hash)
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
        exclude_data = _extract_trained_hashes_pass(B_folders)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    logger.info(f"Computing A \\ B from:\n\t{len(A_files):>6} A file(s) and\n\t{len(exclude_data):>6} B trained sample(s)\n\t-> {output_path}")

    stats = _dataset_set_minus_pass(A_files, exclude_data, output_path)

    logger.info(f"Done - set-minus complete:\n{pprint.pformat(stats)}.")
    logger.info(f"Resulting dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
