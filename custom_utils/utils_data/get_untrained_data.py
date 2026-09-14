from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pprint
import signal
import sys

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


STAT_COLUMNS: tuple[str] = ("advantages", "rewards")


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


def _resolve_input_files(patterns: list[str]) -> list[Path]:
    """Resolve glob *patterns* to a sorted, deduplicated list of files."""
    # Expand all globs and deduplicate using a set, then convert to Path objects
    candidates = [Path(p) for pattern in patterns for p in glob.glob(pattern)]
    files = sorted({p for p in candidates if p.is_file()})
    if not files:
        raise FileNotFoundError(f"No input files matched patterns: {patterns}")
    return files


def _validate_output_path(output_path: Path) -> None:
    """Ensure *output_path* ends with ``.jsonl``, parent dir exists, and file does not already exist."""
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Output path must end with .jsonl, got: {output_path}")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)


def _worker_process_file(file_path: str, preprocess_fn: Callable[[str], str]) -> list[str]:
    """Opens a single JSONL file and processes it."""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                try:
                    data = json.loads(line)
                    processed = preprocess_fn(data)
                    results.append(processed)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_num} in {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        
    return list(results)

def get_trained_dataset_hash(data: dict) -> str:
    # # data.keys()=dict_keys(['agent_ref', 'content', 'rewards', 'input_lengths', 'token_ids', 'token_loss_mask', 'sample_loss_mask', 'advantages', 'generation_logprobs', 'prev_logprobs', 'full_result', 'idx'])
    # print(data["agent_ref"])
    # print(data["full_result"][0]["responses_create_params"]['input'])
    # exit(0)
    try:
        return str({
            "agent_ref": {"name": data["agent_ref"][0]["name"]},
            "responses_create_params": {
                "input": data["full_result"][0]["responses_create_params"]["input"],
                "tools": data["full_result"][0]["responses_create_params"].get("tools", []),
            }
        })
    except Exception as e:
        pprint.pprint(data["full_result"][0])
        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)

def get_nemo_gym_dataset_hash(data: dict) -> str:
    return str({
        "agent_ref": {"name": data["agent_ref"]["name"]},
        "responses_create_params": {
            "input": data["responses_create_params"]["input"],
            # "tools": data["responses_create_params"].get("tools", []),
        },
    })

def extract_trained_dataset_hash(
    training_step_folder: str | Path,
    agent_list: list | set | None = None,
    step_index_range: Sequence[int] | None = None,
) -> set[str]:
    """Load, filter, and concatenate training-step JSONL datasets, writes to output_path.

    Args:
        training_step_folder:
            Folder containing files named `train_data_step{N}.jsonl`.

        agent_list:
            Optional collection of valid agent names. Only rows whose
            first `agent_ref` entry has a matching `name` are retained.

        step_index_range:
            Optional sequence of step indices to process. If omitted,
            all steps from 1 through the maximum discovered step are
            considered.

    Raises:
        FileNotFoundError:
            If no training-step files are found.

        ValueError:
            If step filenames cannot be parsed, the step range is invalid,
            or no matching records are found.
    """
    folder = Path(training_step_folder)
    agents = set(agent_list) if agent_list is not None else None

    if step_index_range is None:
        files = sorted(
            folder.glob("train_data_step*.jsonl"),
            key=lambda p: int(p.stem.removeprefix("train_data_step")),
        )
    else:
        files = [folder / f"train_data_step{i}.jsonl" for i in step_index_range]

    print(f"Found {len(files)} training step(s) in training log folder.\nStarting parallel processing, this may take a few minutes...")
    worker_with_args = partial(_worker_process_file, preprocess_fn=get_trained_dataset_hash)

    all_outputs: list[str] = []
    with ProcessPoolExecutor() as executor:
        # 3. Map the file list to the partial function
        iterator = executor.map(worker_with_args, files, chunksize=1)
        
        for file_results in tqdm(iterator, total=len(files), desc="Processing files"):
            all_outputs.extend(file_results)

    print(f'Total "trained" samples from logs:           {len(all_outputs):>6}')
    all_outputs = set(all_outputs)
    print(f'Total "trained" samples after deduplication: {len(all_outputs):>6}')

    return all_outputs


def dataset_set_minus(A_files: list[Path], exclude_data: set[str], output_path: Path) -> dict[str, int]:
    r"""Perform a set-minus operation (A \\ B) on JSONL datasets.

    Lines are compared as raw stripped strings; invalid JSON lines in B are
    silently skipped, invalid JSON lines in A are warned about and skipped.
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

                    line_hash = get_nemo_gym_dataset_hash(line_dict)
                    if line_hash in exclude_data:
                        excluded_count += 1
                        continue
                    if line_str in seen_in_A:
                        internal_dupes += 1
                        continue

                    seen_in_A.add(line_str)
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
        description="Extract the untrained data from a given dataset(s) (A) trainining log (B).\n**NOTE**: Not full proof, function is slightly overly kill and slightly filters more than what is optimal",
    )
    parser.add_argument(
        "--a",
        nargs="+",
        required=True,
        type=str,
        help="Input JSONL file(s) or glob pattern(s) for dataset group A.",
    )
    parser.add_argument(
        "--b",
        nargs="+",
        required=True,
        type=Path,
        help="Input folder or glob pattern(s) for training log (folder) B.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./untrained_data.jsonl"),
        help="Path to the output JSONL file (default: %(default)s).",
    )

    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Preemtively agree to everything, even the devil's proposal to buy your soul",
    )

    args = parser.parse_args()
    A_patterns: list[str] = args.a
    B_folder: Path = args.b.resolve()
    output_path: Path = args.output_path.resolve()

    try:
        A_files = _resolve_input_files(A_patterns)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    try:
        _validate_output_path(output_path)
    except FileExistsError:
        if not args.yes:
            response = input(f"{output_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Existing output file will not be overridden; exiting.")
                sys.exit(0)
    
    exclude_data = extract_trained_dataset_hash(B_folder)
    logger.info(f"Computing A \\ B from:\n\t{len(A_files):>6} A file(s) and\n\t{len(exclude_data):>6} B trained samples(s)\n\t-> {output_path}")

    stats = dataset_set_minus(A_files, exclude_data, output_path)

    logger.info(f"Done - set-minus complete:\n{pprint.pformat(stats)}.")
    logger.info(f"Resulting dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
