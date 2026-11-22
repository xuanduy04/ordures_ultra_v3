from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
from collections.abc import Iterator
from heapq import heapify, heappop, heappush
from pathlib import Path

from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _preprocess_underscore_args(argv: list[str]) -> list[str]:
    """Replace underscores with hyphens in ``--arg_name`` / ``--arg_name=val`` prefixes.

    argparse normalises ``-`` and ``_`` internally, so ``--output-path`` and ``--output_path`` would normally map to
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
            ["wc", "-l"], stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
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


def _resolve_group_files(input_dir: Path) -> list[tuple[str, list[Path]]]:
    """Resolve *input_dir* into ``(group name, JSONL files)`` pairs, one group per subdirectory."""
    groups: list[tuple[str, list[Path]]] = []
    for directory in sorted(input_dir.iterdir()):
        if not directory.is_dir():
            continue
        files = [path for path in sorted(directory.rglob("*.jsonl")) if path.is_file() and path.stat().st_size]
        if files:
            groups.append((directory.name, files))
    if not groups:
        raise FileNotFoundError(f"No nonempty JSONL files found in the subdirectories of: {input_dir}")
    return groups


def _load_group_lines(group_name: str, files: list[Path], line_counts: list[int]) -> list[str]:
    """Read every line of one group's *files* into RAM, validating each as a JSON object.

    Blank lines are skipped. The first invalid line aborts with a ``<file>:<line number>`` error.
    """
    lines: list[str] = []
    with tqdm(total=sum(line_counts), desc=f"Loading {group_name}", unit="line") as pbar:
        for path in files:
            with path.open("r", encoding="utf-8") as fin:
                for line_num, line in enumerate(fin, start=1):
                    pbar.update(1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("expected a JSON object")
                        del record
                    except (ValueError, RecursionError) as exc:
                        raise ValueError(f"{path}:{line_num}: {exc}") from exc
                    lines.append(line)
    return lines


def _mixed_epoch_batches(group_lines: list[list[str]], batch_size: int, seed: int) -> Iterator[list[str]]:
    """Yield shuffled batches of rows, each batch proportionally mixed across groups.

    Every batch approximates the global group proportions by always drawing from the group with the largest cumulative
    shortfall, and each group's rows are visited in one seeded permutation order.
    """
    rng = random.Random(seed)
    counts = [len(lines) for lines in group_lines]
    total = sum(counts)
    permutations = [list(range(count)) for count in counts]
    for order in permutations:
        rng.shuffle(order)
    used = [0] * len(group_lines)

    for emitted in range(0, total, batch_size):
        stop = min(emitted + batch_size, total)
        # Largest cumulative shortfall, in integer arithmetic. The heap
        # retains the same tie-breaking rule across batch boundaries.
        quotas = [(used[i] * total - stop * count, i) for i, count in enumerate(counts) if used[i] < count]
        heapify(quotas)
        batch = []
        for _ in range(stop - emitted):
            score, group = heappop(quotas)
            batch.append(group_lines[group][permutations[group][used[group]]])
            used[group] += 1
            if used[group] < counts[group]:
                heappush(quotas, (score + total, group))

        rng.shuffle(batch)
        yield batch


def _write_output(output_path: Path, batches: Iterator[list[str]], total_rows: int) -> None:
    """Write mixed *batches* to *output_path*, one row per line."""
    with output_path.open("w", encoding="utf-8") as fout, tqdm(total=total_rows, desc="Writing", unit="line") as pbar:
        for batch in batches:
            for line in batch:
                fout.write(line)
                fout.write("\n")
            pbar.update(len(batch))


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Shuffle grouped JSONL datasets in RAM and write one proportionally mixed epoch."
    )
    parser.add_argument(
        "input_dir", type=Path, help="Input directory whose subdirectories each hold one environment's JSONL file(s)."
    )
    parser.add_argument(
        "--batch-size",
        "--bs",
        "--batch",
        dest="batch_size",
        type=int,
        required=True,
        help="Rows per proportionally mixed batch (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output JSONL file (default: '<input_dir>/rl_mixed_bs<batch_size>.jsonl').",
    )
    parser.add_argument("--seed", type=int, default=67, help="Random seed for shuffling (default: %(default)s).")
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    if args.batch_size <= 0:
        parser.error(f"Batch size must be a positive integer, got: {args.batch_size}")

    output_path: Path = (
        args.output.resolve() if args.output is not None else input_dir / f"rl_mixed_bs{args.batch_size}.jsonl"
    )

    try:
        groups = _resolve_group_files(input_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    input_files = [path for _, files in groups for path in files]
    if output_path in [path.resolve() for path in input_files]:
        parser.error(f"Input and output paths must differ, got: {output_path}")
    if output_path.parent != input_dir and input_dir in output_path.parents:
        parser.error("Output nested inside an input group would be picked up as a source on the next run.")

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

    logger.info(
        f"Merging {len(groups)} group(s) with batch size {args.batch_size} and seed {args.seed} -> {output_path}"
    )

    line_counts = [[_count_lines(path) for path in files] for _, files in groups]
    total_lines = sum(sum(counts) for counts in line_counts)
    logger.info(f"Found {total_lines:,} line(s) across {len(input_files)} file(s).")

    group_lines: list[list[str]] = []
    try:
        for (name, files), counts in zip(groups, line_counts):
            group_lines.append(_load_group_lines(name, files, counts))
    except (OSError, ValueError) as exc:
        logger.error(f"Failed to load input data: {exc}")
        sys.exit(1)

    sample_counts = [len(lines) for lines in group_lines]
    total_samples = sum(sample_counts)
    if total_samples == 0:
        parser.error("All source files contain only blank lines.")
    for (name, _), sample_count in zip(groups, sample_counts):
        logger.info(f"{name}: {sample_count:,} samples ({sample_count / total_samples:.2%})")

    batches = _mixed_epoch_batches(group_lines, args.batch_size, args.seed)
    _write_output(output_path, batches, total_samples)
    batch_count = (total_samples + args.batch_size - 1) // args.batch_size
    logger.info(f"Done - wrote {total_samples:,} sample(s) in {batch_count:,} batch(es) to {output_path}")


if __name__ == "__main__":
    main()
