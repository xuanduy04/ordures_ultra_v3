from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import sys
from pathlib import Path

from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Buffer size (in bytes) for the fast line-counting pass on large files.
BUF_SIZE = 1024 * 1024


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


def _resolve_input_files(input_pattern: str) -> list[Path]:
    """Resolve *input_pattern* (a direct file path or a glob) to a sorted list of files."""
    candidates = [Path(p) for p in glob.glob(input_pattern)]
    files = sorted(p for p in candidates if p.is_file())
    if not files:
        raise FileNotFoundError(f"No input files matched pattern: {input_pattern}")
    return files


def _count_lines(files: list[Path]) -> dict[Path, int]:
    """First pass: count the number of lines in each input file."""
    line_counts: dict[Path, int] = {}
    for file_path in files:
        # Using a buffer-based read for maximum speed on large files
        with file_path.open("rb") as f:
            lines = 0
            read_f = f.raw.read
            # Wrap in tqdm; Note: total is unknown initially, so it will show iterations/sec
            with tqdm(unit=" line", desc=f"Counting {file_path.name}") as pbar:
                buf = read_f(BUF_SIZE)
                while buf:
                    newlines = buf.count(b"\n")
                    lines += newlines
                    pbar.update(newlines)
                    buf = read_f(BUF_SIZE)
            line_counts[file_path] = lines
    return line_counts


def _sample_pass(
    files: list[Path],
    output_path: Path,
    keep_indices: set[int],
    line_counts: dict[Path, int],
    sample_size: int,
) -> int:
    """Second pass: stream the files in the same order and write selected lines."""
    written = 0
    global_idx = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as fin:
                for line in tqdm(
                    fin,
                    total=line_counts[file_path],
                    desc=f"Sampling {file_path.name}",
                    unit="line",
                ):
                    if global_idx in keep_indices:
                        try:
                            # Validate that it is valid JSON
                            data = json.loads(line.strip())
                            # Write back out as JSONL
                            json.dump(data, fout, ensure_ascii=False)
                            fout.write("\n")
                            written += 1
                        except json.JSONDecodeError as exc:
                            logger.warning(f"Skipping invalid JSON at line {global_idx}: {exc}")
                    global_idx += 1
                    # Performance optimization: stop early if we have written all requested samples
                    if written == sample_size:
                        break
                if written == sample_size:
                    break
    return written


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Randomly sample lines from a large JSONL file without loading it all into memory.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input JSONL file or a glob pattern (e.g. './*.jsonl').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
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
        "--size",
        type=int,
        required=True,
        help="Number of samples to extract.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed for sampling (default: %(default)s).",
    )

    args = parser.parse_args()

    input_pattern: str = args.input
    output_path: Path = args.output.resolve()
    sample_size: int = args.size
    seed: int = args.seed

    if sample_size <= 0:
        parser.error("Sample size must be a positive integer.")

    try:
        files = _resolve_input_files(input_pattern)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if output_path in [f.resolve() for f in files]:
        parser.error(f"Input and output paths must differ, got: {output_path}")

    try:
        _validate_output_path(output_path)
    except FileExistsError:
        if not args.yes:
            response = input(f"{output_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Existing output file will not be overridden; exiting.")
                sys.exit(0)

    logger.info(f"Sampling {sample_size} line(s) from {len(files)} input file(s) -> {output_path}")

    line_counts = _count_lines(files)
    total_lines = sum(line_counts.values())
    for file_path, count in line_counts.items():
        logger.info(f"Found {count:>6} line(s) in '{file_path.name}'.")

    if total_lines == 0:
        parser.error("Input dataset is empty.")

    if sample_size > total_lines:
        logger.warning(
            f"Requested sample size ({sample_size}) is larger than the total "
            f"number of lines ({total_lines}). Sampling all lines instead."
        )
        sample_size = total_lines

    keep_indices = set(random.Random(seed).sample(range(total_lines), sample_size))
    written = _sample_pass(files, output_path, keep_indices, line_counts, sample_size)

    logger.info(f"Done - wrote {written} entries to {output_path}.")


if __name__ == "__main__":
    main()
