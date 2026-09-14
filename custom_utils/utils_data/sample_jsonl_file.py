from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
from pathlib import Path

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


def _sample_pass(
    input_path: Path,
    output_path: Path,
    keep_indices: set[int],
    total_lines: int,
    sample_size: int,
) -> int:
    """Stream the input file and write selected lines."""
    written = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(
            tqdm(fin, total=total_lines, desc=f"Sampling {input_path.name}", unit="line")
        ):
            if idx in keep_indices:
                try:
                    # Validate that it is valid JSON
                    data = json.loads(line.strip())

                    # Write back out as JSONL
                    json.dump(data, fout, ensure_ascii=False)
                    fout.write("\n")
                    written += 1

                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping invalid JSON at line {idx}: {exc}")

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
        type=Path,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output JSONL file (default: '<input stem>-sam<size>.jsonl').",
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

    input_path: Path = args.input.resolve()
    sample_size: int = args.size
    seed: int = args.seed

    if sample_size <= 0:
        parser.error("Sample size must be a positive integer.")

    output_path: Path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(input_path.stem + f"-sam{sample_size}.jsonl")
    )

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    if output_path == input_path:
        parser.error(f"Input and output paths must differ, got: {input_path}")

    try:
        _validate_output_path(output_path)
    except FileExistsError:
        if not args.yes:
            response = input(f"{output_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Existing output file will not be overridden; exiting.")
                sys.exit(0)

    logger.info(f"Sampling {sample_size} line(s) from {input_path} -> {output_path}")

    total_lines = _count_lines(input_path)

    if total_lines == 0:
        parser.error("Empty input file.")
    logger.info(f"Found {total_lines} line(s) in '{input_path.name}'.")

    if sample_size > total_lines:
        parser.error(f"Sample size ({sample_size}) exceeds the total number of lines ({total_lines}).")

    keep_indices = set(random.Random(seed).sample(range(total_lines), sample_size))
    written = _sample_pass(input_path, output_path, keep_indices, total_lines, sample_size)

    if written < sample_size:
        logger.warning(
            f"WROTE ONLY {written}/{sample_size} LINES ({100 * written // sample_size}) to output "
            "due to encountering invalid JSON lines."
        )
    logger.info(f"Done - wrote {written}/{sample_size} entries to {output_path}.")


if __name__ == "__main__":
    main()
