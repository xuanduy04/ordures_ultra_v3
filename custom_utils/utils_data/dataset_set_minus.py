from __future__ import annotations

import argparse
import glob
import json
import logging
import pprint
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


def _resolve_input_files(patterns: list[str]) -> list[Path]:
    """Resolve glob *patterns* to a sorted, deduplicated list of files."""
    # Expand all globs and deduplicate using a set, then convert to Path objects
    candidates = [Path(p) for pattern in patterns for p in glob.glob(pattern)]
    files = sorted({p for p in candidates if p.is_file()})
    if not files:
        raise FileNotFoundError(f"No input files matched patterns: {patterns}")
    return files


def dataset_set_minus(A_files: list[Path], B_files: list[Path], output_path: Path) -> dict[str, int]:
    """Perform a set-minus operation (A \\ B) on JSONL datasets.

    Lines are compared as raw stripped strings; invalid JSON lines in B are
    silently skipped, invalid JSON lines in A are warned about and skipped.
    """

    logger.info("Loading exclusion dataset(s) (B)...")
    exclude_data: set[str] = set()
    for b_path in tqdm(B_files, desc="B files processed", unit="file"):
        with b_path.open("r", encoding="utf-8") as infile:
            for line in tqdm(infile, desc=f"Reading {b_path.name}", unit="line", leave=False):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    json.loads(line_str)
                    exclude_data.add(line_str)
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(exclude_data)} unique exclusion signatures. Processing dataset(s) (A)...")

    seen_in_A: set[str] = set()
    kept_count = 0
    excluded_count = 0
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
                        json.loads(line_str)
                    except json.JSONDecodeError:
                        skipped_count += 1
                        logger.warning(f"Skipping invalid JSON on line {line_num} in {a_path}. It was not copied.")
                        continue

                    if line_str in exclude_data or line_str in seen_in_A:
                        excluded_count += 1
                        continue

                    seen_in_A.add(line_str)
                    outfile.write(line_str + "\n")
                    kept_count += 1

    return {
        "unique_lines_kept": kept_count,
        "excluded_or_internal_dupes": excluded_count,
        "invalid_json_skipped": skipped_count,
    }


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Perform a set-minus operation (A \\ B) on JSONL datasets.",
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
        help="Input JSONL file(s) or glob pattern(s) for dataset B.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./output/dataset_set_minus.jsonl"),
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
        B_files = _resolve_input_files(B_patterns)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if output_path in [f.resolve() for f in A_files + B_files]:
        parser.error(f"Input and output paths must differ, got: {output_path}")

    try:
        _validate_output_path(output_path)
    except FileExistsError:
        if not args.yes:
            response = input(f"{output_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Existing output file will not be overridden; exiting.")
                sys.exit(0)

    logger.info(f"Computing A \\ B from {len(A_files)} A file(s) and {len(B_files)} B file(s) -> {output_path}")

    stats = dataset_set_minus(A_files, B_files, output_path)

    logger.info(f"Done - set-minus complete:\n{pprint.pformat(stats)}.")
    logger.info(f"Resulting dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
