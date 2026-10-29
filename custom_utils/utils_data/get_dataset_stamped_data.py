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


def _stamp_line(line: str, dataset: str) -> dict | None:
    """Parse *line* and return it as a dict with its ``dataset`` field stamped.

    Returns ``None`` for empty, invalid-JSON, or non-object lines.
    """
    stripped = line.strip()
    if not stripped:
        return None

    try:
        entry = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(entry, dict):
        return None

    entry["dataset"] = dataset
    return entry


def _stamp_output_pass(input_files: list[Path], dataset: str, output_path: Path) -> dict[str, int]:
    """Write every input row to *output_path* with its ``dataset`` field stamped.

    Invalid JSON lines and non-object JSON lines are warned about and skipped.
    """
    stamped_count = 0
    skipped_count = 0

    with output_path.open("w", encoding="utf-8") as outfile:
        for input_path in tqdm(input_files, desc="Files processed", unit="file"):
            with input_path.open("r", encoding="utf-8") as infile:
                for line_num, line in tqdm(
                    enumerate(infile, 1), desc=f"Stamping {input_path.name}", unit="line", leave=False
                ):
                    entry = _stamp_line(line, dataset)
                    if entry is None:
                        if line.strip():
                            skipped_count += 1
                            logger.warning(f"Skipping invalid JSON on line {line_num} in {input_path}. It was not copied.")
                        continue

                    json.dump(entry, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    stamped_count += 1

    return {"lines_stamped": stamped_count, "invalid_json_skipped": skipped_count}


def _stamp_in_place_pass(input_files: list[Path], dataset: str) -> dict[str, int]:
    """Rewrite each input file in place with its ``dataset`` field stamped.

    Each file is replaced only after all of its rows were written successfully
    to a sibling temporary file. Invalid JSON lines and non-object JSON lines
    are warned about and dropped.
    """
    stamped_count = 0
    skipped_count = 0
    files_processed = 0

    for input_path in tqdm(input_files, desc="Files processed", unit="file"):
        temp_path = input_path.with_name(input_path.name + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as outfile:
                with input_path.open("r", encoding="utf-8") as infile:
                    for line_num, line in tqdm(
                        enumerate(infile, 1), desc=f"Stamping {input_path.name}", unit="line", leave=False
                    ):
                        entry = _stamp_line(line, dataset)
                        if entry is None:
                            if line.strip():
                                skipped_count += 1
                                logger.warning(f"Skipping invalid JSON on line {line_num} in {input_path}. It was not kept.")
                            continue

                        json.dump(entry, outfile, ensure_ascii=False)
                        outfile.write("\n")
                        stamped_count += 1
            temp_path.replace(input_path)
            files_processed += 1
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    return {"files_processed": files_processed, "lines_stamped": stamped_count, "invalid_json_skipped": skipped_count}


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Uniformly assign the 'dataset' field of all input datasets to the same value.",
    )
    parser.add_argument(
        "input",
        nargs="+",
        type=str,
        help="Input JSONL file(s) or glob pattern(s) of one dataset group.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output JSONL file. If omitted, input files are stamped in place.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Value assigned to the 'dataset' field of every row.",
    )

    args = parser.parse_args()

    dataset: str = args.dataset.strip()
    if not dataset:
        parser.error("--dataset must not be empty.")

    try:
        input_files = _resolve_input_files(args.input)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if args.output is None:
        logger.info(f"Stamping {len(input_files)} input file(s) in place with dataset '{dataset}'")
        stats = _stamp_in_place_pass(input_files, dataset)
        logger.info(f"Done - in-place stamping complete:\n{pprint.pformat(stats)}.")
        return

    output_path: Path = args.output.resolve()

    if output_path in [f.resolve() for f in input_files]:
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

    logger.info(f"Stamping {len(input_files)} input file(s) with dataset '{dataset}' -> {output_path}")

    stats = _stamp_output_pass(input_files, dataset, output_path)

    logger.info(f"Done - stamping complete:\n{pprint.pformat(stats)}.")
    logger.info(f"Resulting dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
