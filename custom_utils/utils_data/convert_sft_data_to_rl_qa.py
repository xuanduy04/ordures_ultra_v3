from __future__ import annotations

import argparse
import json
import logging
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


def _convert_pass(input_path: Path, output_path: Path) -> int:
    """Convert SFT-format records from *input_path* into RL-style question/answer records."""
    processed = 0

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("w", encoding="utf-8") as outfile:

        for idx, line in enumerate(tqdm(infile, desc="Converting", unit="line")):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                record_id = data.get("id")
                messages = data.get("messages", [])

                if not messages:
                    continue

                # Split: all turns except last -> question, last turn content -> ground_truth
                question = messages[:-1]
                last_message = messages[-1]
                ground_truth = (
                    last_message.get("content", "").strip()
                    if isinstance(last_message, dict)
                    else ""
                )

                if not question or not ground_truth:
                    continue

                transformed_data = {
                    "id": record_id,
                    "question": question,
                    "ground_truth": ground_truth,
                }

                json.dump(transformed_data, outfile, ensure_ascii=False)
                outfile.write("\n")
                processed += 1

            except (json.JSONDecodeError, KeyError, IndexError):
                logger.info(f"Skipping malformed line {idx+1} (numbered from 1)")
                continue

    return processed


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Convert an SFT-format JSONL file into RL-style question/answer records.",
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
        help="Path to the output JSONL file (default: '<input stem>-RL_QA.jsonl').",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )

    args = parser.parse_args()

    input_path: Path = Path(args.input).resolve()
    output_path: Path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(input_path.stem + "-RL_QA.jsonl")
    )

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    if output_path == input_path:
        parser.error(f"Input and output paths must differ, got: {output_path}")

    try:
        _validate_output_path(output_path)
    except FileExistsError:
        if not args.yes:
            response = input(f"{output_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Existing output file will not be overridden; exiting.")
                sys.exit(0)

    logger.info(f"Converting {input_path} -> {output_path}")

    processed = _convert_pass(input_path, output_path)

    logger.info(f"Done - wrote {processed} entries to {output_path}.")


if __name__ == "__main__":
    main()
