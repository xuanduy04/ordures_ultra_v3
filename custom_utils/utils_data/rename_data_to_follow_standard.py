from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESOURCES_SERVERS_DIR = (
    Path(__file__).resolve().parents[2] / "3rdparty" / "Gym-workspace" / "Gym" / "resources_servers"
)

LEN_FIELD_PATTERN = re.compile(r"len(\d+)k")
SAM_FIELD_PATTERN = re.compile(r"sam(\d+)k")


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


def _load_valid_data_types() -> set[str] | None:
    """Collect the valid data type names.

    The fixed types ``SFT``, ``RL_QA``, and ``nemo_gym`` are always valid.
    NeMo-Gym data containing exactly 1 environment uses that environment's
    name, so every subdirectory of the Gym ``resources_servers`` directory is
    accepted as well.  If that directory is unavailable, ``None`` is returned
    to signal that any trailing filename field is accepted as a data type.
    """
    valid_data_types = {"SFT", "RL_QA", "nemo_gym"}
    if not RESOURCES_SERVERS_DIR.is_dir():
        logger.warning(
            f"Gym resources_servers directory not found at {RESOURCES_SERVERS_DIR}; "
            "accepting any trailing filename field as the data type."
        )
        return None
    valid_data_types.update(
        entry.name for entry in RESOURCES_SERVERS_DIR.iterdir() if entry.is_dir()
    )
    return valid_data_types


def _consume_metadata_fields(
    fields: list[str], idx: int, len_field: str | None
) -> tuple[int, str | None]:
    """Walk left from *idx*, consuming ``len``/``sam`` metadata fields.

    Returns the index of the first non-metadata field (or ``-1``) and the
    rightmost ``len`` field encountered, normalised to ``len<value>k``.
    """
    while idx >= 0:
        len_match = LEN_FIELD_PATTERN.fullmatch(fields[idx])
        if len_match is not None:
            if len_field is None:
                len_field = f"len{max(1, int(len_match.group(1)))}k"
            idx -= 1
            continue
        if SAM_FIELD_PATTERN.fullmatch(fields[idx]) is not None:
            idx -= 1
            continue
        break
    return idx, len_field


def _parse_filename_right_to_left(
    stem: str, valid_data_types: set[str] | None
) -> tuple[str, str | None, str | None]:
    """Parse a filename stem right-to-left into ``(content, len_field, data_type)``.

    ``sam`` metadata is always discarded (the script recomputes it) and ``len``
    metadata is kept (rightmost wins).  The rightmost non-metadata field is the
    data type when it is in *valid_data_types* (``None`` accepts anything);
    otherwise it is treated as part of the content.  ``data_type`` is ``None``
    when the filename carries no valid data type.
    """
    fields = stem.split("-")

    idx, len_field = _consume_metadata_fields(fields, len(fields) - 1, None)
    if idx < 0:
        return "", len_field, None

    dtype_field = fields[idx]
    if valid_data_types is not None and dtype_field not in valid_data_types:
        return "-".join(fields[: idx + 1]), len_field, None

    idx, len_field = _consume_metadata_fields(fields, idx - 1, len_field)
    if idx < 0:
        return "", len_field, dtype_field

    return "-".join(fields[: idx + 1]), len_field, dtype_field


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


def _round_to_thousands(value: int) -> int:
    """Round *value* to the nearest thousand, representing values below 1k as 1k."""
    return max(1, (value + 500) // 1000)


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Rename a JSONL file inplace to follow the dataset filename standard.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input JSONL file (must end with .jsonl).",
    )
    parser.add_argument(
        "--data-type",
        default=None,
        help="Data type for the renamed file (e.g. SFT, RL_QA, nemo_gym, or a NeMo-Gym "
        "environment name). Overrides the data type parsed from the filename, and is "
        "required when the filename has none (default: parse from the filename).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the target file (default: false).",
    )

    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    data_type_arg: str | None = args.data_type

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    if input_path.suffix.lower() != ".jsonl":
        parser.error(f"Input path must end with .jsonl, got: {input_path}")

    if data_type_arg is not None:
        data_type_arg = data_type_arg.strip()
        if not data_type_arg:
            parser.error("--data-type is empty after stripping")
        if "-" in data_type_arg:
            parser.error(f"--data-type must not contain '-', got: {data_type_arg}")

    valid_data_types = _load_valid_data_types()
    content, len_field, filename_data_type = _parse_filename_right_to_left(
        input_path.stem, valid_data_types
    )

    if not content:
        parser.error(
            f"Could not determine a content segment from filename '{input_path.name}'; "
            "expected '<content>[-<metadata>]-<data type>'."
        )

    if data_type_arg is not None:
        data_type = data_type_arg
    elif filename_data_type is not None:
        data_type = filename_data_type
    else:
        parser.error(
            f"Could not determine the data type from filename '{input_path.name}'; "
            "pass --data-type to specify it (e.g. --data-type SFT)."
        )

    logger.info(
        f"Parsed '{input_path.name}': content='{content}', "
        f"len={len_field if len_field is not None else 'none'}, data type='{data_type}'."
    )

    line_count = _count_lines(input_path)
    if line_count == 0:
        parser.error("Empty input file.")

    sample_count = _round_to_thousands(line_count)
    logger.info(f"Counted {line_count} line(s) -> sam{sample_count}k.")

    stem_parts = [content]
    if len_field is not None:
        stem_parts.append(len_field)
    stem_parts.append(f"sam{sample_count}k")
    stem_parts.append(data_type)
    target_path = input_path.with_name("-".join(stem_parts) + input_path.suffix)

    if target_path == input_path:
        logger.info(f"'{input_path.name}' already follows the filename standard; nothing to do.")
        return

    if target_path.exists() and not args.yes:
        response = input(f"{target_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
        if response not in ("y", "yes"):
            logger.info("Existing target file will not be overridden; exiting.")
            sys.exit(0)

    input_path.rename(target_path)
    logger.info(f"Done - renamed '{input_path.name}' -> '{target_path.name}'.")


if __name__ == "__main__":
    main()
