from __future__ import annotations

import argparse
import glob
import json
import logging
import pprint
import random
import sys
from pathlib import Path

from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Agent name -> number of samples to keep (None means keep all samples of that agent).
AGENT_COUNT: dict[str, int | None] = {
    # RLHF, IF
    "genrm_simple_agent": 7760,
    "genrm_simple_agent_reasoning_off": 3274,
    "instruction_following_simple_agent": 6693,
    "structured_outputs_simple_agent": 1538,
    "multichallenge_simple_agent": 669,
    "reasoning_gym_simple_agent": 1502,
    # Math, Code, MCQ
    "code_gen_simple_agent": 4450,
    "math_with_judge_simple_agent": 3663,
    "mcqa_simple_agent": 2085,
    # Agentic
    "single_step_tool_use_with_argument_comparison_agent": 14516,
    "calendar_simple_agent": 1516,
    "workplace_assistant_simple_agent": 676,
}


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


def _resolve_input_files(input_pattern: str) -> list[Path]:
    """Resolve *input_pattern* (a direct file path or a glob) to a sorted list of files."""
    candidates = [Path(p) for p in glob.glob(input_pattern)]
    files = sorted(p for p in candidates if p.is_file())
    if not files:
        raise FileNotFoundError(f"No input files matched pattern: {input_pattern}")
    return files


def _extract_agent_name(entry: dict) -> str:
    """Extract the agent name from a NeMo-Gym dataset entry."""
    agent_ref = entry["agent_ref"]
    if not isinstance(agent_ref, dict):
        raise ValueError(f"Entry has invalid 'agent_ref' field: expected a dict, got {type(agent_ref).__name__}")
    name = agent_ref["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("Entry has missing or invalid 'agent_ref.name' field")
    return name


def _validate_output_path(output_path: Path) -> None:
    """Ensure *output_path* ends with ``.jsonl``, parent dir exists, and file does not already exist."""
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Output path must end with .jsonl, got: {output_path}")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)


def _count_agents(files: list[Path]) -> tuple[dict[str, int], int]:
    """First pass: count the number of samples per requested agent and total samples across all files."""
    counts: dict[str, int] = {agent: 0 for agent in AGENT_COUNT}
    total_samples = 0
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc=f"Counting {file_path.name}", unit="line"):
                total_samples += 1
                stripped = line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                agent_name = _extract_agent_name(entry)
                if agent_name in counts:
                    counts[agent_name] += 1
    return counts, total_samples


def _validate_counts(counts: dict[str, int]) -> None:
    """Verify every requested agent exists with at least the requested number of samples."""
    missing = [agent for agent, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"Agents in AGENT_COUNT not found in input dataset: {missing}")

    for agent, target in AGENT_COUNT.items():
        if target is not None and counts[agent] < target:
            raise ValueError(
                f"Not enough samples for agent '{agent}': requested {target}, found {counts[agent]}"
            )


def _build_keep_sets(counts: dict[str, int], seed: int) -> dict[str, set[int] | None]:
    """Pick the per-agent occurrence indices to keep (``None`` means keep all)."""
    rng = random.Random(seed)
    keep_sets: dict[str, set[int] | None] = {}
    for agent, target in AGENT_COUNT.items():
        if target is None:
            keep_sets[agent] = None
        else:
            keep_sets[agent] = set(rng.sample(range(counts[agent]), target))
    return keep_sets


def _sample_pass(
    files: list[Path],
    output_path: Path,
    keep_sets: dict[str, set[int] | None],
    total_samples: int,
) -> dict[str, int]:
    """Second pass: stream the files in the same order and write selected samples."""
    seen: dict[str, int] = {agent: 0 for agent in AGENT_COUNT}
    written: dict[str, int] = {agent: 0 for agent in AGENT_COUNT}

    with output_path.open("w", encoding="utf-8") as fout, tqdm(
        total=total_samples, desc="Sampling", unit="line"
    ) as pbar:
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    pbar.update(1)
                    stripped = line.strip()
                    if not stripped:
                        continue
                    entry = json.loads(stripped)
                    agent_name = _extract_agent_name(entry)
                    if agent_name not in keep_sets:
                        continue
                    occurrence_idx = seen[agent_name]
                    seen[agent_name] += 1
                    keep_set = keep_sets[agent_name]
                    if keep_set is None or occurrence_idx in keep_set:
                        json.dump(entry, fout, ensure_ascii=False)
                        fout.write("\n")
                        written[agent_name] += 1
    return written


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Sample per-agent subsets from NeMo-Gym compatible JSONL dataset(s).",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input data file or a glob pattern (e.g. './*.jsonl').",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("./nemo_gym_sample.jsonl"),
        help="Path to the output JSONL file (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=67,
        help="Random seed for sampling (default: %(default)s).",
    )

    args = parser.parse_args()

    input_pattern: str = args.input
    output_path: Path = args.output_path.resolve()
    seed: int = args.seed

    try:
        files = _resolve_input_files(input_pattern)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    if output_path in [f.resolve() for f in files]:
        parser.error(f"Input and output paths must differ, got: {output_path}")

    try:
        _validate_output_path(output_path)
    except FileExistsError:
        response = input(f"{output_path} already exists. Override? Type [y]es/[n]o: ").strip().lower()
        if response not in ("y", "yes"):
            logger.info("Existing output file will not be overridden; exiting.")
            sys.exit(0)

    logger.info(f"Sampling {len(files)} input file(s) -> {output_path}")

    counts, total_samples = _count_agents(files)
    for agent, count in counts.items():
        logger.info(f"Found {count:>6} sample(s) for agent '{agent}'.")

    _validate_counts(counts)

    keep_sets = _build_keep_sets(counts, seed)
    written = _sample_pass(files, output_path, keep_sets, total_samples)

    total = sum(written.values())
    logger.info(f"Done - wrote {total} entries:\n{pprint.pformat(written)}.")


if __name__ == "__main__":
    main()
