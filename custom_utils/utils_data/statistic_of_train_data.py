from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


AGENT_REF_NEEDS_LLM_JUDGE: set[str] = {
    "math_with_judge_simple_agent",
    "genrm_simple_agent",
    "genrm_simple_agent_reasoning_off",
    "multichallenge_simple_agent",
    "jailbreak_detection_simple_agent",
    "over_refusal_detection_simple_agent",
    "genrm_simple_agent",
}


CATEGORY_TO_AGENT_REF: dict[str, set[str]] = {
    "__Unknown__": {
        # Technically, general_qa is meant for "Math, Code, MCQ" type.
        # However, in practice, general_qa works an all-encompassing environment
        'general_qa_simple_agent',
    },
    "RLHF, IF": {
        'citation_format_simple_agent',
        'freeform_formatting_simple_agent',
        'genrm_simple_agent',
        'genrm_simple_agent_reasoning_off',
        'instruction_following_simple_agent',
        'multichallenge_simple_agent',
        'structured_outputs_simple_agent',
        'structured_outputs_v3_simple_agent',
    },
    "Math, Code, MCQ": {
        'code_gen_simple_agent',
        'math_with_judge_simple_agent',
        'mcqa_simple_agent',
        'reasoning_gym_simple_agent',
    },
    "Agentic": {
        'calendar_simple_agent',
        'toolcall_schema_single_step_tool_use_with_argument_comparison_agent',
        'single_step_tool_use_with_argument_comparison_agent',
        'workplace_assistant_simple_agent'
    },
    "Harness": {
        # N/A for now
    },
}

AGENT_REF_TO_CATEGORY = {
    agent_ref: category
    for category, agent_refs in CATEGORY_TO_AGENT_REF.items()
    for agent_ref in agent_refs
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


def _validate_output_path(output_path: Path) -> None:
    """Ensure *output_path* ends with ``.md``, parent dir exists, and file does not already exist."""
    if output_path.suffix.lower() != ".md":
        raise ValueError(f"Output path must end with .md, got: {output_path}")

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


def _summarize_agent_names(input_files: list[Path]) -> Counter:
    """Count agent names across the given JSONL files."""
    counter = Counter()

    for file in input_files:
        with file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(
                tqdm(f, desc=f"Reading {file.name}", unit="lines"),
                start=1,
            ):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON on line {line_num} in {file.name}")
                    continue

                agent_ref = data.get("agent_ref")

                if not isinstance(agent_ref, dict):
                    logger.warning(f"'agent_ref' is missing or not a dict at line {line_num} in {file.name}")
                    continue

                name = agent_ref.get("name")

                if name is None:
                    logger.warning(f"'name' key missing in 'agent_ref' at line {line_num} in {file.name}")
                    continue

                counter[name.strip()] += 1
    return counter


def _generate_summary(results: Counter | dict, output_path: Path) -> None:
    # Standardize input to Counter
    results = Counter(results) if not isinstance(results, Counter) else results
    if not results:
        raise ValueError("No results to summarize.")

    # Calculate stats
    output_stats = defaultdict(list)
    output_catergory_samples = defaultdict(int)
    
    total_count = results.total()
    for name, count in results.most_common():
        percentage = f"{count / total_count * 100:05.2f}%"
        
        judge_type = "x"
        if "genrm" in name.lower():
            judge_type = "GenRM"
        elif name in AGENT_REF_NEEDS_LLM_JUDGE:
            judge_type = "LLM-Judge"
            
        category = AGENT_REF_TO_CATEGORY.get(name, "__Unknown__")
        output_stats[category].append((name, count, percentage, judge_type))
        output_catergory_samples[category] += count
    output_catergory_samples = Counter(output_catergory_samples)

    # Config & Padding
    name_pad = int(max(len(str(name)) for name in results.keys()))
    separator_pad = name_pad + 40
    
    # Print table
    with output_path.open("w", encoding="utf-8") as output:
        # Table header
        print(f"{'Environment':<{name_pad}} | {'Num samples':^11} | {'% samples':^9} | {'Judge':^9} |", file=output)
        print("=" * separator_pad, file=output)
        
        # Table body
        for category, catergory_count in output_catergory_samples.most_common():
            # Category header
            print(f"{category:-^{name_pad}}{'-'*40}", file=output)
    
            # Category body
            for name, count, percentage, judge_type in output_stats[category]:
                print(f"{str(name):<{name_pad}} | {count:>11} | {percentage:>9} | {judge_type:^9} |", file=output)
            
            # Category footer
            catergory_percentage = f"{catergory_count / total_count * 100:05.2f}%"
            print(f"{"TOTAL FOR CATERGORY":>{name_pad}} | {catergory_count:>11} | {catergory_percentage:>9} | {"":^9} |", file=output)
    
        # Table footer
        print("=" * separator_pad, file=output)
        print(f"{"GRAND TOTAL  ":>{name_pad}} | {total_count:>11} | {"100.00%":>9} | {"":^9} |", file=output)
        print("", file=output)


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Get statistic of different agent types of a training data",
    )
    parser.add_argument(
        "input",
        nargs="+",
        type=str,
        help="Input JSONL file(s) or glob pattern(s) for analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./statistic_of_train_data.md"),
        help="Path to the output md file (default: %(default)s).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )

    args = parser.parse_args()

    output_path: Path = args.output.resolve()

    try:
        input_files = _resolve_input_files(args.input)
    except FileNotFoundError as exc:
        parser.error(str(exc))

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
                logger.info("Existing output file will not be overridden; exiting.")
                sys.exit(0)

    logger.info(f"Analyzing {len(input_files)} input file(s) -> {output_path}")

    results = _summarize_agent_names(input_files)
    if not results:
        parser.error("No valid agent_ref entries found in the input data.")

    _generate_summary(results, output_path)
    logger.info(f"Resulting dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
