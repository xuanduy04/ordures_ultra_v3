from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

AGENT_NAME = "general_qa_simple_agent"
EXPECTED_ANSWER = "mock"
N_FRESH_SAMPLES = 3
N_REFINEMENTS = 3
DEFAULT_SEED = 67


def _preprocess_underscore_args(argv: list[str]) -> list[str]:
    """Replace underscores with hyphens in ``--arg_name`` / ``--arg_name=val`` prefixes.

    argparse normalises ``-`` and ``_`` internally, so ``--foo-bar`` and
    ``--foo_bar`` would normally map to different destinations.  This
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


def _measure_chat_tokens(tokenizer: Any, text: str) -> int:
    """Return the token count of a single-user-turn chat template containing *text*."""
    conversation = [{"role": "user", "content": text}]
    encoded = tokenizer.apply_chat_template(conversation, tokenize=True, return_dict=True)
    return len(encoded["input_ids"])


def _allowed_token_ids(tokenizer: Any) -> list[int]:
    """Return vocabulary token IDs that are neither special nor added tokens."""
    excluded = set(tokenizer.all_special_ids)
    excluded.update(tokenizer.get_added_vocab().values())
    allowed_ids = sorted(set(tokenizer.get_vocab().values()) - excluded)
    if not allowed_ids:
        raise ValueError("Tokenizer has no non-special token IDs to sample from.")
    return allowed_ids


def _decode_ids(tokenizer: Any, token_ids: list[int]) -> str:
    """Decode token IDs to text without stripping special tokens or cleaning spaces."""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _make_candidate(tokenizer: Any, token_ids: list[int]) -> tuple[str, int]:
    """Decode *token_ids* and return the decoded text with its chat-templated token count."""
    text = _decode_ids(tokenizer, token_ids)
    return text, _measure_chat_tokens(tokenizer, text)


def _is_better(count: int, best_count: int, target: int) -> bool:
    """Return whether *count* is closer to *target* than *best_count*.

    A tie prefers the count that does not exceed *target*.
    """
    count_distance = abs(count - target)
    best_distance = abs(best_count - target)
    if count_distance != best_distance:
        return count_distance < best_distance
    return count <= target and best_count > target


def _generate_sample(
    tokenizer: Any,
    target_len: int,
    overhead: int,
    allowed_ids: list[int],
    rng: random.Random,
    content_budget: int,
) -> tuple[str, int]:
    """Generate the sample whose chat-templated token count is closest to *target_len*.

    Each of ``N_FRESH_SAMPLES`` fresh attempts starts from ``content_budget`` random
    token IDs.  Up to ``N_REFINEMENTS`` rounds follow: when the measured count is
    below *target_len*, the missing number of token IDs is appended; when it is
    above, the ID sequence is trimmed proportionally.  The best state seen across
    all attempts is returned.
    """
    best_text = ""
    best_count = -1

    for _ in range(N_FRESH_SAMPLES):
        token_ids = rng.choices(allowed_ids, k=content_budget)
        text, count = _make_candidate(tokenizer, token_ids)
        if best_count < 0 or _is_better(count, best_count, target_len):
            best_text, best_count = text, count

        for _ in range(N_REFINEMENTS):
            if count == target_len:
                break
            if count < target_len:
                token_ids.extend(rng.choices(allowed_ids, k=target_len - count))
            else:
                content_count = count - overhead
                new_id_count = max(1, int(len(token_ids) * content_budget / content_count))
                if new_id_count >= len(token_ids):
                    break
                token_ids = token_ids[:new_id_count]
            text, count = _make_candidate(tokenizer, token_ids)
            if _is_better(count, best_count, target_len):
                best_text, best_count = text, count

    return best_text, best_count


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Generate mock general_qa data with a target chat-templated token length.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output JSONL file (default: './mock_data-len<len>.jsonl').",
    )
    parser.add_argument(
        "--len",
        type=int,
        required=True,
        help="Target token length per sample, including the chat template (literal tokens).",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Number of identical samples to write.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to a local HuggingFace tokenizer directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for token sampling (default: %(default)s).",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Preemptively confirm overwrite of the output file (default: false).",
    )

    args = parser.parse_args()

    if args.len <= 0:
        parser.error("--len must be a positive integer.")
    if args.size <= 0:
        parser.error("--size must be a positive integer.")

    tokenizer_dir: Path = args.tokenizer.resolve()
    if not tokenizer_dir.is_dir():
        parser.error(f"Tokenizer directory does not exist: {tokenizer_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    except Exception as exc:
        parser.error(f"Failed to load tokenizer from {tokenizer_dir}: {exc}")

    output_path: Path = (
        args.output.resolve()
        if args.output is not None
        else Path(f"./mock_data-len{args.len}.jsonl").resolve()
    )

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

    try:
        overhead = _measure_chat_tokens(tokenizer, "")
    except Exception as exc:
        parser.error(f"Failed to apply the tokenizer's chat template: {exc}")

    content_budget = args.len - overhead
    if content_budget <= 0:
        parser.error(
            f"--len ({args.len}) must exceed the chat template overhead ({overhead} tokens)."
        )

    try:
        allowed_ids = _allowed_token_ids(tokenizer)
    except ValueError as exc:
        parser.error(str(exc))

    logger.info(
        f"Generating one sample for {args.len} chat-templated tokens "
        f"(template overhead={overhead}, content budget={content_budget})."
    )

    rng = random.Random(args.seed)
    text, count = _generate_sample(
        tokenizer, args.len, overhead, allowed_ids, rng, content_budget
    )

    if count == args.len:
        logger.info(f"Generated a sample with exactly {args.len} chat-templated tokens.")
    else:
        logger.warning(
            f"Best-effort sample has {count} chat-templated tokens instead of the target {args.len}."
        )

    row = {
        "agent_ref": {"name": AGENT_NAME},
        "responses_create_params": {"input": [{"role": "user", "content": text}]},
        "question": text,
        "expected_answer": EXPECTED_ANSWER,
        "should_use_judge": False,
    }
    line = json.dumps(row, ensure_ascii=False) + "\n"

    with output_path.open("w", encoding="utf-8") as fout, tqdm(
        total=args.size, desc="Writing mock data", unit="sample"
    ) as pbar:
        for _ in range(args.size):
            fout.write(line)
            pbar.update(1)

    logger.info(f"Done - wrote {args.size} sample(s) to {output_path}.")


if __name__ == "__main__":
    main()
