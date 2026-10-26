"""Derive a deterministic short-prompt smoke without changing the training pool."""

import argparse
import json
from pathlib import Path
import sys

from integrity import sha256


def select_rows(rows, tokenizer, count, max_prompt_tokens):
    from nemo_rl.algorithms.x_token.gym_protocol import prepare_example

    selected, identities = [], []
    for index, row in enumerate(rows):
        _, prompt = prepare_example(row, index, 1)
        ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
        if 0 < len(ids) <= max_prompt_tokens:
            selected.append(row)
            identities.append({"source_row_index": index, "prompt_tokens": len(ids)})
        if len(selected) == count:
            return selected, identities
    raise ValueError("Too few prompts within the requested smoke budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, required=True)
    args = parser.parse_args()
    if args.count <= 0 or args.max_prompt_tokens <= 0 or args.output_dir.exists():
        raise ValueError("Use positive explicit limits and a new output directory")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nemo_rl.algorithms.utils import get_tokenizer

    tokenizer = get_tokenizer({"name": args.tokenizer, "chat_template_kwargs": {"enable_thinking": False}})
    before = sha256(args.input)
    with args.input.open(encoding="utf-8") as stream:
        selected, ids = select_rows((json.loads(line) for line in stream), tokenizer,
                                    args.count, args.max_prompt_tokens)
    assert sha256(args.input) == before
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target = args.output_dir / "train.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    receipt = {"source_sha256": before, "output_sha256": sha256(target),
               "selection": "first eligible rows in original order; no randomness",
               "selected_rows": ids, "max_prompt_tokens": args.max_prompt_tokens,
               "scope": "mechanical smoke only; original pool remains unchanged"}
    (args.output_dir / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
