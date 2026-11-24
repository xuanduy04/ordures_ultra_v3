# Cross-Tokenizer On-Policy Distillation (X-OPD)

X-OPD is on-policy distillation (OPD) from a teacher whose tokenizer differs from the policy's. It is the same
advantage estimator as same-tokenizer OPD (`OPDAdvantageEstimator`), with the same loss and training loop. Only the
teacher logprobs are obtained differently: the teacher scores its own tokenization of each rollout, and the result is
mapped back onto the policy's tokens.

## Enable X-OPD

Set up OPD exactly as for a same-tokenizer teacher (the `opd` advantage estimator, `on_policy_distillation.enabled` and
the teacher mapping), then turn X-OPD on in an `on_policy_distillation.xtoken` section:

| Key | Meaning |
|---|---|
| `enabled` | `true` turns X-OPD on. With `false`, or without the section, OPD uses a teacher with the policy's tokenizer. |
| `teacher_tokenizer` | Tokenizer config of the teacher (same schema as `policy.tokenizer`). It must be a fast tokenizer whose token IDs match the served teacher model. |
| `alignment_max_combination_length` | Longest run of tokens on one side that can match a single token of the other side during alignment. |
| `alignment_workers` | Optional. Size of one pool of spawned CPU processes that every rollout group shares: the samples of all groups awaiting alignment are spread over it in parallel. Without it, alignment runs in the trajectory collector. The result is the same. |
| `alignment_backend` | Optional. Set to `numba` for compiled CPU dynamic programming and NumPy packing. The collector needs an installed Numba compatible with its NumPy version. Omitting this key or setting `python` uses the reference backend. |

All other OPD settings (`opd_advantage_weight`, the advantage clipping, the ORM blend, ...) apply unchanged.

## How the Teacher Logprobs Are Computed

For each rollout group, the trajectory collector:

1. Renders the prompt messages with the teacher's chat template and appends the policy's decoded response, without its
   trailing stop tokens. A teacher token that straddles the prompt/response boundary counts as prompt.
2. Scores that sequence on the teacher serve, the same way same-tokenizer OPD scores the policy's tokens.
3. Aligns the two tokenizations of the response with the token aligner. A chunk is a span of response text that both
   tokenizations cover exactly.
4. Gives each policy token of a chunk the chunk's summed teacher logprob divided by the chunk's number of policy tokens.
   Where both tokenizers split the text the same way, this is exactly the teacher's logprob of that token.

`OPDAdvantageEstimator` then computes teacher minus policy logprob for every token, as for any OPD teacher. A policy
token without a teacher logprob gets zero OPD advantage (the ORM part is unaffected). That happens for:

- tokens of a chunk whose text differs between the two tokenizations;
- the trailing stop tokens, since the teacher never scores an end of turn;
- with a byte-level policy tokenizer, the tokens after the first sampled byte that never completes a character: the
  teacher scores only the longest prefix of the response that is valid UTF-8 and ends at a sampled token boundary (a
  literal U+FFFD is valid UTF-8 and is scored). The policy's token IDs are unchanged, and the number of such tokens is
  printed;
- every token of a rollout the teacher skips: tools or instructions in the request, a non-text prompt (a string `input`
  and `input_text` content parts count as text, joined as NeMo-Gym's vLLM model server joins them), more than one
  sampled turn, prompt token IDs that are not the policy's chat template of the prompt messages (with the request's
  `metadata.chat_template_kwargs`; the generation prompt may differ, e.g. through the model server's
  `chat_template_kwargs`, as long as it has the same structural special tokens), an empty response, a
  structural special token (anything but `<think>` / `</think>`) on either side, or a response that does not decode
  cleanly (with a byte-level policy tokenizer, one without a valid UTF-8 prefix). Skipped rollouts are printed with
  their reasons and count towards `teacher_logprob_skipped`.

## Verification Status

Covered by CPU unit tests in `tests/unit/algorithms/x_token/`: the aligner, the logprob mapping, the skip rules, and
one rollout group through the real trajectory collector into `OPDAdvantageEstimator`. Not verified in this repository:
GPU training runs, a live teacher endpoint, and distillation quality.
