from dataclasses import dataclass
from typing import Any, Mapping, Optional, cast

import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.opd_outsource import XTokenOPDConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# Special tokens a sampled response may carry as visible content (Qwen reasoning delimiters, possibly unmatched in a
# length-limited rollout). Any other special token (roles, padding, ...) is structural and the teacher skips the rollout.
_CONTENT_SPECIAL_TOKENS = frozenset({"<think>", "</think>"})

# Request parameters the teacher's chat template cannot reproduce from the prompt messages alone.
_UNSUPPORTED_REQUEST_KEYS = frozenset({"tools", "instructions", "previous_response_id"})


def _has_structural_special_token(tokenizer: PreTrainedTokenizerBase, ids: list[int]) -> bool:
    """Whether ``ids`` contain a special token other than the reasoning delimiters."""
    special_ids = set(tokenizer.all_special_ids)
    special = [token for token in ids if token in special_ids]
    return any(piece not in _CONTENT_SPECIAL_TOKENS for piece in tokenizer.convert_ids_to_tokens(special))


def _pad(rows: list[list[int]], pad_token_id: int) -> torch.Tensor:
    """Right-pad ``rows`` into a ``[len(rows), max(1, longest row)]`` tensor."""
    result = torch.full((len(rows), max([1, *map(len, rows)])), pad_token_id, dtype=torch.long)
    for index, row in enumerate(rows):
        result[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return result


def _length_mask(rows: list[list[int]], width: int) -> torch.Tensor:
    """``[len(rows), width]`` mask of the real (unpadded) positions of each row."""
    lengths = torch.tensor([len(row) for row in rows])
    return (torch.arange(width)[None, :] < lengths[:, None]).long()


def _exact_chunks(chunk_id: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    """``chunk_id`` with every chunk that is not an exact text match replaced by ``-1``."""
    keep = (chunk_id >= 0) & exact[chunk_id.clamp(min=0)]
    return chunk_id.masked_fill(~keep, -1)


@dataclass
class TeacherView:
    """One rollout group as the X-OPD teacher scores it, with the alignment back to the policy's tokens.

    A chunk is a span of response text that the policy's and the teacher's tokenizations both cover exactly.

    Attributes:
        input_ids: ``[B, T]`` teacher token IDs: the teacher's chat template of the prompt, then the policy's response
            text. Right-padded.
        input_lengths: ``[B]`` lengths of ``input_ids``; 0 for a rollout the teacher does not score.
        teacher_chunk_id: ``[B, T]`` chunk of each teacher response token, ``-1`` everywhere else.
        student_chunk_id: ``[B, S]`` the same chunks over the policy's flattened rollout tokens.
    """

    input_ids: torch.Tensor
    input_lengths: torch.Tensor
    teacher_chunk_id: torch.Tensor
    student_chunk_id: torch.Tensor

    def to_student(self, teacher_logprobs: torch.Tensor) -> torch.Tensor:
        """Map the teacher's per-position logprobs onto the policy's tokens.

        Each policy token of a chunk gets the chunk's summed teacher logprob divided by the chunk's number of policy
        tokens, so the policy tokens of a chunk sum to the teacher's logprob of its text. Every other position gets NaN,
        which ``OPDAdvantageEstimator`` turns into zero OPD advantage.

        Args:
            teacher_logprobs: ``[B, T]`` teacher logprob of each token of ``input_ids`` (NaN rows were not scored).

        Returns:
            ``[B, S]`` float32 teacher logprobs over the policy's tokens.
        """
        num_chunks = max(int(self.teacher_chunk_id.max()), int(self.student_chunk_id.max())) + 1
        # Positions outside every chunk land in one extra bucket that is never read back.
        teacher_bucket = self.teacher_chunk_id.masked_fill(self.teacher_chunk_id < 0, num_chunks)
        student_bucket = self.student_chunk_id.masked_fill(self.student_chunk_id < 0, num_chunks)
        shape = (teacher_logprobs.shape[0], num_chunks + 1)
        chunk_logprob = torch.zeros(shape).scatter_add_(1, teacher_bucket, teacher_logprobs.float())
        teacher_count = torch.zeros(shape).scatter_add_(1, teacher_bucket, torch.ones(teacher_bucket.shape))
        student_count = torch.zeros(shape).scatter_add_(1, student_bucket, torch.ones(student_bucket.shape))
        per_token = (chunk_logprob / student_count.clamp(min=1)).gather(1, student_bucket)
        # A chunk needs tokens on both sides; one with an empty teacher side carries no teacher logprob.
        has_teacher = teacher_count.gather(1, student_bucket) > 0
        return per_token.masked_fill(~((self.student_chunk_id >= 0) & has_teacher), float("nan"))


class XTokenTeacherViewBuilder:
    """Builds the :class:`TeacherView` of each rollout group for X-OPD.

    X-OPD is OPD from a teacher whose tokenizer differs from the policy's. The teacher sees its own chat template of the
    prompt followed by the policy's decoded response, so both tokenizations cover the same response text. They are
    aligned with :class:`TokenAligner`; a teacher token that straddles the prompt/response boundary counts as prompt.
    """

    def __init__(
        self, student_tokenizer: PreTrainedTokenizerBase, cfg: XTokenOPDConfig, generation_config: Mapping[str, Any]
    ):
        """Load the teacher tokenizer and set up the aligner.

        Args:
            student_tokenizer: Policy tokenizer.
            cfg: ``on_policy_distillation.xtoken``.
            generation_config: ``policy.generation``. Its ``stop_token_ids`` and the policy EOS are stripped from the
                end of a response before alignment, since the teacher never scores an end of turn.
        """
        teacher_tokenizer = get_tokenizer(cfg["teacher_tokenizer"])
        if not teacher_tokenizer.is_fast:
            raise ValueError("X-OPD needs a fast teacher tokenizer (offset mappings)")
        eos = student_tokenizer.eos_token_id
        stop_token_ids = set(eos) if isinstance(eos, list) else {eos}
        if "stop_token_ids" in generation_config and generation_config["stop_token_ids"] is not None:
            stop_token_ids.update(generation_config["stop_token_ids"])
        self.stop_token_ids = frozenset(token for token in stop_token_ids if token is not None)
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.aligner = TokenAligner(
            student_tokenizer,
            teacher_tokenizer,
            cfg["alignment_max_combination_length"],
            num_workers=cfg["alignment_workers"] if "alignment_workers" in cfg else None,
        )

    def __call__(self, batch: BatchedDataDict, student_width: int) -> TeacherView:
        """Tokenize one rollout group for the teacher and align it with the policy's tokens.

        Args:
            batch: One NeMo-Gym rollout group (``message_log`` and ``full_result``).
            student_width: Width of the group's flattened policy tokens.

        Returns:
            The group's :class:`TeacherView`. Skipped rollouts and their reasons are printed.
        """
        prompt_lengths, student_responses, teacher_rows, teacher_starts = [], [], [], []
        skipped: dict[int, str] = {}
        for index, messages in enumerate(batch["message_log"]):
            params = batch["full_result"][index]["responses_create_params"]
            response = messages[1]["token_ids"].tolist() if len(messages) == 2 else []
            while response and response[-1] in self.stop_token_ids:
                response = response[:-1]
            reason = self._skip_reason(params, messages, response)
            teacher_ids: list[int] = []
            start = 0
            if reason is None:
                text = self.student_tokenizer.decode(
                    response, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                if "�" in text:
                    reason = "student_undecodable"
                else:
                    prefix = cast(
                        str,
                        self.teacher_tokenizer.apply_chat_template(
                            params["input"], tokenize=False, add_generation_prompt=True
                        ),
                    )
                    encoded = self.teacher_tokenizer(
                        prefix + text, add_special_tokens=False, return_offsets_mapping=True
                    )
                    teacher_ids = list(encoded["input_ids"])
                    start = next(
                        (
                            pos
                            for pos, (lo, hi) in enumerate(encoded["offset_mapping"])
                            if lo >= len(prefix) and hi > lo
                        ),
                        len(teacher_ids),
                    )
                    if _has_structural_special_token(self.teacher_tokenizer, teacher_ids[start:]):
                        reason = "teacher_special_token"
            if reason is not None:
                skipped[index] = reason
                response, teacher_ids, start = [], [], 0
            prompt_lengths.append(len(messages[0]["token_ids"]) if messages else 0)
            student_responses.append(response)
            teacher_rows.append(teacher_ids)
            teacher_starts.append(start)
        if skipped:
            reasons = ", ".join(f"{index}:{reason}" for index, reason in sorted(skipped.items()))
            print(
                f"[X-OPD] Teacher skips {len(skipped)}/{len(teacher_rows)} rollouts (zero OPD advantage): {reasons}",
                flush=True,
            )

        input_ids = _pad(teacher_rows, self.teacher_tokenizer.pad_token_id)
        teacher_chunk_id = torch.full_like(input_ids, -1)
        student_chunk_id = torch.full((len(teacher_rows), student_width), -1, dtype=torch.long)
        teacher_responses = [row[start:] for row, start in zip(teacher_rows, teacher_starts, strict=True)]
        if any(student_responses):
            # Align the responses only; prompts never enter the alignment.
            student_ids = _pad(student_responses, self.student_tokenizer.pad_token_id)
            teacher_ids_padded = _pad(teacher_responses, self.teacher_tokenizer.pad_token_id)
            alignment = self.aligner.align(
                student_ids,
                teacher_ids_padded,
                student_attention_mask=_length_mask(student_responses, student_ids.shape[1]),
                teacher_attention_mask=_length_mask(teacher_responses, teacher_ids_padded.shape[1]),
            )
            exact = alignment.pair_valid & alignment.pair_is_correct
            # Move the response-local chunk IDs to full-sequence coordinates.
            for index, response in enumerate(student_responses):
                start, count = prompt_lengths[index], len(response)
                student_chunk_id[index, start : start + count] = _exact_chunks(
                    alignment.student_chunk_id[index, :count], exact[index]
                )
                start, count = teacher_starts[index], len(teacher_responses[index])
                teacher_chunk_id[index, start : start + count] = _exact_chunks(
                    alignment.teacher_chunk_id[index, :count], exact[index]
                )
        input_lengths = torch.tensor([len(row) for row in teacher_rows], dtype=torch.long)
        return TeacherView(input_ids, input_lengths, teacher_chunk_id, student_chunk_id)

    def _skip_reason(
        self, params: Mapping[str, Any], messages: list[dict[str, Any]], response: list[int]
    ) -> Optional[str]:
        """Why the teacher cannot reproduce this rollout, or ``None`` if it can."""
        if not messages:
            return "empty_message_log"
        if _UNSUPPORTED_REQUEST_KEYS & set(params):
            return "tools_or_instructions"
        prompt = params["input"]
        if not isinstance(prompt, list) or not all(
            isinstance(message, dict) and "content" in message and isinstance(message["content"], str)
            for message in prompt
        ):
            return "non_text_prompt"
        if len(messages) > 2 or (len(messages) == 2 and messages[1]["role"] != "assistant"):
            return "multi_turn"
        # The teacher prompt is rebuilt from the prompt messages, so the policy must have seen exactly their template.
        expected = self.student_tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        if messages[0]["token_ids"].tolist() != list(expected):
            return "prompt_not_chat_template"
        if not response:
            return "empty_response"
        if _has_structural_special_token(self.student_tokenizer, response):
            return "student_special_token"
        return None
