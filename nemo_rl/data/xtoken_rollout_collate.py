"""Preserve sampled student IDs while aligning response-only teacher tokens."""

from dataclasses import fields

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class PromptCollator:
    """Carry prompts through prefetch workers without precomputing rollouts."""

    def __init__(self, prompt_source):
        if prompt_source not in {"raw_text", "messages"}:
            raise ValueError("prompt_source must be raw_text or messages")
        self.prompt_source = prompt_source

    def __call__(self, data):
        prompts = []
        for datum in data:
            messages = datum["message_log"]
            if self.prompt_source == "raw_text":
                if len(messages) != 1:
                    raise ValueError("raw_text expects exactly one text message")
                prompt = [{"role": "user", "content": messages[0]["content"]}]
            else:
                prompt = [{"role": item["role"], "content": item["content"]} for item in messages]
            if not prompt or prompt[-1]["role"] != "user":
                raise ValueError("An on-policy prompt must end with a user message")
            prompts.append(prompt)
        return BatchedDataDict(
            prompts=prompts,
            sample_mask=torch.tensor([item["loss_multiplier"] for item in data], dtype=torch.float32),
            idx=[item["idx"] for item in data],
        )


def _pad_ids(rows, tokenizer, limit, divisor):
    length = ((max(map(len, rows)) + divisor - 1) // divisor) * divisor
    if length > limit:
        raise ValueError(f"Sequence requires {length} tokens but limit is {limit}; no silent truncation")
    length = max(length, divisor)
    result = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
    for index, row in enumerate(rows):
        result[index, :len(row)] = torch.tensor(row, dtype=torch.long)
    return result


class RolloutCrossTokenizerCollator:
    """Align sampled responses independently of the two model chat prefixes.

    Student sequences are copied from generation output. Teachers receive their
    own chat prefix and the decoded response. Offset mappings exclude any
    teacher token straddling the response boundary; non-exact alignment chunks
    are excluded. Terminal control tokens remain context and carry zero loss.
    """

    def __init__(self, student_tokenizer, teacher_tokenizers, aligners,
                 student_limit, teacher_limits, student_divisor, teacher_divisors,
                 stop_token_ids):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizers = teacher_tokenizers
        self.aligners = aligners
        self.student_limit = student_limit
        self.teacher_limits = teacher_limits
        self.student_divisor = student_divisor
        self.teacher_divisors = teacher_divisors
        self.stop_token_ids = set(stop_token_ids)
        if not (len(teacher_tokenizers) == len(aligners) == len(teacher_limits) == len(teacher_divisors)):
            raise ValueError("Teacher metadata lengths differ")
        for tokenizer in [student_tokenizer, *teacher_tokenizers]:
            if tokenizer.pad_token_id is None:
                raise ValueError("Tokenizers must have an explicit pad token")
        if any(value <= 0 for value in [student_limit, student_divisor, *teacher_limits, *teacher_divisors]):
            raise ValueError("Sequence limits and padding divisors must be positive")

    def __call__(self, prompt_batch, generation, prompt_ids, policy_version, rollout_id):
        rows, responses, texts, prompt_lengths = [], [], [], []
        for index, prompt in enumerate(prompt_ids):
            total = int(generation["unpadded_sequence_lengths"][index])
            if total < len(prompt) or total > generation["output_ids"].shape[1]:
                raise ValueError("Generation returned an invalid unpadded sequence length")
            row = generation["output_ids"][index, :total].cpu().tolist()
            if row[:len(prompt)] != prompt:
                raise ValueError("Generation output changed the sampled student prompt IDs")
            response = row[len(prompt):]
            if response and response[-1] in self.stop_token_ids:
                response = response[:-1]
            if any(token in self.student_tokenizer.all_special_ids for token in response):
                raise ValueError("A non-terminal control token appeared in the response")
            text = self.student_tokenizer.decode(
                response, skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
            if "\ufffd" in text:
                raise ValueError("Response has replacement characters; teacher text cannot preserve its bytes")
            rows.append(row)
            responses.append(response)
            texts.append(text)
            prompt_lengths.append(len(prompt))
        student_ids = _pad_ids(rows, self.student_tokenizer, self.student_limit, self.student_divisor)
        token_mask = torch.zeros_like(student_ids)
        for index, response in enumerate(responses):
            start = prompt_lengths[index]
            token_mask[index, start:start + len(response)] = 1
        sample_mask = prompt_batch["sample_mask"].clone()
        sample_mask *= torch.tensor([bool(response) for response in responses], dtype=sample_mask.dtype)
        if not bool((sample_mask > 0).any()):
            raise ValueError("No non-empty response in this rollout batch; no optimizer update was made")
        out = BatchedDataDict(
            input_ids=student_ids,
            input_lengths=torch.tensor(list(map(len, rows)), dtype=torch.long),
            token_mask=token_mask,
            sample_mask=sample_mask,
            idx=prompt_batch["idx"],
            behavior_version=torch.full((len(rows),), policy_version, dtype=torch.long),
            rollout_id=torch.full((len(rows),), rollout_id, dtype=torch.long),
        )
        student_response_ids = _pad_ids(
            responses, self.student_tokenizer, self.student_limit, 1,
        )
        student_response_mask = torch.zeros_like(student_response_ids)
        for index, response in enumerate(responses):
            student_response_mask[index, :len(response)] = 1

        for teacher_index, aligner in enumerate(self.aligners):
            if aligner is None:
                continue
            tokenizer = self.teacher_tokenizers[teacher_index]
            if not tokenizer.is_fast:
                raise ValueError("On-policy teacher alignment requires a fast tokenizer with offset mappings")
            teacher_rows, teacher_responses, teacher_starts = [], [], []
            for index, messages in enumerate(prompt_batch["prompts"]):
                prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                encoded = tokenizer(prefix + texts[index], add_special_tokens=False, return_offsets_mapping=True)
                ids = list(encoded["input_ids"])
                offsets = encoded["offset_mapping"]
                eligible = [position for position, (lo, hi) in enumerate(offsets)
                            if lo >= len(prefix) and hi > lo]
                start = eligible[0] if eligible else len(ids)
                response = ids[start:]
                if any(token in tokenizer.all_special_ids for token in response):
                    raise ValueError("Teacher response text produced a control token")
                teacher_rows.append(ids)
                teacher_responses.append(response)
                teacher_starts.append(start)
            teacher_ids = _pad_ids(
                teacher_rows, tokenizer, self.teacher_limits[teacher_index], self.teacher_divisors[teacher_index],
            )
            teacher_mask = torch.zeros_like(teacher_ids)
            teacher_response_ids = _pad_ids(teacher_responses, tokenizer, self.teacher_limits[teacher_index], 1)
            teacher_response_mask = torch.zeros_like(teacher_response_ids)
            for index, response in enumerate(teacher_responses):
                teacher_response_mask[index, :len(response)] = 1
                teacher_mask[index, teacher_starts[index]:teacher_starts[index] + len(response)] = 1
            alignment = aligner.align(
                student_response_ids, teacher_response_ids,
                student_attention_mask=student_response_mask,
                teacher_attention_mask=teacher_response_mask,
            )
            # Only byte-equivalent response chunks survive; prefixes never enter alignment.
            alignment.pair_valid &= alignment.pair_is_correct
            if not bool((alignment.pair_valid & (sample_mask > 0).unsqueeze(-1)).any()):
                raise ValueError(f"Teacher {teacher_index}: no exact response chunk survived alignment")
            student_chunks = torch.full_like(student_ids, -1)
            teacher_chunks = torch.full_like(teacher_ids, -1)
            student_exact = torch.zeros_like(student_ids, dtype=torch.bool)
            teacher_exact = torch.zeros_like(teacher_ids, dtype=torch.bool)
            for index, response in enumerate(responses):
                start = prompt_lengths[index]
                student_chunks[index, start:start + len(response)] = alignment.student_chunk_id[index, :len(response)]
                student_exact[index, start:start + len(response)] = alignment.student_exact_partition_mask[index, :len(response)]
                count = len(teacher_responses[index])
                start = teacher_starts[index]
                teacher_chunks[index, start:start + count] = alignment.teacher_chunk_id[index, :count]
                teacher_exact[index, start:start + count] = alignment.teacher_exact_partition_mask[index, :count]
            alignment.student_chunk_id = student_chunks
            alignment.teacher_chunk_id = teacher_chunks
            alignment.student_exact_partition_mask = student_exact
            alignment.teacher_exact_partition_mask = teacher_exact
            out[f"teacher_{teacher_index}_input_ids"] = teacher_ids
            out[f"teacher_{teacher_index}_input_lengths"] = torch.tensor(list(map(len, teacher_rows)), dtype=torch.long)
            out[f"teacher_{teacher_index}_token_mask"] = teacher_mask
            for field in fields(alignment):
                out[f"alignment_{teacher_index}_{field.name}"] = getattr(alignment, field.name)
        return out
