"""Keep sampled IDs intact when text alignment cannot preserve a response."""


def response_control_tokens(tokenizer, response):
    """Classify special-token occurrences without decoding or rewriting content.

    Qwen reasoning delimiters are visible content, including an unmatched
    delimiter in a length-limited rollout. Role, padding and other structural
    controls are not safe response content. Callers mask their sample rather
    than joining the surrounding tokens into a different training sequence.
    """
    special_ids = set(tokenizer.all_special_ids)
    positions = [index for index, token in enumerate(response) if token in special_ids]
    if not positions:
        return []
    pieces = tokenizer.convert_ids_to_tokens([response[index] for index in positions])
    return [{"response_position": index, "token_id": int(response[index]),
             "token": str(piece), "action": "retain_reasoning" if piece in {"<think>", "</think>"} else "mask_sample"}
            for index, piece in zip(positions, pieces)]


def response_for_alignment(tokenizer, response, needs_text):
    """Return an alignment view; never repair or trim the sampled sequence."""
    if not needs_text:
        return response, "", None
    text = tokenizer.decode(response, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if "\ufffd" in text:
        return [], "", "replacement_character_in_student_decode"
    return response, text, None
