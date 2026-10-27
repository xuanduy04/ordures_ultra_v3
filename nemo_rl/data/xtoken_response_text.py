"""Keep sampled IDs intact when text alignment cannot preserve a response."""


def response_for_alignment(tokenizer, response, needs_text):
    """Return an alignment view; never repair or trim the sampled sequence."""
    if not needs_text:
        return response, "", None
    text = tokenizer.decode(response, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if "\ufffd" in text:
        return [], "", "replacement_character_in_student_decode"
    return response, text, None
