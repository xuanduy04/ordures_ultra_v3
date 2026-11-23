import json
from bisect import bisect_right

from tokenizers.decoders import ByteLevel, Decoder
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

_BYTE_LEVEL_TO_BYTE = {char: value for value, char in bytes_to_unicode().items()}


def uses_byte_level_decoder(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Whether the student backend declares a ByteLevel decoder."""
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return False
    backend = tokenizer.backend_tokenizer
    decoder = backend.decoder
    if decoder is None or isinstance(decoder, Decoder):
        return isinstance(decoder, ByteLevel)
    # Accelerated backends can wrap the decoder. Inspect the declared pipeline once per collator.
    try:
        return json.loads(backend.to_str())["decoder"]["type"] == "ByteLevel"
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _byte_level_response_prefix(
    tokenizer: PreTrainedTokenizerBase, response: list[int], text: str
) -> tuple[list[int], str] | None:
    """Return the longest verifiable UTF-8 prefix ending at a sampled token boundary."""
    # ByteLevel decodes the joined bytes lossily. A replacement character can also be literal UTF-8.
    raw = bytearray()
    token_ends = []
    for piece in tokenizer.convert_ids_to_tokens(response):
        if piece is None:
            return None
        try:
            raw.extend(bytes(_BYTE_LEVEL_TO_BYTE[char] for char in piece))
        except KeyError:
            # ByteLevel treats the entire piece as literal UTF-8 if it contains a character outside its alphabet.
            raw.extend(piece.encode("utf-8"))
        token_ends.append(len(raw))
    if raw.decode("utf-8", errors="replace") != text:
        return None
    try:
        return response, raw.decode("utf-8")
    except UnicodeDecodeError as error:
        valid_end = error.start

    # The token before the first bad byte can itself end inside a valid character completed by the next token.
    count = bisect_right(token_ends, valid_end)
    while count and token_ends[count - 1] < valid_end and raw[token_ends[count - 1]] & 0xC0 == 0x80:
        count -= 1
    if not count or not token_ends[count - 1]:
        return None
    prefix, prefix_text = response[:count], raw[: token_ends[count - 1]].decode("utf-8")
    if tokenizer.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False) != prefix_text:
        return None
    return prefix, prefix_text


def response_control_tokens(tokenizer, response):
    """Classify special-token occurrences without decoding or rewriting content.

    Qwen reasoning delimiters are visible content, including an unmatched delimiter in a length-limited rollout. Role,
    padding and other structural controls are not safe response content. Callers mask their sample rather than joining
    the surrounding tokens into a different training sequence.
    """
    special_ids = set(tokenizer.all_special_ids)
    positions = [index for index, token in enumerate(response) if token in special_ids]
    if not positions:
        return []
    pieces = tokenizer.convert_ids_to_tokens([response[index] for index in positions])
    return [
        {
            "response_position": index,
            "token_id": int(response[index]),
            "token": str(piece),
            "action": "retain_reasoning" if piece in {"<think>", "</think>"} else "mask_sample",
        }
        for index, piece in zip(positions, pieces)
    ]


def response_for_alignment(tokenizer, response, needs_text, *, byte_level):
    """Return an alignment view; never repair or trim the sampled sequence."""
    if not needs_text:
        return response, "", None
    text = tokenizer.decode(response, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if "\ufffd" not in text:
        return response, text, None
    if byte_level:
        decoded = _byte_level_response_prefix(tokenizer, response, text)
        if decoded is not None:
            prefix, prefix_text = decoded
            return prefix, prefix_text, None
    return [], "", "replacement_character_in_student_decode"
