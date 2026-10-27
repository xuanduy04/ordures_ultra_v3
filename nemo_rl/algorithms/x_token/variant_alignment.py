"""Byte-exact synchronized atoms, opt-in alongside legacy projection alignment."""

import re

import torch

from nemo_rl.algorithms.x_token.token_aligner import AlignmentBatch


def byte_decoder():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, extra = values.copy(), 256
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(extra)
            extra += 1
    return {chr(char): value for char, value in zip(chars, values)}


_BYTE_DECODER = byte_decoder()


def piece_codec(tokenizer):
    backend = tokenizer.backend_tokenizer
    description = repr(backend.pre_tokenizer) + repr(backend.decoder)
    if "ByteLevel" in description:
        def decode(piece):
            return b"".join(bytes([_BYTE_DECODER[ch]]) if ch in _BYTE_DECODER else ch.encode("utf-8") for ch in piece)
        return decode
    if "Metaspace" in description or "ByteFallback" in description or "Gemma" in type(tokenizer).__name__:
        def decode(piece):
            if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
                return bytes([int(piece[3:5], 16)])
            return piece.replace("▁", " ").encode("utf-8")
        return decode
    raise ValueError("Variant byte alignment requires a ByteLevel or SentencePiece-compatible tokenizer")


def shared_ids(student, teacher):
    def vocabulary(tokenizer):
        decode = piece_codec(tokenizer)
        excluded = set(tokenizer.all_special_ids)
        candidates = {}
        for piece, token in tokenizer.get_vocab().items():
            if token in excluded or re.fullmatch(r"<unused\d+>", piece):
                continue
            key = decode(piece)
            if key:
                candidates.setdefault(key, set()).add(token)
        return {key: next(iter(ids)) for key, ids in candidates.items() if len(ids) == 1}
    s, t = vocabulary(student), vocabulary(teacher)
    pairs = [(s[key], t[key]) for key in sorted(s.keys() & t.keys())]
    if student.eos_token_id is not None and teacher.eos_token_id is not None:
        pairs.append((student.eos_token_id, teacher.eos_token_id))
    if not pairs:
        raise ValueError("The tokenizer pair has no safe shared vocabulary")
    return [a for a, b in pairs], [b for a, b in pairs]


def synchronized_segments(student_pieces, teacher_pieces):
    """Smallest nonempty token spans ending at a common byte boundary."""
    if b"".join(student_pieces) != b"".join(teacher_pieces):
        raise ValueError("Student and teacher response bytes differ; no approximate alignment")
    def boundaries(pieces):
        output, size = {}, 0
        for index, piece in enumerate(pieces):
            if not piece:
                raise ValueError("Zero-byte content tokens cannot define synchronized chunks")
            size += len(piece)
            output[size] = index + 1
        return output
    s, t = boundaries(student_pieces), boundaries(teacher_pieces)
    segments, previous_s, previous_t = [], 0, 0
    for boundary in sorted(s.keys() & t.keys()):
        segments.append((previous_t, t[boundary], previous_s, s[boundary]))
        previous_s, previous_t = s[boundary], t[boundary]
    return segments


class SynchronizedAligner:
    """Keep native X-Token chunks as GRASS groups, minimal byte atoms as segments."""

    def __init__(self, student, teacher, native):
        self.student, self.teacher, self.native = student, teacher, native
        self.student_decode, self.teacher_decode = piece_codec(student), piece_codec(teacher)

    def align(self, student_ids, teacher_ids, *, student_attention_mask, teacher_attention_mask):
        native = self.native.align(student_ids, teacher_ids,
            student_attention_mask=student_attention_mask, teacher_attention_mask=teacher_attention_mask)
        rows = []
        for row in range(len(student_ids)):
            s = student_ids[row][student_attention_mask[row].bool()].tolist()
            t = teacher_ids[row][teacher_attention_mask[row].bool()].tolist()
            rows.append(synchronized_segments(
                [self.student_decode(piece) for piece in self.student.convert_ids_to_tokens(s)],
                [self.teacher_decode(piece) for piece in self.teacher.convert_ids_to_tokens(t)]))
        width = max((len(row) for row in rows), default=0)
        valid = torch.zeros((len(rows), width), dtype=torch.bool)
        s_chunks, t_chunks = torch.full_like(student_ids, -1), torch.full_like(teacher_ids, -1)
        self.native_groups = torch.full((len(rows), width), -1, dtype=torch.long)
        for row, segments in enumerate(rows):
            for index, (ts, te, ss, se) in enumerate(segments):
                valid[row, index] = True
                s_chunks[row, ss:se] = index
                t_chunks[row, ts:te] = index
                s_native = native.student_chunk_id[row, ss:se].unique()
                t_native = native.teacher_chunk_id[row, ts:te].unique()
                if len(s_native) == len(t_native) == 1 and int(s_native[0]) == int(t_native[0]) >= 0:
                    group = int(s_native[0])
                    if bool(native.pair_valid[row, group] & native.pair_is_correct[row, group]):
                        self.native_groups[row, index] = group
        return AlignmentBatch(valid, valid.clone(), s_chunks >= 0, t_chunks >= 0,
            s_chunks, t_chunks, torch.tensor([len(row) for row in rows], dtype=torch.long))


def wrap_aligners(aligners, student, teachers):
    return [None if aligner is None else SynchronizedAligner(student, teachers[i], aligner)
            for i, aligner in enumerate(aligners)]
