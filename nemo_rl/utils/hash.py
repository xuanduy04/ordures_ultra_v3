import hashlib
import struct


_U64 = struct.Struct("<Q")


def hash_list_dicts(obj: list[dict[str, str]]) -> int:
    h = hashlib.blake2b(digest_size=64)

    update = h.update
    pack_u64 = _U64.pack

    update(pack_u64(len(obj)))

    for d in obj:
        update(pack_u64(len(d)))

        for key in sorted(d):
            kb = key.strip().encode("utf-8")
            value = d[key]
            vb = value.strip().encode("utf-8") if isinstance(value, str) else b""

            update(pack_u64(len(kb)))
            update(kb)
            update(pack_u64(len(vb)))
            update(vb)

    return int.from_bytes(h.digest(), "little")


def get_user_turns(obj: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the non-empty ``user`` turns of input conversation."""
    return [
        {
            "role": turn["role"].strip(),
            "content": turn["content"].strip()
        } for turn in obj
        if isinstance(turn.get("content"), str)
        and turn["content"].strip()
        and isinstance(turn.get("role"), str)
        and turn["role"].strip() == "user"
    ]
