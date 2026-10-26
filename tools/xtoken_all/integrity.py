"""Stream fingerprints of executable source without reading runtime secrets."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_snapshot(root):
    root = Path(root).resolve()
    files = {}
    for directory in ("nemo_rl", "examples", "custom_utils", "tools/xtoken_all"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".yml"} and "__pycache__" not in path.parts:
                files[path.relative_to(root).as_posix()] = sha256(path)
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "files": files}
