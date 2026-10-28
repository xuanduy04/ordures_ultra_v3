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
    verify_company_source(root)
    files = {}
    for directory in ("nemo_rl", "examples", "custom_utils", "tools/xtoken_all"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".yml", ".json"} and "__pycache__" not in path.parts:
                files[path.relative_to(root).as_posix()] = sha256(path)
    for name in ("pyproject.toml", "uv.lock", "default_runtime_env.yaml", ".python-version", "run_xtoken.sh", "ensure_runtime.sh"):
        path = root / name
        if path.is_file():
            files[name] = sha256(path)
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "files": files}


def verify_company_source(root):
    """Enforce the pinned legacy modules when this source carries that contract."""
    root = Path(root).resolve()
    lineage = root / "tools/xtoken_all/super-runtime-lineage.json"
    if not lineage.is_file():
        return None
    data = json.loads(lineage.read_text(encoding="utf-8"))
    if "company_baseline" not in data:
        return None
    baseline = data["company_baseline"]
    for relative, expected in baseline["normalized_lf_sha256"].items():
        path = (root / relative).resolve()
        path.relative_to(root)
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError("Protected company module differs from " + baseline["commit"] + ": " + relative)
    return {"commit": baseline["commit"], "files_checked": len(baseline["normalized_lf_sha256"])}
