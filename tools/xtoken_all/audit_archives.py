"""Read company snapshots and export the exact Gym source closure, not a venv."""

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

from integrity import sha256


GYM_PREFIX = "custom_opt_ultra_selfbuilt/nemo-rl/3rdparty/Gym-workspace/Gym/"
GYM_DIRS = ("nemo_gym/", "responses_api_models/vllm_model/",
            "responses_api_agents/simple_agent/", "resources_servers/general_qa/",
            "resources_servers/utils_qa/", "resources_servers/utils_outsource/")
GYM_ROOT_FILES = {"LICENSE", "ATTRIBUTIONS.md", "pyproject.toml", ".python-version"}


def canonical_sha(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def source_members(archive):
    """Select only the reviewed services and shared SDK; no environment secrets."""
    selected = {}
    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith(GYM_PREFIX):
            continue
        relative = info.filename[len(GYM_PREFIX):]
        parts = PurePosixPath(relative).parts
        if ".." in parts or relative.startswith("/") or "\\" in relative:
            raise ValueError("Unsafe archive member")
        if any(part in {"__pycache__", ".ipynb_checkpoints", ".venv"} for part in parts):
            continue
        if not (relative.startswith(GYM_DIRS) or relative in GYM_ROOT_FILES):
            continue
        if relative in selected or info.file_size > 2 * 1024 * 1024:
            raise ValueError("Duplicate/oversized SDK member")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("SDK payload must not contain symlinks")
        selected[relative] = archive.read(info)
    return selected


def build_contract(custom_zip, archive_a, archive_b, metadata_root):
    files = {}
    with zipfile.ZipFile(custom_zip) as archive:
        files = source_members(archive)
        prefix = "custom_opt_ultra_selfbuilt/nemo-rl/"
        metadata = {}
        for name in ("pyproject.toml", "uv.lock", "default_runtime_env.yaml", ".python-version"):
            data = archive.read(prefix + name)
            actual = (Path(metadata_root) / name).read_bytes()
            if canonical_sha(data) != canonical_sha(actual):
                raise ValueError("Root metadata differs: " + name)
            metadata[name] = canonical_sha(data)
        automodel = prefix + "3rdparty/Automodel-workspace/Automodel/"
        fsdp = ast.parse(archive.read(automodel + "nemo_automodel/components/distributed/fsdp2.py"))
        cls = next(node for node in fsdp.body if isinstance(node, ast.ClassDef) and node.name == "FSDP2Manager")
        fields = [node.target.id for node in cls.body if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
        checkpoint = ast.parse(archive.read(automodel + "nemo_automodel/components/checkpoint/checkpointing.py"))
        method = next(node for node in ast.walk(checkpoint) if isinstance(node, ast.FunctionDef) and node.name == "load_base_model")
        checkpoint_args = [arg.arg for arg in method.args.args + method.args.kwonlyargs]
        zip_core = {info.filename[len(prefix):]: canonical_sha(archive.read(info))
                    for info in archive.infolist() if not info.is_dir()
                    and info.filename.startswith(prefix + "nemo_rl/") and info.filename.endswith(".py")}
    shared = []
    with tarfile.open(archive_b, "r:gz") as archive:
        prefix = "super-120b/nemo-rl/custom_opt_ultra_selfbuilt/nemo-rl/"
        for member in archive:
            if member.isfile() and member.name.startswith(prefix + "nemo_rl/") and member.name.endswith(".py"):
                relative = member.name[len(prefix):]
                if relative in zip_core:
                    if canonical_sha(archive.extractfile(member).read()) != zip_core[relative]:
                        raise ValueError("ZIP and Code B disagree: " + relative)
                    shared.append(relative)
    if not shared:
        raise ValueError("Code B is not the expected Super snapshot")
    with tarfile.open(archive_a, "r:gz") as archive:
        code_a_files = sum(member.isfile() for member in archive)
    return {"schema_version": 1, "contract": "G1.1",
            "archives": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                         for path in (custom_zip, archive_a, archive_b)},
            "root_metadata_canonical_sha256": metadata,
            "gym_files": {name: canonical_sha(data) for name, data in sorted(files.items())},
            "source_api": {"fsdp2_dataclass_fields": fields, "checkpoint_load_base_model_arguments": checkpoint_args,
                           "zip_and_code_b_identical_python_files": len(shared), "code_a_files": code_a_files},
            "scope": "Source/API contract only; no installed dependency or GPU execution claim"}, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-zip", type=Path, required=True)
    parser.add_argument("--code-a", type=Path, required=True)
    parser.add_argument("--code-b", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new audit output directory")
    contract, files = build_contract(args.custom_zip, args.code_a, args.code_b, args.metadata_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "super-source-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    target = args.output_dir / "reviewed-gym-source.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    print(json.dumps({"files": len(files), "gym_source_zip_bytes": target.stat().st_size,
                      "gym_source_zip_sha256": sha256(target), "scope": contract["scope"]}, indent=2))


if __name__ == "__main__":
    main()
