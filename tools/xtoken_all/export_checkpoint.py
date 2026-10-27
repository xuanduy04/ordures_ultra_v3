"""Export Super Automodel sharded safetensors using its installed consolidator."""

import argparse
import importlib
import inspect
import json
from pathlib import Path
import shutil
import struct

import yaml

from integrity import sha256


def header(path, limit):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors length: " + str(path))
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > limit or size > path.stat().st_size - 8:
            raise ValueError("Invalid safetensors header length")
        return json.loads(stream.read(size))


def inspect_checkpoint(checkpoint, settings):
    checkpoint = checkpoint.resolve()
    training = json.loads((checkpoint / "training_info.json").read_text())
    if checkpoint.name != "step_" + str(training["total_steps"]):
        raise ValueError("Export a finalized step_N checkpoint matching training_info")
    weights = checkpoint / "policy/weights/model"
    metadata = weights / ".hf_metadata"
    files = sorted(weights.glob("*.safetensors"))
    if not files:
        raise ValueError("No Automodel safetensors shards; do not use this utility for native .distcp")
    keys = set()
    for path in files:
        keys.update(key for key in header(path, settings["header_limit_bytes"]) if key != "__metadata__")
    return {"checkpoint": str(checkpoint), "step": training["total_steps"], "weights": str(weights), "shard_count": len(files),
        "source_tensor_names": sorted(keys), "source_bytes": sum(p.stat().st_size for p in files),
        "hf_metadata": str(metadata), "config_present": (metadata / "config.json").is_file(),
        "mapping_present": (metadata / "fqn_to_file_index_mapping.json").is_file(),
        "tokenizer_present": (checkpoint / "policy/tokenizer/tokenizer_config.json").is_file(),
        "inspection_only": True}


def validate_load(output, text):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model, info = AutoModelForCausalLM.from_pretrained(str(output), local_files_only=True,
        torch_dtype="auto", attn_implementation="eager", output_loading_info=True)
    if any(info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("HF loading reports missing/unexpected/mismatched keys or errors: " + str(info))
    tokenizer = AutoTokenizer.from_pretrained(str(output), local_files_only=True)
    inputs = tokenizer(text, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Non-finite logits after CPU HF loading")
    return {"success": True, "device": "cpu", "logits_shape": list(logits.shape),
            "loading_info": info, "scope": "HF load + finite forward, not source-policy numerical parity"}


def export(checkpoint, output, settings, base_model, check_load):
    report = inspect_checkpoint(checkpoint, settings)
    checkpoint, output = checkpoint.resolve(), output.resolve()
    if output.exists() or checkpoint == output or checkpoint in output.parents or output in checkpoint.parents:
        raise FileExistsError("Use a new export directory outside the source checkpoint")
    weights, metadata = Path(report["weights"]), Path(report["hf_metadata"])
    mapping_path = metadata / "fqn_to_file_index_mapping.json"
    if not mapping_path.is_file():
        raise ValueError("Missing .hf_metadata mapping: inspect hidden files; no shard-name guessing")
    mapping = json.loads(mapping_path.read_text())
    if not mapping or set(mapping) != set(report["source_tensor_names"]):
        raise ValueError("HF mapping and saved tensor names differ")
    if any(type(index) is not int or index < 0 for index in mapping.values()):
        raise ValueError("Invalid HF output shard mapping")
    config_path = metadata / "config.json"
    config_origin = "checkpoint_hf_metadata"
    if not config_path.is_file():
        if base_model is None:
            raise ValueError("Missing saved HF config; supply --base-model only after checking the original training architecture")
        config_path = base_model / "config.json"
        config_origin = "explicit_base_model"
    model_config = json.loads(config_path.read_text())
    if model_config["model_type"] != "qwen3":
        raise ValueError("This export qualification is scoped to Qwen3, not model-family remapping")
    tokenizer = checkpoint / "policy/tokenizer"
    if not (tokenizer / "tokenizer_config.json").is_file():
        raise ValueError("Missing checkpoint tokenizer; do not substitute a different tokenizer")
    module = importlib.import_module("nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors")
    consolidate = module.consolidate_safetensors_files
    required = {"input_dir", "output_dir", "fqn_to_index_mapping", "num_threads"}
    if not required.issubset(inspect.signature(consolidate).parameters):
        raise ValueError("Installed Automodel consolidation API differs from reviewed Super source")
    files = sorted(weights.glob("*.safetensors"))
    stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in files}
    hashes = {p.name: sha256(p) for p in files} if settings["hash_source_weights"] else None
    output.mkdir(parents=True, exist_ok=False)
    try:
        consolidate(input_dir=str(weights), output_dir=str(output), fqn_to_index_mapping=mapping,
                    num_threads=settings["num_threads"])
        for source in (config_path, metadata / "generation_config.json"):
            if source.is_file():
                shutil.copy2(source, output / source.name)
        for source in tokenizer.iterdir():
            if source.is_file() and (source.suffix in (".json", ".jinja", ".model") or source.name in ("merges.txt", "vocab.txt")):
                if source.name in ("config.json", "generation_config.json"):
                    continue
                shutil.copy2(source, output / source.name)
        index_path = output / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        if set(index["weight_map"]) != set(mapping):
            raise ValueError("Consolidated HF index does not contain exactly the saved tensors")
        output_keys = set()
        for name in set(index["weight_map"].values()):
            path = (output / name).resolve()
            path.relative_to(output)
            data = header(path, settings["header_limit_bytes"])
            output_keys.update(key for key in data if key != "__metadata__")
        if output_keys != set(mapping):
            raise ValueError("Consolidated files differ from HF index")
        if any((p.stat().st_size, p.stat().st_mtime_ns) != stats[str(p)] for p in files):
            raise ValueError("Source shards changed during export")
        if hashes is not None and any(sha256(p) != hashes[p.name] for p in files):
            raise ValueError("Source weights changed during export")
        load = validate_load(output, settings["probe_text"]) if check_load else None
        result = {"success": True, "source_checkpoint": str(checkpoint), "output": str(output),
            "source_weights_sha256": hashes, "source_weights_modified": False,
            "mapping_sha256": sha256(mapping_path), "config_sha256": sha256(config_path),
            "config_origin": config_origin, "tensor_count": len(mapping),
            "consolidator_module": str(Path(module.__file__).resolve()),
            "consolidator_sha256": sha256(module.__file__), "hf_load_validation": load,
            "source_policy_numerical_parity_verified": False}
        (output / "export-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception:
        (output / "FAILED").write_text("Incomplete or unvalidated export; do not serve.\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--validate-load", action="store_true")
    args = parser.parse_args()
    settings_path = args.settings if args.settings is not None else Path(__file__).with_suffix(".yaml")
    settings = yaml.safe_load(settings_path.read_text())
    if args.inspect:
        result = inspect_checkpoint(args.checkpoint, settings)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required for conversion")
        result = export(args.checkpoint, args.output_dir, settings, args.base_model, args.validate_load)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
