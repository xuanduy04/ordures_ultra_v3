"""Export fixtures and numerical merge using the reviewed Super byte-copy code."""

import ast
import concurrent.futures
from dataclasses import dataclass, field
import glob
import importlib.util
import io
import json
import logging
import math
import mmap
import os
from pathlib import Path
import struct
import sys
import time
from types import SimpleNamespace
from typing import Any, Optional
import zipfile

import numpy as np
import pytest
from safetensors.numpy import load_file
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/xtoken_all"))
spec = importlib.util.spec_from_file_location("checkpoint_export_test", ROOT / "tools/xtoken_all/export_checkpoint.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def shard(path, values, offset):
    values = np.asarray(values, dtype=np.float32)
    metadata = {"model.embed_tokens.weight": {"dtype": "F32", "shape": list(values.shape), "data_offsets": [0, values.nbytes]},
        "__metadata__": {"DCP_SHARDING_INFO": json.dumps({"model.embed_tokens.weight": {"saved_offsets": offset}})}}
    raw = json.dumps(metadata).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + values.tobytes())


@pytest.fixture
def checkpoint(tmp_path):
    step = tmp_path / "step_20"
    weights = step / "policy/weights/model"
    metadata = weights / ".hf_metadata"
    metadata.mkdir(parents=True)
    (step / "training_info.json").write_text('{"total_steps":20}')
    shard(weights / "shard-00001-model-00001-of-00001.safetensors", [[1, 2], [3, 4]], [0, 0])
    shard(weights / "shard-00002-model-00001-of-00001.safetensors", [[5, 6], [7, 8]], [2, 0])
    (metadata / "fqn_to_file_index_mapping.json").write_text('{"model.embed_tokens.weight":1}')
    (metadata / "config.json").write_text('{"model_type":"qwen3"}')
    tokenizer = step / "policy/tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer_config.json").write_text('{"fixture":true}')
    settings = yaml.safe_load((ROOT / "tools/xtoken_all/export_checkpoint.yaml").read_text())
    return step, tmp_path / "hf-export", settings


def reviewed_consolidator(monkeypatch):
    if "XTOKEN_COMPANY_ZIP" not in os.environ:
        pytest.skip("Set XTOKEN_COMPANY_ZIP to the already-staged company source ZIP")
    archive_path = Path(os.environ["XTOKEN_COMPANY_ZIP"])
    with zipfile.ZipFile(archive_path) as archive:
        hf_source = archive.read(next(name for name in archive.namelist() if name.endswith("_backports/hf_utils.py"))).decode()
        source = archive.read(next(name for name in archive.namelist() if name.endswith("_backports/consolidate_hf_safetensors.py"))).decode()
    namespace = {"__name__": __name__, "json": json, "struct": struct, "io": io, "Any": Any, "Optional": Optional,
        "dataclass": dataclass, "field": field, "concurrent": concurrent, "glob": glob,
        "logging": logging, "math": math, "mmap": mmap, "os": os, "time": time,
        "torch": SimpleNamespace(finfo=lambda dtype: SimpleNamespace(bits=32)),
        "dist": SimpleNamespace(ProcessGroup=object)}
    selected = []
    for node in ast.parse(hf_source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {
            "FILE_NAME", "SHARDED_FILE_NAME", "SUFFIX", "CUSTOM_METADATA_KEY", "DEFAULT_EXTRA_METADATA_KEY",
            "SAVED_OFFSETS_KEY", "SHAPE_KEY", "DTYPE_KEY", "DATA_OFFSETS_KEY"}:
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in {"_get_dcp_custom_metadata", "_gen_file_name", "_get_safetensors_file_metadata"}:
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "reviewed_hf_utils", "exec"), namespace)
    namespace["_metadata_fn"] = "model.safetensors.index.json"
    definitions = [node for node in ast.parse(source).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "reviewed_super_consolidator", "exec"), namespace)
    namespace["logger"] = logging.getLogger("export-fixture")
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(_getdtype=lambda name: name))
    return SimpleNamespace(consolidate_safetensors_files=namespace["consolidate_safetensors_files"], __file__=str(archive_path))


def test_real_reviewed_consolidator_recovers_shard_offsets_numerically(checkpoint, monkeypatch):
    step, output, settings = checkpoint
    module = reviewed_consolidator(monkeypatch)
    monkeypatch.setattr(exporter.importlib, "import_module", lambda name: module)
    before = {p.name: p.read_bytes() for p in (step / "policy/weights/model").glob("*.safetensors")}
    result = exporter.export(step, output, settings, None, False)
    tensor = load_file(str(output / "model-00001-of-00001.safetensors"))["model.embed_tokens.weight"]
    np.testing.assert_array_equal(tensor, np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32))
    assert result["success"] and result["source_policy_numerical_parity_verified"] is False
    assert result["hf_load_validation"] is None
    assert json.loads((output / "tokenizer_config.json").read_text()) == {"fixture": True}
    assert all(p.read_bytes() == before[p.name] for p in (step / "policy/weights/model").glob("*.safetensors"))


@pytest.mark.parametrize("failure", ["mapping", "config", "tokenizer", "output", "header"])
def test_unsafe_or_incomplete_exports_fail_before_conversion(checkpoint, failure):
    step, output, settings = checkpoint
    metadata = step / "policy/weights/model/.hf_metadata"
    if failure == "mapping":
        (metadata / "fqn_to_file_index_mapping.json").write_text('{"wrong":1}')
    elif failure == "config":
        (metadata / "config.json").unlink()
    elif failure == "tokenizer":
        (step / "policy/tokenizer/tokenizer_config.json").unlink()
    elif failure == "output":
        output.mkdir()
    else:
        next((step / "policy/weights/model").glob("*.safetensors")).write_bytes(b"bad")
    with pytest.raises((ValueError, FileExistsError)):
        exporter.export(step, output, settings, None, False)
    assert not (output / "export-receipt.json").exists()


def test_hf_load_failure_never_produces_success_receipt(checkpoint, monkeypatch):
    step, output, settings = checkpoint
    module = reviewed_consolidator(monkeypatch)
    monkeypatch.setattr(exporter.importlib, "import_module", lambda name: module)
    def fail_load(*args):
        raise ValueError("HF shape mismatch")
    monkeypatch.setattr(exporter, "validate_load", fail_load)
    with pytest.raises(ValueError, match="shape mismatch"):
        exporter.export(step, output, settings, None, True)
    assert (output / "FAILED").exists() and not (output / "export-receipt.json").exists()
