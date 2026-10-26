"""Keep unchanged-server acceptance separate from measured scoring and mode proof."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


TOOLS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("teacher_probe_contract", TOOLS / "teacher_probe_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
POLICY = yaml.safe_load((TOOLS / "teacher_probe_policy.yaml").read_text())
REMOTE = {"url": "http://teacher:8000", "model": "gemma"}


def receipt():
    basis = contract.mode_basis(REMOTE, "0.31.0", None, POLICY)
    return {
        "success": True, "teacher_score_contract_satisfied": True,
        "score_request_completed": True, "tokenizer_parity_probes": 3,
        "top_k": 20, "returned_count": 21, "representation": "topk_logprobs",
        "returned_probability_mass": 0.9, "normalization_atol": 0.0001,
        "vllm_version_observed": "0.31.0",
        **{key: value for key, value in basis.items() if key != "accepted"},
    }


def evidence(mode):
    return {"teacher_url": REMOTE["url"], "served_model": REMOTE["model"],
            "logprobs_mode": mode, "verification_source": "operator_checked_server_args",
            "observed_at_utc": "2026-10-27T00:00:00+00:00",
            "evidence_description": "Operator read running server arguments"}


def test_known_default_allows_measured_scores_without_claiming_mode_proof():
    result = receipt()
    contract.validate_teacher_probe(result, POLICY)
    assert result["mode_assumption_accepted"] is True
    assert result["server_logprobs_mode_verified"] is False
    result["returned_count"] = 20
    contract.validate_teacher_probe(result, POLICY)


@pytest.mark.parametrize("version", ["0.30.0", "0.31.0+custom", "unknown"])
def test_unknown_exact_version_cannot_infer_default(version):
    assert contract.mode_basis(REMOTE, version, None, POLICY)["accepted"] is False


@pytest.mark.parametrize("mode", ["raw_logits", "processed_logits", "processed_logprobs"])
def test_explicit_incompatible_mode_never_falls_back_to_default(mode):
    result = contract.mode_basis(REMOTE, "0.31.0", evidence(mode), POLICY)
    assert result["accepted"] is False
    assert result["mode_basis"] == "deployment_config"


def test_explicit_raw_mode_is_deployment_evidence():
    result = receipt()
    result.update({key: value for key, value in
                   contract.mode_basis(REMOTE, "custom", evidence("raw_logprobs"), POLICY).items()
                   if key != "accepted"})
    contract.validate_teacher_probe(result, POLICY)
    assert result["server_logprobs_mode_verified"] is True
    assert result["mode_assumption_accepted"] is False


@pytest.mark.parametrize("field,value", [("teacher_url", "http://other"), ("served_model", "other")])
def test_evidence_must_belong_to_this_endpoint(field, value):
    item = evidence("raw_logprobs")
    item[field] = value
    with pytest.raises(ValueError, match="another teacher"):
        contract.mode_basis(REMOTE, "0.31.0", item, POLICY)


@pytest.mark.parametrize("field,value", [
    ("success", False), ("teacher_score_contract_satisfied", False),
    ("score_request_completed", False), ("tokenizer_parity_probes", 0),
    ("top_k", 19), ("returned_count", 19), ("returned_count", 22),
    ("representation", "logits"), ("returned_probability_mass", float("nan")),
    ("returned_probability_mass", 1.1), ("returned_probability_mass", 0),
    ("normalization_atol", float("inf")), ("normalization_atol", -1),
    ("effective_logprobs_mode", "raw_logits"), ("server_logprobs_mode_verified", True),
    ("mode_assumption_accepted", False), ("mode_reference_url", "unreviewed"),
    ("server_mode_verification_source", "guessed"), ("mode_basis", "unconfirmed"),
    ("vllm_version_observed", "0.31.0+custom"),
])
def test_receipt_cannot_pass_without_every_measured_and_policy_condition(field, value):
    result = receipt()
    result[field] = value
    with pytest.raises(ValueError):
        contract.validate_teacher_probe(result, POLICY)


def test_operator_can_disable_default_assumptions():
    policy = deepcopy(POLICY)
    policy["allow_upstream_default_assumption"] = False
    assert contract.mode_basis(REMOTE, "0.31.0", None, policy)["accepted"] is False
    with pytest.raises(ValueError):
        contract.validate_teacher_probe(receipt(), policy)


def test_failed_probe_invalidates_old_success_before_runtime_import(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    probe_spec = importlib.util.spec_from_file_location("teacher_probe_entry", TOOLS / "probe_teacher.py")
    probe = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe)
    config = tmp_path / "effective.yaml"
    config.write_text("teachers: []\n")
    output = tmp_path / "teacher-probe.json"
    output.write_text('{"success": true}')
    monkeypatch.setattr(sys, "argv", ["probe_teacher.py", "--config", str(config)])
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ModuleNotFoundError):
        probe.main()
    assert json.loads(output.read_text())["success"] is False
