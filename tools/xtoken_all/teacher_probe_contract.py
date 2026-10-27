"""Separate measured scoring capability from a documented mode assumption."""

import math


def mode_basis(remote, version, evidence, policy):
    """Explicit deployment evidence takes precedence over upstream defaults."""
    if evidence is not None:
        if evidence["teacher_url"].rstrip("/") != remote["url"].rstrip("/") or evidence["served_model"] != remote["model"]:
            raise ValueError("Mode evidence belongs to another teacher")
        if evidence["verification_source"] not in ("operator_checked_server_args", "server_reported_config"):
            raise ValueError("Unsupported mode evidence source")
        if not evidence["observed_at_utc"] or not evidence["evidence_description"]:
            raise ValueError("Incomplete deployment mode evidence")
        accepted = evidence["logprobs_mode"] == "raw_logprobs"
        return {"accepted": accepted, "mode_basis": "deployment_config",
                "effective_logprobs_mode": evidence["logprobs_mode"],
                "server_logprobs_mode_verified": accepted, "mode_assumption_accepted": False,
                "server_mode_verification_source": evidence["verification_source"],
                "mode_reference_url": None}
    known = policy["known_upstream_defaults"]
    if policy["allow_upstream_default_assumption"] and version in known:
        fact = known[version]
        accepted = fact["logprobs_mode"] == "raw_logprobs"
        return {"accepted": accepted, "mode_basis": "upstream_default_assumption",
                "effective_logprobs_mode": fact["logprobs_mode"],
                "server_logprobs_mode_verified": False, "mode_assumption_accepted": accepted,
                "server_mode_verification_source": "versioned_upstream_default",
                "mode_reference_url": fact["reference_url"]}
    return {"accepted": False, "mode_basis": "unconfirmed",
            "effective_logprobs_mode": None, "server_logprobs_mode_verified": False,
            "mode_assumption_accepted": False, "server_mode_verification_source": None,
            "mode_reference_url": None}


def score_contract(config=None):
    """Legacy helpers remain top20; variant transport is selected in the YAML."""
    if config is None or "loss_fn" not in config or "variant" not in config["loss_fn"]:
        return "topk", 20, None
    remote = config["teachers"][0]["remote"]
    mode = config["loss_fn"]["variant"]["score_mode"]
    if mode not in ("topk", "full") or remote["score_mode"] != mode:
        raise ValueError("Variant score mode differs from remote transport")
    return mode, remote["top_k"], remote["model_vocab_size"]


def score_events_match(events, config):
    mode, k, vocab = score_contract(config)
    if mode == "full":
        return bool(events) and all(item["representation"] == "full_logprobs" and item["model_vocab_size"] == vocab for item in events)
    return bool(events) and all(item["representation"] == "topk_logprobs" and item["top_k"] == k for item in events)


def validate_teacher_probe(receipt, policy, config=None):
    """Operational pass needs real scores/parity; a default is never labeled verified."""
    if receipt["success"] is not True or receipt["teacher_score_contract_satisfied"] is not True:
        raise ValueError("Teacher score contract has not passed")
    if receipt["score_request_completed"] is not True or receipt["tokenizer_parity_probes"] != 3:
        raise ValueError("Missing real scoring request or tokenizer parity")
    mode, k, vocab = score_contract(config)
    if mode == "full":
        if receipt["representation"] != "full_logprobs" or receipt["top_k"] is not None or receipt["returned_count"] != vocab:
            raise ValueError("Expected the entire configured teacher vocabulary")
    else:
        if receipt["top_k"] != k or not k <= receipt["returned_count"] <= k + 1:
            raise ValueError("Expected configured top-k, possibly plus the observed token")
        if receipt["representation"] != "topk_logprobs":
            raise ValueError("Wrong teacher score representation")
    tolerance = receipt["normalization_atol"]
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid normalization tolerance")
    if not math.isfinite(receipt["returned_probability_mass"]) or not 0 < receipt["returned_probability_mass"] <= 1 + tolerance:
        raise ValueError("Invalid retained probability mass")
    if mode == "full" and abs(receipt["returned_probability_mass"] - 1) > tolerance:
        raise ValueError("Full teacher distribution must have unit probability mass")
    if receipt["effective_logprobs_mode"] != "raw_logprobs":
        raise ValueError("Teacher mode is incompatible with this recipe")
    if receipt["mode_basis"] == "deployment_config":
        if receipt["server_mode_verification_source"] not in ("operator_checked_server_args", "server_reported_config"):
            raise ValueError("Missing deployment evidence source")
        if receipt["server_logprobs_mode_verified"] is not True or receipt["mode_assumption_accepted"] is not False:
            raise ValueError("Deployment evidence must be distinguished from assumptions")
    elif receipt["mode_basis"] == "upstream_default_assumption":
        if receipt["server_mode_verification_source"] != "versioned_upstream_default":
            raise ValueError("Missing upstream default source")
        version = receipt["vllm_version_observed"]
        if not policy["allow_upstream_default_assumption"] or version not in policy["known_upstream_defaults"]:
            raise ValueError("No accepted upstream default for this exact version")
        fact = policy["known_upstream_defaults"][version]
        if receipt["server_logprobs_mode_verified"] is not False or receipt["mode_assumption_accepted"] is not True:
            raise ValueError("An upstream assumption must not be stamped as verified")
        if receipt["mode_reference_url"] != fact["reference_url"] or fact["logprobs_mode"] != "raw_logprobs":
            raise ValueError("Default reference differs from the reviewed policy")
    else:
        raise ValueError("Teacher mode remains unconfirmed")
