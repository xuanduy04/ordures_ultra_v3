"""Fail-closed boundaries shared by Gym setup and its offline preflight."""

import hashlib
import json
from pathlib import Path


def reject_inherited_gym_config(environ):
    """The Gym SDK otherwise prefers these values over our explicit config."""
    for name in ("NEMO_GYM_CONFIG_DICT", "NEMO_GYM_CONFIG_PATH"):
        if name in environ and environ[name]:
            raise ValueError("Start this run with a clean Gym process environment: " + name)
    if "NRL_FORCE_REBUILD_VENVS" in environ and environ["NRL_FORCE_REBUILD_VENVS"].lower() in {"true", "1", "yes"}:
        raise ValueError("This run must reuse provisioned workers: NRL_FORCE_REBUILD_VENVS")


def validate_ray_address(address):
    """Only a concrete existing head is accepted, without local auto-start."""
    if not isinstance(address, str) or ":" not in address or "://" in address:
        raise ValueError("Gym requires an explicit existing Ray head host:port")
    host, port = address.rsplit(":", 1)
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError("Invalid Ray head address")
    return address


def init_existing_gym_ray(config, ray_module, environ):
    """Attach to the approved head; never fall back to creating another one."""
    reject_inherited_gym_config(environ)
    address = validate_ray_address(config.on_policy["nemo_gym"]["ray_address"])
    if ray_module.is_initialized():
        raise RuntimeError("Use a fresh driver process for the Gym smoke")
    ray_module.init(address=address, log_to_driver=True, runtime_env={"env_vars": dict(environ)})
    try:
        nodes = [node for node in ray_module.nodes() if node["Alive"]]
        expected_nodes = config.cluster["num_nodes"]
        expected_gpus = expected_nodes * config.cluster["gpus_per_node"]
        if len(nodes) != expected_nodes or ray_module.cluster_resources()["GPU"] != expected_gpus:
            raise RuntimeError("Existing Ray topology differs from the effective config")
    except BaseException:
        ray_module.shutdown()
        raise


def verify_gym_source(source_root, contract_path):
    """Check the exact reviewed SDK closure, independent of its install path."""
    root = Path(source_root).resolve()
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    for relative, expected in contract["gym_files"].items():
        path = (root / relative).resolve()
        path.relative_to(root)
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError("Gym source differs from the reviewed archive: " + relative)
    return {"files_checked": len(contract["gym_files"]),
            "archive_sha256": contract["archives"]["custom_opt_ultra_selfbuilt.zip"]["sha256"]}


def validate_resolved_gym_models(global_config, template_kwargs):
    """Reject request transformations that change the behavior being distilled."""
    for name in ("policy_model", "policy_model_reasoning_off"):
        model = global_config[name]["responses_api_models"]["vllm_model"]
        if model["return_token_id_information"] is not True or model["uses_reasoning_parser"] is not False:
            raise ValueError("Gym must expose sampled IDs without a reasoning parser: " + name)
        if model["chat_template_kwargs"] != template_kwargs:
            raise ValueError("Student and Gym chat template kwargs differ")
        if "extra_body" in model and model["extra_body"]:
            raise ValueError("Gym extra_body can change the sampling distribution")
    qa = global_config["general_qa"]["resources_servers"]["general_qa"]
    if qa["should_use_judge"] is not False:
        raise ValueError("G1 records deterministic QA reward without LLM judge inference")
