"""Fail-closed boundaries shared by Gym setup and its offline preflight."""

import hashlib
import json
import os
from pathlib import Path


def gym_actor_runtime_env(knobs, source_root, environ):
    """Mirror Super create_env's actor env merge, using only an existing venv.

    Super resolves NemoGym through the actor registry and the per-actor venv
    store. X-Token pins that provisioned interpreter explicitly to avoid the
    provisioning branch of create_local_venv_on_each_node in production.
    Gym service interpreters are selected separately by the SDK's venv_dir.
    """
    reject_inherited_gym_config(environ)
    if "worker_python" in knobs:
        raise ValueError("Re-render r7 with actor_python; worker_python conflated actor and service runtimes")
    executable = Path(os.path.abspath(knobs["actor_python"]))
    services = Path(knobs["venv_dir"]).resolve()
    if not executable.is_file() or not services.is_dir():
        raise FileNotFoundError("Existing NemoGym actor interpreter and service venv store are required")
    if executable.parent.resolve().is_relative_to(services):
        raise ValueError("NemoGym actor_python must not point into the thin Gym service venv store")
    inherited_path = environ["PYTHONPATH"] if "PYTHONPATH" in environ else ""
    env = dict(environ)
    env.update(UV_OFFLINE="1", HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join(part for part in
                   (str(Path(source_root).resolve()), str(Path(knobs["source_root"]).resolve()), inherited_path) if part),
               VIRTUAL_ENV=str(executable.parent.parent),
               UV_PROJECT_ENVIRONMENT=str(executable.parent.parent),
               NEMO_GYM_VENV_DIR=str(services))
    return {"py_executable": str(executable), "env_vars": env}


def validate_gym_actor_report(report, runtime):
    """Bind successful real imports to the interpreter used to launch NemoGym."""
    executable = Path(runtime["py_executable"])
    if report["role"] != "gym_actor" or os.path.abspath(report["python_executable"]) != str(executable):
        raise ValueError("Gym actor probe used another interpreter")
    if Path(report["python_prefix"]).resolve() != executable.parent.parent.resolve():
        raise ValueError("Gym actor probe lost its virtualenv prefix")
    paths = runtime["env_vars"]["PYTHONPATH"].split(os.pathsep)
    if Path(report["module"]).resolve() != Path(paths[0]) / "nemo_rl/environments/nemo_gym.py":
        raise ValueError("Gym actor probe imported another NeMo source tree")
    if Path(report["gym_source_root"]).resolve() != Path(paths[1]):
        raise ValueError("Gym actor probe imported another Gym source tree")
    if report["actor_import_completed"] is not True or report["cuda_operations_requested"] is not False:
        raise ValueError("Missing CPU-only import check of the real NemoGym actor")
    if set(report["checked_actor_apis"]) != {"_spinup", "run_rollouts", "shutdown"}:
        raise ValueError("Incomplete NemoGym actor API check")


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
