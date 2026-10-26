"""Import/API probe inside an existing worker interpreter; no CUDA operations."""

import argparse
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def module_path(module):
    return str(Path(module.__file__).resolve())


def probe(role, gym_root):
    result = {"role": role, "cuda_operations_requested": False}
    if role == "gym":
        module = importlib.import_module("nemo_gym")
        if Path(module.PARENT_DIR).resolve() != gym_root.resolve():
            raise ValueError("Gym sub-venv imported another source tree")
        result["module"] = module_path(module)
        result["code_version"] = module.__version__
        return result
    module_name = {
        "policy": "nemo_rl.models.policy.workers.dtensor_policy_worker_v2",
        "generation": "nemo_rl.models.generation.vllm.vllm_worker_async",
    }[role]
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve() != ROOT / (module_name.replace(".", "/") + ".py"):
        raise ValueError("Worker imported a different NeMo source tree")
    result["module"] = module_path(module)
    for name in ("torch", "transformers", "ray", "nemo-automodel"):
        result[name] = importlib.metadata.version(name)
    if role == "generation":
        result["vllm"] = importlib.metadata.version("vllm")
        cls = module.VllmAsyncGenerationWorker
        if hasattr(cls, "__ray_metadata__"):
            cls = cls.__ray_metadata__.modified_class
        for name in ("wake_up_async", "sleep_async", "_setup_vllm_openai_api_server"):
            if not callable(getattr(cls, name)):
                raise TypeError("Missing async generation API: " + name)
        return result
    from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

    signature = inspect.signature(FSDP2Manager)
    contract = json.loads((ROOT / "tools/xtoken_all/super-source-contract.json").read_text())
    if "dp_size" in signature.parameters:
        signature.bind(**{name: None for name in contract["source_api"]["fsdp2_dataclass_fields"]})
        result["fsdp_api"] = "legacy_dp_size"
    elif {"config", "device_mesh"}.issubset(signature.parameters):
        signature.bind(config=None, device_mesh=None, moe_mesh=None)
        from nemo_automodel.components.distributed.config import FSDP2Config
        from nemo_automodel.components.distributed.mesh_utils import create_device_mesh

        result["fsdp_api"] = "explicit_config_mesh"
        result["fsdp_config_signature"] = str(inspect.signature(FSDP2Config))
        result["create_mesh_signature"] = str(inspect.signature(create_device_mesh))
    else:
        raise TypeError("Unsupported installed FSDP2 API")
    checkpoint_signature = inspect.signature(Checkpointer.load_base_model)
    kwargs = {"device": None, "root_dir": None, "model_name": None, "load_base_model": True}
    if "peft_init_method" in checkpoint_signature.parameters:
        kwargs["peft_init_method"] = None
    checkpoint_signature.bind(None, None, **kwargs)
    result.update(fsdp_module=inspect.getfile(FSDP2Manager), fsdp_signature=str(signature),
                  checkpoint_module=inspect.getfile(Checkpointer), checkpoint_signature=str(checkpoint_signature))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("policy", "generation", "gym"), required=True)
    parser.add_argument("--gym-root", type=Path)
    args = parser.parse_args()
    if args.role == "gym" and args.gym_root is None:
        parser.error("Gym probe requires --gym-root")
    print("XTOKEN_RUNTIME_PROBE=" + json.dumps(probe(args.role, args.gym_root)), flush=True)


if __name__ == "__main__":
    main()
