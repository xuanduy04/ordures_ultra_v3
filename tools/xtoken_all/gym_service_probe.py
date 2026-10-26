"""Import the real HTTP service app in its own venv, without importing NeMo RL."""

import argparse
import importlib
import json
from pathlib import Path
import sys


SERVICES = {
    "model": ("responses_api_models.vllm_model.app", "VLLMModel"),
    "agent": ("responses_api_agents.simple_agent.app", "SimpleAgent"),
    "qa": ("resources_servers.general_qa.app", "GeneralQAResourcesServer"),
}


def probe(service, gym_root):
    root = Path(gym_root).resolve()
    sys.path.insert(0, str(root))
    sdk = importlib.import_module("nemo_gym")
    if Path(sdk.PARENT_DIR).resolve() != root:
        raise ValueError("Service imported another Gym source tree")
    name, class_name = SERVICES[service]
    app = importlib.import_module(name)
    if Path(app.__file__).resolve() != root / (name.replace(".", "/") + ".py"):
        raise ValueError("Service app imported another source tree")
    if not callable(getattr(app, class_name)):
        raise TypeError("Missing Gym service class: " + class_name)
    return {"service": service, "python_executable": sys.executable,
            "python_prefix": sys.prefix, "module": str(Path(app.__file__).resolve()),
            "app_import_completed": True, "servers_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", choices=tuple(SERVICES), required=True)
    parser.add_argument("--gym-root", type=Path, required=True)
    args = parser.parse_args()
    print("XTOKEN_GYM_SERVICE_PROBE=" + json.dumps(probe(args.service, args.gym_root)), flush=True)


if __name__ == "__main__":
    main()
