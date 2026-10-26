"""CPU-only stream handshake on the already approved Ray cluster."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from integrity import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allocation-receipt", type=Path, required=True)
    args = parser.parse_args()
    path = args.run_dir / "ray-stream-check.json"
    config_path = args.run_dir / "effective.yaml"
    config_hash = sha256(config_path)
    path.write_text(json.dumps({"success": False, "effective_config_sha256": config_hash}) + "\n")
    from omegaconf import OmegaConf
    import ray

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from nemo_rl.algorithms.x_token.gym_compat import (
        reject_inherited_gym_config, validate_ray_address, gym_actor_runtime_env, validate_gym_actor_report,
    )
    from nemo_rl.algorithms.x_token.gym_protocol import restore_order
    import os

    reject_inherited_gym_config(os.environ)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    knobs = cfg["on_policy"]["nemo_gym"]
    runtime = gym_actor_runtime_env(knobs, root, os.environ)
    lease = json.loads(args.allocation_receipt.read_text())
    assert lease["success"] and lease["ray_ownership_verified"]
    assert 0 <= (datetime.now(timezone.utc) - datetime.fromisoformat(lease["checked_at_utc"])).total_seconds() < 300
    address = validate_ray_address(knobs["ray_address"])
    assert lease["ray_address"] == address
    ray.init(address=address, log_to_driver=False)
    actor, stream, completed = None, None, False
    try:
        node_ids = sorted(node["NodeID"] for node in ray.nodes() if node["Alive"])
        assert node_ids == sorted(lease["ray_node_ids"]) and ray.cluster_resources()["GPU"] == 4

        @ray.remote(num_cpus=0, max_restarts=0, max_task_retries=0)  # pragma: no cover
        class Stream:
            def runtime_identity(self, gym_root):
                from tools.xtoken_all.runtime_probe import probe

                return probe("gym_actor", Path(gym_root))

            async def values(self):
                yield 1, "second"
                await asyncio.sleep(0.01)
                yield 0, "first"

        actor = Stream.options(runtime_env=runtime).remote()
        actor_report = ray.get(actor.runtime_identity.remote(knobs["source_root"]),
                               timeout=knobs["result_timeout_s"])
        validate_gym_actor_report(actor_report, runtime)
        stream = actor.values.remote()
        returned = []
        deadline = time.monotonic() + knobs["ray_probe_timeout_s"]
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not ray.wait([stream], timeout=max(remaining, 0))[0]:
                raise TimeoutError("Ray stream handshake timed out")
            try:
                reference = next(stream)
            except StopIteration:
                break
            returned.append(ray.get(reference, timeout=max(deadline - time.monotonic(), 0)))
        assert restore_order(returned, 2) == ["first", "second"]
        completed = True
        path.write_text(json.dumps({"success": True, "effective_config_sha256": config_hash,
            "checked_at_utc": datetime.now(timezone.utc).isoformat(), "ray_node_ids": node_ids,
            "received_indices": [index for index, _ in returned], "gpu_tasks_submitted": 0,
            "actor_runtime": actor_report,
            "scope": "NemoGym real imports in matching Ray runtime; async generator -> ray.wait/get -> restore_order; no model training"}, indent=2) + "\n")
    finally:
        if stream is not None and not completed:
            ray.cancel(stream)
        if actor is not None:
            ray.kill(actor)
        ray.shutdown()


if __name__ == "__main__":
    main()
