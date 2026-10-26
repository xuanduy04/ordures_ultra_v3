"""Capture live UUID/occupancy evidence after the operator verifies Ray ownership."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-label", required=True)
    parser.add_argument("--expected-gpu-uuids", nargs=4, required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--operator-owned-ray-verified", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new allocation receipt")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    mapping = {int(line.split(",")[0]): line.split(",")[1].strip() for line in raw.splitlines()}
    uuids = [mapping[index] for index in range(4)]
    assert uuids == args.expected_gpu_uuids, "GPU UUID/allocation mismatch"
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    assert not any(line.split(",")[0].strip() in uuids for line in processes.splitlines()), "Target GPUs have compute processes"
    import ray
    ray.init(address=args.ray_address, log_to_driver=False)
    try:
        nodes = [node for node in ray.nodes() if node["Alive"]]
        assert len(nodes) == 1 and ray.cluster_resources()["GPU"] == 4, "Expected one four-GPU Ray node"
        payload = {"success": True, "node_label": args.node_label, "gpu_indices": [0, 1, 2, 3],
                   "gpu_uuids": uuids, "ray_address": args.ray_address,
                   "ray_node_ids": [node["NodeID"] for node in nodes],
                   "ray_ownership_verified": args.operator_owned_ray_verified,
                   "ray_ownership_verification_source": "operator_checked_pid_and_cmdline",
                   "checked_at_utc": datetime.now(timezone.utc).isoformat()}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
