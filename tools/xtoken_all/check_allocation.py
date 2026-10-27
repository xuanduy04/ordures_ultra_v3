"""Capture live UUID/occupancy evidence after the operator verifies Ray ownership."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess


def owned_gcs_processes(port):
    """Read only local process identity; never infer ownership from an open port."""
    owned = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.geteuid():
                continue
            args = (directory / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if Path(args[0]).name != "gcs_server":
                continue
            # ray start's --port becomes --gcs_server_port on the GCS child.
            ports = []
            for index, value in enumerate(args):
                for flag in ("--gcs_server_port", "--port"):
                    if value.startswith(flag + "="):
                        ports.append(value.split("=", 1)[1])
                    elif value == flag:
                        ports.append(args[index + 1] if index + 1 < len(args) else "")
            if ports and set(ports) == {str(port)}:
                owned.append(int(directory.name))
        except (OSError, ProcessLookupError):
            continue
    return owned


def ensure_local_head(address, gpu_count, start_if_missing, executable, timeout_s):
    host, port_text = address.rsplit(":", 1)
    port = int(port_text)
    local = subprocess.check_output(["hostname", "-I"], text=True).split()
    if host not in local:
        raise ValueError("Automatic setup only accepts this pod's concrete local IP")
    pids = owned_gcs_processes(port)
    if not pids:
        with socket.socket() as connection:
            connection.settimeout(1)
            occupied = connection.connect_ex((host, port)) == 0
        if occupied or not start_if_missing:
            raise ValueError("No owned local GCS head; port occupied or automatic start disabled")
        if executable is None or timeout_s is None:
            raise ValueError("Ray executable and timeout are required for setup")
        subprocess.run([str(executable), "start", "--head", "--node-ip-address=" + host,
            "--port=" + str(port), "--num-gpus=" + str(gpu_count), "--include-dashboard=False"],
            check=True, timeout=timeout_s)
        pids = owned_gcs_processes(port)
    if len(pids) != 1:
        raise ValueError("Expected exactly one owned local GCS process at the selected port")
    return host, pids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-label", required=True)
    parser.add_argument("--expected-gpu-uuids", nargs="+", required=True)
    parser.add_argument("--gpu-indices", nargs="+", type=int)
    parser.add_argument("--ray-address", required=True)
    ownership = parser.add_mutually_exclusive_group(required=True)
    ownership.add_argument("--operator-owned-ray-verified", action="store_true")
    ownership.add_argument("--auto-local", action="store_true")
    parser.add_argument("--start-if-missing", action="store_true")
    parser.add_argument("--ray-executable", type=Path)
    parser.add_argument("--timeout-s", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new allocation receipt")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    mapping = {int(line.split(",")[0]): line.split(",")[1].strip() for line in raw.splitlines()}
    indices = args.gpu_indices if args.gpu_indices is not None else list(range(len(args.expected_gpu_uuids)))
    if not indices or len(set(indices)) != len(indices) or len(indices) != len(args.expected_gpu_uuids):
        raise ValueError("Explicit GPU indices and expected UUIDs must have matching unique entries")
    uuids = [mapping[index] for index in indices]
    assert uuids == args.expected_gpu_uuids, "GPU UUID/allocation mismatch"
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    assert not any(line.split(",")[0].strip() in uuids for line in processes.splitlines()), "Target GPUs have compute processes"
    host, gcs_pids = None, []
    if args.auto_local:
        host, gcs_pids = ensure_local_head(args.ray_address, len(indices), args.start_if_missing,
                                         args.ray_executable, args.timeout_s)
    import ray
    ray.init(address=args.ray_address, log_to_driver=False)
    try:
        nodes = [node for node in ray.nodes() if node["Alive"]]
        assert len(nodes) == 1 and ray.cluster_resources()["GPU"] == len(indices), "Ray topology differs from selected local GPUs"
        if args.auto_local and nodes[0]["NodeManagerAddress"] != host:
            raise ValueError("Ray head belongs to another node")
        payload = {"success": True, "node_label": args.node_label, "gpu_indices": indices,
                   "gpu_uuids": uuids, "ray_address": args.ray_address,
                   "ray_node_ids": [node["NodeID"] for node in nodes],
                   "ray_ownership_verified": True,
                   "ray_ownership_verification_source": "launcher_checked_local_pid_and_node_ip" if args.auto_local else "operator_checked_pid_and_cmdline",
                   "local_gcs_pids": gcs_pids,
                   "checked_at_utc": datetime.now(timezone.utc).isoformat()}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
