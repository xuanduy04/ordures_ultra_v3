import os
import resource
import subprocess
import sys

PAGE_SIZE_BYTES = os.sysconf("SC_PAGE_SIZE")


def print_gpu_usage():
    # 1. Query GPU statistics and UUIDs
    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # 2. Query active compute processes/applications
    gpu_processes = {}
    try:
        app_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Group processes by GPU UUID
        for line in app_result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                g_uuid, pid, proc_name, mem = parts[0], parts[1], parts[2], parts[3]
                gpu_processes.setdefault(g_uuid, []).append(
                    {"pid": pid, "name": proc_name, "memory": mem}
                )
    except subprocess.CalledProcessError:
        # Fails gracefully if there are permission issues or no active compute apps
        pass

    out_lines = ["GPU statistics:"]

    for line in gpu_result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
            
        gpu_id, uuid, mem_used, mem_total, util = parts
        out_lines.append(f"  GPU {gpu_id}: {mem_used:>6} / {mem_total:>6} MiB | util {util:>2}%")

        # Print running processes associated with this GPU UUID
        procs = gpu_processes.get(uuid, [])
        if procs:
            out_lines.append(f"  GPU {gpu_id} active processes:")
            for p in procs:
                out_lines.append(f"      - Memory {p['memory']:>6} MiB: {p['name']} [PID {p['pid']}]")
        else:
            out_lines.append("    Active processes: None")

    try:
        with open("/proc/self/statm") as statm_file:
            resident_pages = int(statm_file.read().split()[1])
        current_gib = resident_pages * PAGE_SIZE_BYTES / (1024**3)
        peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
        out_lines.append(
            f"  [rss]: current={current_gib:.2f} GiB, peak={peak_gib:.2f} GiB"
        )
    except Exception as e:
        out_lines.append(f"  [rss]: Could not log rss\n  Error: {e}")

    out_lines.append("")
    print("\n".join(out_lines), flush=True)


if __name__ == "__main__":
    try:
        print_gpu_usage()
    except Exception as e:
        print(f"GPU monitoring error: {e}", file=sys.stderr)
        sys.exit(1)