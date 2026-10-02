import subprocess
import sys


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
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                g_uuid, pid, proc_name, mem = parts[0], parts[1], parts[2], parts[3]
                if g_uuid not in gpu_processes:
                    gpu_processes[g_uuid] = []
                gpu_processes[g_uuid].append(
                    {"pid": pid, "name": proc_name, "memory": mem}
                )
    except subprocess.CalledProcessError:
        # Fails gracefully if there are permission issues or no active compute apps
        pass

    print("GPU statistics:", flush=True)

    for line in gpu_result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
            
        gpu_id, uuid, mem_used, mem_total, util = parts
        print(f"  GPU {gpu_id}: {mem_used:>6} / {mem_total:>6} MiB | util {util:>2}%", flush=True)

        # Print running processes associated with this GPU UUID
        procs = gpu_processes.get(uuid, [])
        if procs:
            print(f"  GPU {gpu_id} active processes:", flush=True)
            for p in procs:
                print(f"      - Memory {p['memory']:>6} MiB: {p['name']} [PID {p['pid']}]", flush=True)
        else:
            print("    Active processes: None", flush=True)
            
    
    def _log_rss() -> None:
        """Print current and peak resident set size for memory diagnostics."""
        try:
            import resource
            import os
    
            with open("/proc/self/statm") as statm_file:
                resident_pages = int(statm_file.read().split()[1])
            page_size_bytes = os.sysconf("SC_PAGE_SIZE")
            current_gib = resident_pages * page_size_bytes / (1024**3)
            peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)
            print(
                f"  [rss]: current={current_gib:.2f} GiB, peak={peak_gib:.2f} GiB",
                flush=True,
            )
        except Exception as e:
            print(f"  [rss]: Could not log rss\n  Error: {e}")

    _log_rss()
    print("", flush=True)


if __name__ == "__main__":
    try:
        print_gpu_usage()
    except Exception as e:
        print(f"GPU monitoring error: {e}", file=sys.stderr)
        sys.exit(1)
