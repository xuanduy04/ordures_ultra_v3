import sys
import os
import re
from collections import defaultdict
import statistics
import argparse
from tqdm.auto import tqdm

def parse_training_logs(file_path: str, align: bool = False):
    # Regex to capture metric_name and seconds
    metric_regex = re.compile(r'^\s*•?\s*([\w_/]+):\s*([\d.]+)\s*s')
    
    all_steps_metrics = []
    current_metrics = {}
    in_timing_section = False

    # Get file size for a precise tqdm byte-based progress bar if lines aren't known
    file_size = os.path.getsize(file_path)

    print("Parsing log file...")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Wrap the file iterator in tqdm to track reading progress by bytes
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing") as pbar:
            for line in f:
                pbar.update(len(line.encode('utf-8', 'ignore'))) # Update progress bar
                
                # Clean weird ascii artifacting safely
                clean_line = line.encode('utf-8', 'ignore').decode('utf-8').strip()
                
                # Detect start of a new log block
                if "Logged data to" in clean_line:
                    if current_metrics:
                        all_steps_metrics.append(current_metrics)
                        current_metrics = {}
                    in_timing_section = False
                    continue
                
                # Detect Timing subcategory
                if "Timing:" in clean_line:
                    in_timing_section = True
                    continue
                
                # Parse metrics inside the timing block
                if in_timing_section:
                    if not clean_line or (":" in clean_line and "•" not in clean_line and not metric_regex.match(clean_line)):
                        if not any(char.isdigit() for char in clean_line):
                            in_timing_section = False
                            continue
                    
                    match = metric_regex.match(clean_line)
                    if match:
                        key = match.group(1)
                        seconds = float(match.group(2))
                        current_metrics[key] = seconds

            # Append the last block if it exists after loop finishes
            if current_metrics:
                all_steps_metrics.append(current_metrics)

    if not all_steps_metrics:
        print("\n❌ No valid log blocks found.")
        return
    
    # --- Aggregate data across all steps ---
    grouped_metrics = defaultdict(list)
    total_step_times = []  # Tracks the sum of each individual step to compute its overall std dev
    
    for step in tqdm(all_steps_metrics, desc="Aggregating steps"):
        step_total = 0.0
        for key, value in step.items():
            if key.lower() == 'total_step_time':
                continue
            grouped_metrics[key].append(value)
            step_total += value
        
        # Track the per-step total dynamically
        total_step_times.append(step_total)

    # Calculate metrics for individual components
    averages = {}
    std_devs = {}
    for key, values in grouped_metrics.items():
        averages[key] = statistics.mean(values)
        std_devs[key] = statistics.stdev(values) if len(values) > 1 else 0.0
    
    # Calculate metrics for the overall Total Step Time
    avg_total_step_time = statistics.mean(total_step_times) if total_step_times else 0.0
    std_total_step_time = statistics.stdev(total_step_times) if len(total_step_times) > 1 else 0.0
    
    # --- Print Output Table ---
    max_key_len = max((len(str(k)) for k in averages.keys()), default=0) if align else 0
    print(f"\n⏳ Timing Summary (Averaged across {len(all_steps_metrics)} steps):")
    if align:
        print(f"  • {'Total step time:':<{max_key_len}} {avg_total_step_time:>8.2f}s ± {std_total_step_time:>6.2f}s")
    else:
        print(f"  • Total step time: {avg_total_step_time:.2f}s ± {std_total_step_time:.2f}s")
    print("   " + "-" * (67 + (2 if align else -1)))
    
    # Print components by highest duration first
    for key, avg_sec in sorted(averages.items(), key=lambda x: x[1], reverse=True):
        percentage = (avg_sec / avg_total_step_time) * 100 if avg_total_step_time > 0 else 0
        std_val = std_devs[key]
        
        if align:
            print(f"  • {f'{key}:':<{max_key_len + 1}} {avg_sec:>7.2f}s ± {std_val:>7.2f}s ({percentage:>4.1f}%)")
        else:
            print(f"  • {key}: {avg_sec:.2f}s ± {std_val:.2f}s ({percentage:.1f}%)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get training time breakdown statistic from a training log file."
    )
    
    parser.add_argument(
        "log_file",
        help="Path to the training log file"
    )
    
    parser.add_argument(
        "-a", "--align",
        action="store_true",
        help="Enable alignment processing for the logs."
    )

    args = parser.parse_args()

    parse_training_logs(args.log_file, align=args.align)
