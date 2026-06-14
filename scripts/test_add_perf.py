#!/usr/bin/env python3
"""
Automated performance test script for FP32 ADD operation.
Tests different thread counts and compares performance.

python3 -u ./scripts/test_add_perf.py 2>&1 | tee test_add_perf_report.txt

"""

import subprocess
import re
import time
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "ggml-hexagon.cfg"
LOG_TAG = "ggmlop_dsp_add"

def run_command(cmd, cwd=None):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def get_current_thread_count():
    """Read current thread count from config file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            content = f.read()

        # Find thread_counts in [cdsp] section
        match = re.search(r'\[cdsp\].*?thread_counts\s*=\s*(\d+)', content, re.DOTALL)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"Error reading config: {e}")

    return None

def set_thread_count(count):
    """Set thread count in config file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            content = f.read()

        # Replace thread_counts in [cdsp] section
        pattern = r'(thread_counts\s*=\s*)(\d+)'
        content = re.sub(pattern, rf'\g<1>{count}', content)

        with open(CONFIG_FILE, 'w') as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"Error writing config: {e}")
        return False

def adb_push_config():
    """Push config to device."""
    run_command("adb push scripts/ggml-hexagon.cfg /data/local/tmp/")

def adb_clear_logcat():
    """Clear adb logcat buffer."""
    run_command("adb logcat -c")
    time.sleep(0.5)

def run_add_test():
    """Run ADD test and return DSP logs."""
    # Run the test
    run_command("./scripts/build-run-android.sh run_testop ADD")

    # Wait for logs to flush
    time.sleep(1.0)

    # Capture DSP logs
    stdout, stderr, ret = run_command(f"adb logcat -d | grep '{LOG_TAG}'")

    return stdout

def parse_dsp_logs(logs):
    """Parse DSP logs to extract execution times."""
    results = {}

    # Pattern: [ggmlop_dsp_add, 244]: elapse time of ADDf32_4096x4096f32_4096x4096 is 7240 us
    pattern = r'\[ggmlop_dsp_add, \d+\]: elapse time of (ADD\w+)\s+is\s+(\d+)\s+us'

    for match in re.finditer(pattern, logs):
        op_name = match.group(1)
        duration = int(match.group(2))

        if op_name not in results:
            results[op_name] = []

        results[op_name].append(duration)

    return results

def calculate_stats(times):
    """Calculate statistics for a list of times."""
    if not times:
        return None

    times_sorted = sorted(times)
    count = len(times)
    avg = sum(times) / count
    min_val = min(times)
    max_val = max(times)
    median = times_sorted[count // 2] if count % 2 == 1 else (times_sorted[count // 2 - 1] + times_sorted[count // 2]) / 2

    return {
        'count': count,
        'avg': avg,
        'min': min_val,
        'max': max_val,
        'median': median
    }

def print_results(results, thread_count):
    """Print formatted results."""
    print(f"\n>>> Thread Count: {thread_count}")
    print("-" * 100)
    print(f"{'Operator':<60} {'Count':>6} {'Min(us)':>10} {'Avg(us)':>10} {'Max(us)':>10} {'Median(us)':>10}")
    print("-" * 100)

    for op_name in sorted(results.keys()):
        stats = calculate_stats(results[op_name])
        if stats:
            print(f"{op_name:<60} {stats['count']:>6} {stats['min']:>10} {stats['avg']:>10.0f} {stats['max']:>10} {stats['median']:>10.0f}")

def main():
    """Main function."""
    print("=" * 100)
    print("FP32 ADD Performance Automated Test (No Build)")
    print("=" * 100)

    # Get current thread count
    current_thread_count = get_current_thread_count()
    if current_thread_count:
        print(f"Current thread_count: {current_thread_count}")
    else:
        print("Could not read thread_count from config")
        return

    # Test with different thread counts
    thread_counts_to_test = [1, 2, 3, 4, 5, 6, 7, 8]

    all_results = {}

    for thread_count in thread_counts_to_test:
        print(f"\n>>> Testing thread_count = {thread_count}")

        # Set thread count
        if not set_thread_count(thread_count):
            continue

        # Push config to device
        print(f"    Pushing config...")
        adb_push_config()

        # Clear logcat
        print(f"    Clearing logs...")
        adb_clear_logcat()

        # Run test
        print(f"    Running ADD test...")
        logs = run_add_test()

        # Parse results
        results = parse_dsp_logs(logs)

        if results:
            all_results[thread_count] = results
            print_results(results, thread_count)
        else:
            print(f"    No results captured!")

    # Summary comparison
    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 100)

    # Focus on 4096x4096 matrix
    if all_results:
        target_ops = []
        for op_name in all_results[max(all_results.keys())].keys():
            if '4096x4096' in op_name:
                target_ops.append(op_name)

        if target_ops:
            print(f"\n{'Thread':<10} {'Time(us)':<15} {'Speedup vs 1T':<15} {'Speedup vs 2T':<15}")
            print("-" * 55)

            baseline_times = {}
            for thread_count in sorted(all_results.keys()):
                for op_name in target_ops:
                    stats = calculate_stats(all_results[thread_count].get(op_name, []))
                    if stats:
                        time_us = stats['avg']
                        if thread_count == 1:
                            baseline = time_us
                            baseline_times[thread_count] = baseline
                            speedup_vs_1t = 1.0
                            speedup_vs_2t = "-"
                        else:
                            baseline = baseline_times.get(1, time_us)
                            speedup_vs_1t = baseline / time_us if baseline > 0 else 0
                            speedup_vs_2t_val = baseline_times.get(2, time_us)
                            speedup_vs_2t = f"{speedup_vs_2t_val / time_us:.2f}x" if speedup_vs_2t_val > 0 else "-"

                        print(f"{thread_count:<10} {time_us:<15.0f} {speedup_vs_1t:<15.2f}x {speedup_vs_2t:<15}")
                        break  # Only show one row per thread count

    print("\n" + "=" * 100)
    print("Test completed!")
    print("=" * 100)

if __name__ == "__main__":
    main()
