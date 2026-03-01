import numpy as np
import time
import os
import numba
from numba import jit, prange

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Configuration
num_iterations = 50
array_sizes = [100_000, 1_000_000, 10_000_000, 50_000_000]

def get_cpu_info():
    """Get physical and logical core counts"""
    logical_cores = os.cpu_count()

    if PSUTIL_AVAILABLE:
        physical_cores = psutil.cpu_count(logical=False)
    else:
        physical_cores = None

    return physical_cores, logical_cores

@jit(nopython=True, parallel=True)
def compute_arrays_parallel(arr1, arr2):
    result = np.empty_like(arr1)
    for i in prange(len(arr1)):
        result[i] = arr1[i] * arr2[i] + arr1[i] - arr2[i]
    return result

def benchmark(arr1, arr2, num_threads, iterations):
    """Run benchmark with specified number of threads"""
    numba.set_num_threads(num_threads)

    # Warm-up run
    _ = compute_arrays_parallel(arr1, arr2)

    # Benchmark
    start_time = time.perf_counter()
    for i in range(iterations):
        result = compute_arrays_parallel(arr1, arr2)
    end_time = time.perf_counter()

    return end_time - start_time

# Get max available threads
max_threads = numba.get_num_threads()

# Generate thread counts to test (1, 2, 4, 8, ... up to max)
thread_counts = []
t = 1
while t <= max_threads:
    thread_counts.append(t)
    t *= 2
if thread_counts[-1] != max_threads:
    thread_counts.append(max_threads)

# Get CPU info
physical_cores, logical_cores = get_cpu_info()

print(f"Benchmarking array sizes vs thread counts")
print(f"Data type: float32")
print(f"Iterations per test: {num_iterations}")
print()
print("CPU Information:")
if physical_cores is not None:
    print(f"  Physical cores: {physical_cores}")
    print(f"  Logical cores:  {logical_cores} (includes hyperthreads)")
    if logical_cores > physical_cores:
        print(f"  Hyperthreading: Enabled ({logical_cores // physical_cores}x)")
    else:
        print(f"  Hyperthreading: Disabled")
else:
    print(f"  Logical cores: {logical_cores}")
    print(f"  (Install psutil for physical core count: pip install psutil)")
print(f"  Numba threads: {max_threads}")
print()

# Store all results for summary table
all_results = {}

for array_size in array_sizes:
    print("=" * 65)
    print(f"Array size: {array_size:,}")
    print("=" * 65)
    print(f"{'Threads':<10} {'Total Time (s)':<16} {'Avg (ms)':<12} {'Speedup':<10}")
    print("-" * 65)

    # Create arrays for this size
    arr1 = np.random.rand(array_size).astype(np.float32)
    arr2 = np.random.rand(array_size).astype(np.float32)

    baseline_time = None
    all_results[array_size] = {}

    for num_threads in thread_counts:
        elapsed_time = benchmark(arr1, arr2, num_threads, num_iterations)
        avg_time = elapsed_time / num_iterations * 1000

        if baseline_time is None:
            baseline_time = elapsed_time
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed_time

        all_results[array_size][num_threads] = {
            'time': elapsed_time,
            'avg': avg_time,
            'speedup': speedup
        }

        print(f"{num_threads:<10} {elapsed_time:<16.4f} {avg_time:<12.4f} {speedup:<10.2f}x")

    # Free memory before next array size
    del arr1, arr2
    print()

# Print summary comparison table
print("=" * 65)
print("SPEEDUP SUMMARY (comparing single-thread to max threads)")
print("=" * 65)
print(f"{'Array Size':<20} {'1 Thread (ms)':<16} {f'{max_threads} Threads (ms)':<18} {'Speedup':<10}")
print("-" * 65)

for array_size in array_sizes:
    time_1 = all_results[array_size][1]['avg']
    time_max = all_results[array_size][max_threads]['avg']
    speedup = time_1 / time_max
    print(f"{array_size:<20,} {time_1:<16.4f} {time_max:<18.4f} {speedup:<10.2f}x")

print("=" * 65)
