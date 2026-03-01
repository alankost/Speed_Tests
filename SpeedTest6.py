import numpy as np
import time
import numba
from numba import jit, prange

# Configuration
array_size = 1_000_000
num_iterations = 100

@jit(nopython=True, parallel=True)
def compute_arrays_parallel(arr1, arr2):
    result = np.empty_like(arr1)
    for i in prange(len(arr1)):
        result[i] = arr1[i] * arr2[i] + arr1[i] - arr2[i]
    return result

def benchmark(arr1, arr2, num_threads):
    """Run benchmark with specified number of threads"""
    numba.set_num_threads(num_threads)

    # Warm-up run
    _ = compute_arrays_parallel(arr1, arr2)

    # Benchmark
    start_time = time.perf_counter()
    for i in range(num_iterations):
        result = compute_arrays_parallel(arr1, arr2)
    end_time = time.perf_counter()

    return end_time - start_time

# Create numpy arrays with float32
arr1 = np.random.rand(array_size).astype(np.float32)
arr2 = np.random.rand(array_size).astype(np.float32)

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

print(f"Array size: {array_size:,}")
print(f"Data type: float32")
print(f"Iterations: {num_iterations}")
print(f"Max available threads: {max_threads}")
print()
print("=" * 55)
print(f"{'Threads':<10} {'Total Time (s)':<16} {'Avg (ms)':<12} {'Speedup':<10}")
print("=" * 55)

results = []
baseline_time = None

for num_threads in thread_counts:
    elapsed_time = benchmark(arr1, arr2, num_threads)
    avg_time = elapsed_time / num_iterations * 1000

    if baseline_time is None:
        baseline_time = elapsed_time
        speedup = 1.0
    else:
        speedup = baseline_time / elapsed_time

    results.append((num_threads, elapsed_time, avg_time, speedup))
    print(f"{num_threads:<10} {elapsed_time:<16.4f} {avg_time:<12.4f} {speedup:<10.2f}x")

print("=" * 55)
