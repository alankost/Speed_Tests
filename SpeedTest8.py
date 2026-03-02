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
array_size = 10_000_000  # Use larger array for clearer results

# Configure your CPU here (adjust for your specific CPU)
P_CORES = 64   # Number of P-cores
E_CORES = 0   # Number of E-cores
# P-cores are typically indices 0 to P_CORES-1 on Intel hybrid CPUs
P_CORE_INDICES = list(range(P_CORES))
ALL_CORE_INDICES = list(range(P_CORES + E_CORES))

@jit(nopython=True, parallel=True)
def compute_arrays_parallel(arr1, arr2):
    result = np.empty_like(arr1)
    for i in prange(len(arr1)):
        result[i] = arr1[i] * arr2[i] + arr1[i] - arr2[i]
    return result

def set_cpu_affinity(core_indices):
    """Set which CPU cores this process can use"""
    if not PSUTIL_AVAILABLE:
        return False
    try:
        p = psutil.Process()
        p.cpu_affinity(core_indices)
        return True
    except Exception as e:
        print(f"Warning: Could not set CPU affinity: {e}")
        return False

def get_cpu_affinity():
    """Get current CPU affinity"""
    if not PSUTIL_AVAILABLE:
        return None
    try:
        p = psutil.Process()
        return p.cpu_affinity()
    except Exception:
        return None

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

def run_benchmark_suite(arr1, arr2, thread_counts, label):
    """Run benchmarks for a set of thread counts"""
    print(f"\n{'=' * 65}")
    print(f"{label}")
    print(f"CPU Affinity: {get_cpu_affinity()}")
    print(f"{'=' * 65}")
    print(f"{'Threads':<10} {'Total Time (s)':<16} {'Avg (ms)':<12} {'Speedup':<10}")
    print("-" * 65)

    results = {}
    baseline_time = None

    for num_threads in thread_counts:
        elapsed_time = benchmark(arr1, arr2, num_threads, num_iterations)
        avg_time = elapsed_time / num_iterations * 1000

        if baseline_time is None:
            baseline_time = elapsed_time
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed_time

        results[num_threads] = {
            'time': elapsed_time,
            'avg': avg_time,
            'speedup': speedup
        }

        print(f"{num_threads:<10} {elapsed_time:<16.4f} {avg_time:<12.4f} {speedup:<10.2f}x")

    return results

# Main
if not PSUTIL_AVAILABLE:
    print("ERROR: psutil is required for this benchmark")
    print("Install with: pip install psutil")
    exit(1)

print("P-Core vs All-Core Benchmark")
print(f"Array size: {array_size:,}")
print(f"Data type: float32")
print(f"Iterations per test: {num_iterations}")
print(f"\nCPU Configuration (edit P_CORES and E_CORES in script if incorrect):")
print(f"  P-cores: {P_CORES} (indices {P_CORE_INDICES[0]}-{P_CORE_INDICES[-1]})")
print(f"  E-cores: {E_CORES} (indices {P_CORES}-{P_CORES + E_CORES - 1})")
print(f"  Total:   {P_CORES + E_CORES} cores")

# Create arrays
arr1 = np.random.rand(array_size).astype(np.float32)
arr2 = np.random.rand(array_size).astype(np.float32)

# Thread counts for P-cores only (1, 2, 4, 8)
p_thread_counts = [t for t in [1, 2, 4, 8] if t <= P_CORES]
if p_thread_counts[-1] != P_CORES:
    p_thread_counts.append(P_CORES)

# Thread counts for all cores
all_thread_counts = p_thread_counts.copy()
for t in [P_CORES + E_CORES]:
    if t not in all_thread_counts:
        all_thread_counts.append(t)
all_thread_counts.sort()

# Benchmark 1: P-cores only
set_cpu_affinity(P_CORE_INDICES)
numba.set_num_threads(P_CORES)
p_core_results = run_benchmark_suite(arr1, arr2, p_thread_counts, "P-CORES ONLY")

# Benchmark 2: All cores
set_cpu_affinity(ALL_CORE_INDICES)
numba.set_num_threads(P_CORES + E_CORES)
all_core_results = run_benchmark_suite(arr1, arr2, all_thread_counts, "ALL CORES (P + E)")

# Summary comparison
print(f"\n{'=' * 65}")
print("COMPARISON SUMMARY")
print(f"{'=' * 65}")

# Compare max P-core performance vs all-core performance
p_max_threads = max(p_thread_counts)
all_max_threads = max(all_thread_counts)

p_best_avg = p_core_results[p_max_threads]['avg']
all_best_avg = all_core_results[all_max_threads]['avg']

print(f"\nBest P-core only ({p_max_threads} threads):  {p_best_avg:.4f} ms avg")
print(f"Best all cores ({all_max_threads} threads):    {all_best_avg:.4f} ms avg")

if all_best_avg < p_best_avg:
    improvement = (p_best_avg - all_best_avg) / p_best_avg * 100
    print(f"\nUsing all cores is {improvement:.1f}% faster")
else:
    slowdown = (all_best_avg - p_best_avg) / p_best_avg * 100
    print(f"\nUsing all cores is {slowdown:.1f}% SLOWER (E-cores adding overhead)")

# Also compare same thread count (8 threads) on P-cores vs all cores
if P_CORES in all_core_results:
    p_8_avg = p_core_results[P_CORES]['avg']
    all_8_avg = all_core_results[P_CORES]['avg']
    print(f"\nAt {P_CORES} threads:")
    print(f"  P-cores only: {p_8_avg:.4f} ms")
    print(f"  All cores:    {all_8_avg:.4f} ms")
    if all_8_avg < p_8_avg:
        print(f"  (All cores {((p_8_avg - all_8_avg) / p_8_avg * 100):.1f}% faster - OS using P-cores anyway)")
    else:
        print(f"  (P-cores only {((all_8_avg - p_8_avg) / all_8_avg * 100):.1f}% faster - affinity helps)")

print(f"\n{'=' * 65}")
