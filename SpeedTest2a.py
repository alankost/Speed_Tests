import numpy as np
import time
from numba import jit

# Configuration
array_size = 1_000_000
num_iterations = 100

# This function is like function in SpeedTest2.py but uses explicit numpy operations instead of the shorthand expression. It is expected to have similar performance characteristics, but the explicit operations may provide clarity in some contexts.
@jit(nopython=True)
def compute_arrays(arr1, arr2):
    return np.subtract(np.add(np.multiply(arr1, arr2), arr1), arr2)

# Create numpy arrays
arr1 = np.random.rand(array_size)
arr2 = np.random.rand(array_size)

# Warm-up run (triggers JIT compilation)
_ = compute_arrays(arr1, arr2)

# Benchmark the loop
start_time = time.perf_counter()

for i in range(num_iterations):
    result = compute_arrays(arr1, arr2)

end_time = time.perf_counter()

elapsed_time = end_time - start_time
avg_time_per_iteration = elapsed_time / num_iterations

print(f"Array size: {array_size:,}")
print(f"Iterations: {num_iterations}")
print(f"Total time: {elapsed_time:.4f} seconds")
print(f"Average time per iteration: {avg_time_per_iteration * 1000:.4f} ms")