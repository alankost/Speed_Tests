import numpy as np
import time
from numba import vectorize

# Configuration
array_size = 1_000_000
num_iterations = 100

@vectorize(['float32(float32, float32)'], target='parallel')
def compute_vectorized(a, b):
    return a * b + a - b

# Create numpy arrays with float32
arr1 = np.random.rand(array_size).astype(np.float32)
arr2 = np.random.rand(array_size).astype(np.float32)

# Warm-up run (triggers JIT compilation)
_ = compute_vectorized(arr1, arr2)

# Benchmark the loop
start_time = time.perf_counter()

for i in range(num_iterations):
    result = compute_vectorized(arr1, arr2)

end_time = time.perf_counter()

elapsed_time = end_time - start_time
avg_time_per_iteration = elapsed_time / num_iterations

print(f"Array size: {array_size:,}")
print(f"Data type: float32")
print(f"Iterations: {num_iterations}")
print(f"Total time: {elapsed_time:.4f} seconds")
print(f"Average time per iteration: {avg_time_per_iteration * 1000:.4f} ms")
