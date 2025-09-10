import os
import time

import numpy as np
import torch

# Parameters
filename = "benchmark_data.dat"
shape = (10000, 10000)  # Large array
dtype = np.float32
torch_dtype = torch.float32

# Calculate file size
element_size = np.dtype(dtype).itemsize
file_size = shape[0] * shape[1] * element_size

# Create a large file and write random data to it
if not os.path.exists(filename) or os.path.getsize(filename) != file_size:
    data = np.random.rand(*shape).astype(dtype)
    with open(filename, "wb") as f:
        f.write(data.tobytes())

# Benchmark numpy.memmap
start_time = time.time()
data_np = np.memmap(filename, dtype=dtype, mode="r", shape=shape)
tensor_np = torch.from_numpy(data_np)
np_load_time = time.time() - start_time
print(f"np.memmap load time: {np_load_time:.4f} seconds")

# Benchmark torch.UntypedStorage
start_time = time.time()
storage = torch.UntypedStorage.from_file(filename, shared=True, nbytes=file_size)
tensor = torch.FloatTensor(storage).reshape(shape)
torch_load_time = time.time() - start_time
print(f"torch.UntypedStorage load time: {torch_load_time:.4f} seconds")

# Set NumPy print precision
# np.set_printoptions(precision=4)

# Print part of the arrays to compare precision
print("NumPy memmap array sample:\n", data_np[:5, :5])
print("PyTorch tensor sample:\n", tensor[:5, :5].numpy())

# Output the results
print(f"Numpy memmap load time: {np_load_time:.4f} seconds")
print(f"Torch UntypedStorage load time: {torch_load_time:.4f} seconds")
