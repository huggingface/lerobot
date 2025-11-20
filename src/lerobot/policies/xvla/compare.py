import numpy as np
import os

# --- CHANGE THESE TWO PATHS ---
dir_a = "/home/jade_choghari/robot/lerobot"
dir_b = "/home/jade_choghari/robot/robot2/lerobot"

# keys = ["input_ids", "image_input", "image_mask", "domain_id", "proprio"]
keys = ["domain_id"]

def load_np(path, key):
    file_path = os.path.join(path, f"{key}.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    return np.load(file_path)

for k in keys:
    print(f"\n===== {k} =====")
    a = load_np(dir_a, k)
    b = load_np(dir_b, k)
    print(a)
    print(b)
    print("Shapes:", a.shape, b.shape)
    same_shape = a.shape == b.shape
    print("Same shape:", same_shape)

    if not same_shape:
        continue

    # Allclose for numeric, array_equal for boolean
    if a.dtype == bool or b.dtype == bool:
        equal = np.array_equal(a, b)
        print("Equal:", equal)

        # For boolean arrays, diff = xor
        diff = np.sum(a ^ b)
        print("Num differing elements:", diff)
    else:
        close = np.allclose(a, b, atol=1e-6, rtol=1e-6)
        print("Allclose:", close)

        diff = np.max(np.abs(a - b))
        print("Max difference:", diff)
