"""Benchmark: prepare_observation_for_inference"""
import time
import numpy as np
import torch
from lerobot.policies.utils import prepare_observation_for_inference as optimized_fn

def original_fn(observation, device, task=None, robot_type=None):
    for name in observation:
        observation[name] = torch.from_numpy(observation[name])
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)
    observation["task"] = task if task else ""
    observation["robot_type"] = robot_type if robot_type else ""
    return observation

def make_obs():
    return {
        "observation.images.top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "observation.images.wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "observation.state": np.random.rand(14).astype(np.float32),
    }

def bench(fn, device, n=200, label=""):
    for _ in range(10):
        fn(make_obs(), device)
    times = []
    for _ in range(n):
        obs = make_obs()
        t0 = time.perf_counter()
        fn(obs, device)
        times.append((time.perf_counter() - t0) * 1000)
    mean_ms = sum(times) / len(times)
    p95_ms = sorted(times)[int(0.95 * len(times))]
    print(f"{label:30s}  mean={mean_ms:.3f}ms  p95={p95_ms:.3f}ms  (n={n})")
    return mean_ms

if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Device: {device}\n")
    orig = bench(original_fn, device, label="original")
    opt  = bench(optimized_fn, device, label="optimized")
    print(f"\nSpeedup: {(orig - opt) / orig * 100:+.1f}%")
