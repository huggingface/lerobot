#!/usr/bin/env python
"""
Minimal Policy inference + benchmarking.

Features:
- End-to-end pipeline: dataset -> pre/post-processors -> policy.select_action
- Latency benchmarking with warmup, N trials, and M forwards/trial
- Reports mean/std/min/max and p50/p95 latencies (ms) per forward
- CPU RSS and CUDA (peak) memory footprint
- Works on CPU or CUDA; syncs properly for fair GPU timings

Example:
  python smolvla_bench.py \
    --repo_id AdilZtn/grab_red_cube_test_25 --episode 0 --sample_index 10 \
    --device cuda --num_trials 100 --forwards_per_trial 10 --warmup 20
"""

import argparse
import os
import statistics
import time
from typing import List

import torch
import psutil

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.factory import make_pre_post_processors


def bytes_to_human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser(description="SmolVLA inference + latency benchmark")
    parser.add_argument("--repo_id", type=str, default="AdilZtn/grab_red_cube_test_25",
                        help="HF dataset repo_id with language instructions")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to load")
    parser.add_argument("--sample_index", type=int, default=10, help="Sample index in the episode")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_obs_steps", type=int, default=1, help="Obs steps for SmolVLA")
    parser.add_argument("--n_action_steps", type=int, default=50, help="Action steps for SmolVLA")
    parser.add_argument("--chunk_size", type=int, default=50, help="Chunk size for SmolVLA")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of timing trials")
    parser.add_argument("--forwards_per_trial", type=int, default=1, help="Number of forwards per trial")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup forwards (not timed)")
    parser.add_argument("--print_each_trial", action="store_true", help="Print each trial's aggregate time")
    parser.add_argument("--policy_type", type=str, default="smolvla", help="Type of policy to benchmark")
    args = parser.parse_args()

    # Seed & deterministic-ish setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False  # leave False to avoid perf cliffs

    # Device
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if args.device == "cuda" and not use_cuda:
        print("[!] CUDA requested but unavailable. Falling back to CPU.")

    # Load dataset metadata
    ds_meta = LeRobotDatasetMetadata(args.repo_id)

    # Policy config & creation
    cfg = make_policy_config(
        args.policy_type,
        n_obs_steps=args.n_obs_steps,
        chunk_size=args.chunk_size, # comment this if policy_type = "diffusion"
        n_action_steps=args.n_action_steps,
        device=device,
    )

    policy: PreTrainedPolicy = make_policy(cfg, ds_meta=ds_meta)
    policy.eval()
    policy.to(device)

    # Pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=ds_meta.stats)

    # Dataset sample
    dataset = LeRobotDataset(args.repo_id, episodes=[args.episode])
    sample = dataset[args.sample_index]

    # Preprocess once; we will reuse the same batch for all forwards (typical for latency bench)
    preprocessed_batch = preprocessor(sample)

    # Helper to sync for fair timings
    def _sync():
        if use_cuda:
            torch.cuda.synchronize()

    # Warmup (to stabilize kernels/caches)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = policy.select_action(preprocessed_batch)
        _sync()

    # Memory footprint before timing
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # Timing
    trial_times_sec: List[float] = []

    with torch.no_grad():
        for t in range(args.num_trials):
            _sync()
            t0 = time.perf_counter()
            for _ in range(args.forwards_per_trial):
                _ = policy.select_action(preprocessed_batch)
            _sync()
            t1 = time.perf_counter()
            trial_dur = t1 - t0
            trial_times_sec.append(trial_dur)
            if args.print_each_trial:
                print(f"[trial {t+1:03d}] total {trial_dur*1000:.3f} ms "
                      f"({(trial_dur/args.forwards_per_trial)*1000:.3f} ms/forward)")

    # Memory footprint after timing
    rss_after = process.memory_info().rss
    rss_delta = rss_after - rss_before
    cuda_peak = torch.cuda.max_memory_allocated() if use_cuda else 0

    # Do a single real inference and postprocess to verify everything still works
    with torch.no_grad():
        action = policy.select_action(preprocessed_batch)
    postprocessed_action = postprocessor(action)

    # Summaries
    # Per-forward latencies in ms
    per_forward_ms = [(d / args.forwards_per_trial) * 1000.0 for d in trial_times_sec]
    per_forward_ms_sorted = sorted(per_forward_ms)

    mean_ms = statistics.fmean(per_forward_ms) if per_forward_ms else float("nan")
    std_ms = statistics.pstdev(per_forward_ms) if len(per_forward_ms) > 1 else 0.0
    min_ms = per_forward_ms_sorted[0] if per_forward_ms_sorted else float("nan")
    max_ms = per_forward_ms_sorted[-1] if per_forward_ms_sorted else float("nan")
    p50_ms = percentile(per_forward_ms_sorted, 50)
    p95_ms = percentile(per_forward_ms_sorted, 95)

    # Model size
    num_params = sum(p.numel() for p in policy.parameters())

    print("\n=== Inference Benchmark for ===", args.policy_type)
    print(f"Device: {device}")
    print(f"Trials: {args.num_trials} | Forwards/Trial: {args.forwards_per_trial} | Warmup: {args.warmup}")
    print(f"Model params: {num_params:,}")

    print("\nLatency per forward (ms):")
    print(f"  mean: {mean_ms:.3f}  std: {std_ms:.3f}")
    print(f"  min:  {min_ms:.3f}   max: {max_ms:.3f}")
    print(f"  p50:  {p50_ms:.3f}   p95: {p95_ms:.3f}")

    print("\nMemory footprint:")
    print(f"  CPU RSS before: {bytes_to_human(rss_before)}")
    print(f"  CPU RSS after : {bytes_to_human(rss_after)}  (Δ {bytes_to_human(rss_delta)})")
    if use_cuda:
        print(f"  CUDA peak allocated: {bytes_to_human(cuda_peak)} "
              f"(reset by reset_peak_memory_stats before timing)")

    # Quick shape dump from this run
    try:
        print("\nAction shapes:")
        print(f"  raw: {tuple(action.shape)}")
        print(f"  postprocessed: {tuple(postprocessed_action.shape)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
