#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Benchmark the production StreamingLeRobotDataset path used by lerobot-train."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from lerobot.datasets import StreamingLeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Use the first N episodes.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--episode-pool-size", type=int, default=32)
    parser.add_argument("--prefetch-episodes", type=int, default=8)
    parser.add_argument("--byte-budget-gb", type=float, default=8.0)
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--measure-batches", type=int, default=128)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        return subprocess.run(
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main_process_max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024**2 if sys.platform == "darwin" else rss / 1024


def child_process_max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return rss / 1024**2 if sys.platform == "darwin" else rss / 1024


def main() -> None:
    args = parse_args()
    episodes = list(range(args.episodes)) if args.episodes is not None else None

    init_start = time.perf_counter()
    dataset = StreamingLeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=episodes,
        revision=args.revision,
        data_root=args.data_root,
        episode_pool_size=args.episode_pool_size,
        prefetch_episodes=args.prefetch_episodes,
        byte_budget_gb=args.byte_budget_gb,
        max_num_shards=max(1, args.num_workers),
        return_uint8=True,
    )
    dataset_init_s = time.perf_counter() - init_start

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if args.num_workers else None,
        persistent_workers=args.num_workers > 0,
    )
    iterator = iter(loader)
    waits: list[float] = []
    measured_samples = 0
    measured_indices: list[int] = []
    first_batch_s = 0.0
    exhausted = False

    try:
        for batch_index in range(args.warmup_batches + args.measure_batches):
            wait_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                exhausted = True
                break
            wait_s = time.perf_counter() - wait_start
            if batch_index == 0:
                first_batch_s = wait_s
            if batch_index >= args.warmup_batches:
                waits.append(wait_s)
                indices = batch["index"].reshape(-1).tolist()
                measured_indices.extend(int(index) for index in indices)
                measured_samples += len(indices)
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()

    measured_wall_s = sum(waits)
    summary = {
        "repo_id": args.repo_id,
        "revision": str(dataset.revision),
        "git_commit": git_commit(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "dataset_init_s": dataset_init_s,
        "first_batch_s": first_batch_s,
        "measured_batches": len(waits),
        "measured_samples": measured_samples,
        "measured_wall_s": measured_wall_s,
        "samples_s": measured_samples / measured_wall_s if measured_wall_s else 0.0,
        "batch_wait_mean_ms": statistics.fmean(waits) * 1000 if waits else 0.0,
        "batch_wait_p50_ms": percentile(waits, 0.50) * 1000,
        "batch_wait_p95_ms": percentile(waits, 0.95) * 1000,
        "batch_wait_p99_ms": percentile(waits, 0.99) * 1000,
        "duplicate_indices": measured_samples - len(set(measured_indices)),
        "epoch_exhausted": exhausted,
        "main_process_max_rss_mb": main_process_max_rss_mb(),
        "worker_process_max_rss_mb": child_process_max_rss_mb(),
        "config": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "episode_pool_size": args.episode_pool_size,
            "prefetch_episodes": args.prefetch_episodes,
            "byte_budget_gb": args.byte_budget_gb,
            "warmup_batches": args.warmup_batches,
            "measure_batches": args.measure_batches,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
