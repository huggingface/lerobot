# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataloading-only benchmark for StreamingLeRobotDataset.

A dummy consumer pulls batches and moves them to the device; no model runs, so the numbers isolate the
data pipeline (parquet read + video decode + delta windowing + shuffle). Reports per-node throughput and
sample-latency percentiles, plus video-decoder-cache reuse stats, and emits JSON + CSV.

Frame modes (matching the streaming design targets):
  - ``single``: one frame, all cameras (target >= 120 frames/s/node).
  - ``sarm``:   an 8-step window spaced 1s (delta over 8s) (target >= 320 frames/s/node).

Example (stream from the Hub, single node):

    python benchmarks/streaming/benchmark_streaming.py \
        --repo_id pepijn223/robocasa_pretrain_human300_v4 --mode sarm \
        --batch_size 64 --num_workers 12 --num_batches 200 --out_dir benchmarks/streaming/results

Distributed / multinode runs go through Accelerate; see ``slurm/benchmark_streaming_robocasa.sh``. Set
``--source`` purely for labeling the output (``hub`` / ``bucket`` / ``warmed_bucket``); the actual source
is whatever ``--repo_id``/``--root`` point at. See the README for bucket prewarming.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDatasetMetadata, StreamingLeRobotDataset
from lerobot.utils.constants import ACTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--root", type=str, default=None, help="Local/prewarmed root (else stream from Hub).")
    parser.add_argument(
        "--data_files_root",
        type=str,
        default=None,
        help="fsspec root for bulk data/videos, e.g. hf://buckets/<owner>/<name>. Metadata still loads "
        "from --repo_id on the Hub. Use for bucket / warmed_bucket sources.",
    )
    parser.add_argument("--mode", choices=["single", "sarm"], default="single")
    parser.add_argument("--source", type=str, default="hub", help="Label only: hub | bucket | warmed_bucket.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--buffer_size", type=int, default=2000)
    parser.add_argument("--video_decoder_cache_size", type=int, default=None)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--warmup_batches", type=int, default=5, help="Excluded from steady-state stats.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="benchmarks/streaming/results")
    return parser.parse_args()


def build_dataset(args: argparse.Namespace, meta: LeRobotDatasetMetadata) -> StreamingLeRobotDataset:
    # sarm: an 8-step window spaced 1s => an 8s delta window (the SARM stress case).
    delta_timestamps = {ACTION: [float(t) for t in range(8)]} if args.mode == "sarm" else None
    return StreamingLeRobotDataset(
        args.repo_id,
        root=args.root,
        data_files_root=args.data_files_root,
        delta_timestamps=delta_timestamps,
        buffer_size=args.buffer_size,
        video_decoder_cache_size=args.video_decoder_cache_size,
        tolerance_s=1e-3,
    )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    dataset = build_dataset(args, meta)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    sample_latencies_ms: list[float] = []
    frames = 0
    first_batch_latency_s = None
    steady_start = None  # wall-clock start of the post-warmup measurement window

    t_start = time.perf_counter()
    t_prev = t_start
    for i, batch in enumerate(loader):
        # Dummy consume: move tensors to the device, mimicking what a real trainer would do.
        for value in batch.values():
            if torch.is_tensor(value):
                value.to(device, non_blocking=device.type == "cuda")
        now = time.perf_counter()
        if first_batch_latency_s is None:
            first_batch_latency_s = now - t_start

        if i == args.warmup_batches:
            # Start the steady window here; the slow first batch and the prefetch queue it filled are
            # excluded so throughput reflects sustained production, not draining a pre-filled queue.
            steady_start = now
        elif i > args.warmup_batches:
            sample_latencies_ms.append((now - t_prev) / args.batch_size * 1000.0)
            frames += args.batch_size
        t_prev = now
        if i + 1 >= args.num_batches:
            break

    now = time.perf_counter()
    elapsed = now - t_start
    # Wall-clock throughput over the steady window. NOT sum(inter-batch gaps): under async prefetch those
    # gaps collapse to ~0 (the consumer drains a pre-filled queue) and overstate throughput by ~100x.
    steady_elapsed_s = (now - steady_start) if steady_start is not None else elapsed
    cache_stats = dataset.video_decoder_cache_stats()

    results = {
        "repo_id": args.repo_id,
        "source": args.source,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "buffer_size": args.buffer_size,
        "num_cameras": len(meta.video_keys),
        "fps": meta.fps,
        "device": str(device),
        "frames_measured": frames,
        "first_batch_latency_s": round(first_batch_latency_s or float("nan"), 4),
        "frames_per_s_node": round(frames / steady_elapsed_s, 2) if steady_elapsed_s else 0.0,
        "samples_per_s": round(frames / steady_elapsed_s, 2) if steady_elapsed_s else 0.0,
        "p50_sample_latency_ms": round(statistics.median(sample_latencies_ms), 3)
        if sample_latencies_ms
        else None,
        "p95_sample_latency_ms": round(percentile(sample_latencies_ms, 95), 3),
        "p99_sample_latency_ms": round(percentile(sample_latencies_ms, 99), 3),
        "wallclock_s": round(elapsed, 2),
        "video_decoder_cache": cache_stats,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.source}_{args.mode}_bs{args.batch_size}_w{args.num_workers}"
    (out_dir / f"{tag}.json").write_text(json.dumps(results, indent=2))
    flat = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in results.items()}
    with open(out_dir / f"{tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)

    print("Command config:", vars(args))
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_dir / tag}.json and .csv")


if __name__ == "__main__":
    main()
