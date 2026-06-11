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
import os
import statistics
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDatasetMetadata, StreamingLeRobotDataset
from lerobot.utils.constants import ACTION


def _tree_rss_bytes() -> int:
    """Sum RSS of this process and all its descendants via /proc (Linux only; 0 elsewhere).

    DataLoader workers are separate processes, so the parent's own RSS misses most of the pipeline's
    memory. Walking the process tree captures the real footprint (parquet buffers + decoders + shuffle).
    """
    try:
        children: dict[int, list[int]] = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as f:
                    ppid = int(f.read().split(") ", 1)[1].split()[1])
                children.setdefault(ppid, []).append(int(entry))
            except (OSError, ValueError, IndexError):
                pass
        total, stack = 0, [os.getpid()]
        while stack:
            cur = stack.pop()
            try:
                with open(f"/proc/{cur}/statm") as f:
                    total += int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
            stack.extend(children.get(cur, []))
        return total
    except OSError:
        return 0


class PeakRSSSampler:
    """Background thread tracking peak process-tree RSS for the duration of the `with` block."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, _tree_rss_bytes())
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PeakRSSSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


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
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="DataLoader batches prefetched per worker. Higher hides IO/decode latency but raises RAM "
        "(prefetch_factor x num_workers x batch_size decoded frames held in flight). Ignored if num_workers=0.",
    )
    parser.add_argument("--buffer_size", type=int, default=2000)
    parser.add_argument(
        "--max_num_shards",
        type=int,
        default=16,
        help="Cap on concurrently-open stream shards. Each open shard holds ~one parquet row group in "
        "RAM; reading from an hf:// bucket buffers ~5x more per shard than hf:// datasets, so lower this "
        "(e.g. to num_workers) for bucket sources to avoid OOM. All data is still covered via re-sharding.",
    )
    parser.add_argument("--video_decoder_cache_size", type=int, default=None)
    parser.add_argument(
        "--episode_pool_size",
        type=int,
        default=None,
        help="A3 shuffle: keep this many full episodes live and sample frames uniformly across them "
        "(mixing radius = this many episodes). Unset = default per-shard reservoir shuffle.",
    )
    parser.add_argument(
        "--video_decode_device",
        type=str,
        default="cpu",
        help="Decode device passed to torchcodec. 'cuda' offloads decode to the GPU's NVDEC engine "
        "(needs a CUDA-enabled torchcodec build). With num_workers>0 this forces the 'spawn' start method.",
    )
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
        max_num_shards=args.max_num_shards,
        video_decoder_cache_size=args.video_decoder_cache_size,
        video_decode_device=args.video_decode_device,
        episode_pool_size=args.episode_pool_size,
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

    gpu_decode = args.video_decode_device.startswith("cuda")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # GPU-decoded frames are already on the GPU, so CPU pinning is irrelevant (and pinning CUDA
        # tensors errors). Pin only when decode is on CPU and we copy to a CUDA device.
        pin_memory=device.type == "cuda" and not gpu_decode,
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        # CUDA cannot initialize in forked workers; NVDEC decode in workers needs the spawn start method.
        multiprocessing_context="spawn" if gpu_decode and args.num_workers > 0 else None,
    )

    sample_latencies_ms: list[float] = []
    episodes_per_batch: list[int] = []  # shuffle-randomness proxy: distinct episodes within a batch
    frames = 0
    first_batch_latency_s = None
    steady_start = None  # wall-clock start of the post-warmup measurement window

    t_start = time.perf_counter()
    t_prev = t_start
    with PeakRSSSampler() as rss:
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
                ep = batch.get("episode_index")
                if torch.is_tensor(ep):
                    episodes_per_batch.append(int(torch.unique(ep).numel()))
            t_prev = now
            if i + 1 >= args.num_batches:
                break
    peak_rss_gb = round(rss.peak_bytes / 1e9, 2) if rss.peak_bytes else None

    now = time.perf_counter()
    elapsed = now - t_start
    # Wall-clock throughput over the steady window. NOT sum(inter-batch gaps): under async prefetch those
    # gaps collapse to ~0 (the consumer drains a pre-filled queue) and overstate throughput by ~100x.
    steady_elapsed_s = (now - steady_start) if steady_start is not None else elapsed
    cache_stats = dataset.video_decoder_cache_stats()
    timing = dataset.timing_stats()  # cumulative decode/fetch seconds summed across workers
    # Image (camera frame) resolution as decoded, e.g. [C, H, W]. Read from the dataset feature contract.
    image_shape = (
        list(meta.features[meta.video_keys[0]]["shape"]) if meta.video_keys else None
    )
    # Decode/fetch overlap in wall-clock (workers run in parallel), so normalize against the total worker
    # budget (num_workers x wallclock) to express each stage as a fraction of available worker time.
    worker_budget_s = max(args.num_workers, 1) * elapsed
    decode_pct = round(100 * timing["decode_s_total"] / worker_budget_s, 1) if worker_budget_s else None
    fetch_pct = round(100 * timing["fetch_s_total"] / worker_budget_s, 1) if worker_budget_s else None

    # A 0-frame run is a failure, not a 0-throughput result: the pipeline produced no batches (decode
    # error swallowed in workers, all batches dropped by drop_last, etc.). Exit non-zero so the job is
    # never reported green with NaN/zero numbers.
    if frames == 0:
        raise SystemExit(
            f"FAILED: measured 0 frames over {args.num_batches} requested batches "
            f"(cache misses={cache_stats.get('misses', 0)}, hits={cache_stats.get('hits', 0)}). "
            "The data pipeline yielded no usable batches — inspect worker logs for decode errors. "
            "Try --num_workers 0 to surface the underlying exception directly."
        )

    results = {
        "repo_id": args.repo_id,
        "source": args.source,
        "mode": args.mode,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "buffer_size": args.buffer_size,
        "episode_pool_size": args.episode_pool_size,
        "episodes_per_batch_mean": round(statistics.mean(episodes_per_batch), 1)
        if episodes_per_batch
        else None,
        # Fraction of a batch that is distinct episodes; ~1.0 ≈ map-style uniform, low ≈ correlated.
        "shuffle_randomness_frac": round(statistics.mean(episodes_per_batch) / args.batch_size, 3)
        if episodes_per_batch
        else None,
        "num_cameras": len(meta.video_keys),
        "image_shape": image_shape,
        "fps": meta.fps,
        "device": str(device),
        "video_decode_device": args.video_decode_device,
        "peak_rss_gb": peak_rss_gb,
        "frames_measured": frames,
        "first_batch_latency_s": round(first_batch_latency_s or float("nan"), 4),
        "frames_per_s_node": round(frames / steady_elapsed_s, 2) if steady_elapsed_s else 0.0,
        "samples_per_s": round(frames / steady_elapsed_s, 2) if steady_elapsed_s else 0.0,
        "p50_sample_latency_ms": round(statistics.median(sample_latencies_ms), 3)
        if sample_latencies_ms
        else None,
        "p95_sample_latency_ms": round(percentile(sample_latencies_ms, 95), 3),
        "p99_sample_latency_ms": round(percentile(sample_latencies_ms, 99), 3),
        "total_time_s": round(elapsed, 2),
        "steady_time_s": round(steady_elapsed_s, 2),
        "wallclock_s": round(elapsed, 2),
        "decode_s_total": timing["decode_s_total"],
        "fetch_s_total": timing["fetch_s_total"],
        "decode_pct_worker_time": decode_pct,
        "fetch_pct_worker_time": fetch_pct,
        "video_decoder_cache": cache_stats,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_tag = f"_ep{args.episode_pool_size}" if args.episode_pool_size else ""
    tag = (
        f"{args.source}_{args.mode}_bs{args.batch_size}_w{args.num_workers}"
        f"_pf{args.prefetch_factor}{pool_tag}_{args.video_decode_device}"
    )
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
