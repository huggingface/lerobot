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

"""Single-image dataloading benchmark across the LeRobot loaders, MADE TO RUN ON A COMPUTE CLUSTER (SLURM).

This one file is both the orchestrator and the worker:

  * Run it with no ``--scenario`` (from a login node) and it submits a SERIAL sbatch chain of all
    scenarios below (no two network-bound jobs overlap, so CDN numbers stay clean).
  * Run it with ``--scenario <name>`` and it executes that single benchmark (this is what each sbatch
    job calls). The 2-node scenario is launched with ``srun`` and reads ``RANK``/``WORLD_SIZE`` so the
    streaming dataset splits shards per node.

Scenarios (all single-frame / non-SARM):
  1. ``mmap_local``             map-style LeRobotDataset over a LOCAL copy (``--local_root``, no network).
  2. ``mmap_local_maxworkers``  same, but workers scaled to saturate the node's cores (decode-bound).
  3. ``stream_hub``             StreamingLeRobotDataset from the Hub (allenai/MolmoAct2-BimanualYAM-Dataset).
  4. ``stream_bucket``          StreamingLeRobotDataset from a warmed storage bucket (1 node).
  5. ``stream_bucket_2node``    same warmed bucket, 2 nodes (split_dataset_by_node, per-rank results).

Reported per run: peak process-tree RSS (max memory), parallel throughput (samples/s, where a sample
is one timestep, plus decoded_frames/s = samples/s x num_cameras),
single-process throughput, shuffle randomness fraction (distinct episodes per batch / batch size),
fetch vs decode split (% of single-process per-sample time), first-batch latency, and p50/p95/p99
sample latency. Results are written as JSON + CSV under ``--out_dir``.

Submit the whole chain (from a login node, inside the repo). Point the scheduler env vars at your own
cluster's account/partition/qos, and ``--local_root`` at a local copy of the map-style dataset:
    ACCOUNT=<account> PARTITION=<partition> QOS=<qos> \\
        python examples/scaling/benchmark_dataloading.py --local_root /path/to/local/dataset
"""

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, StreamingLeRobotDataset
from lerobot.datasets.partition import group_episodes_by_files, partition_episodes

ROBOCASA_REPO = "pepijn223/robocasa_pretrain_human300_v4"
MOLMO_REPO = "allenai/MolmoAct2-BimanualYAM-Dataset"
MOLMO_BUCKET = "hf://buckets/pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket"
# MolmoAct2 is published without a codebase-version git tag, so the version-safe loader would refuse
# it; "main" pins the branch directly and skips that check.
MOLMO_REVISION = "main"

# Per-scenario sbatch shape. mem is generous for the streaming legs (32k-episode, 3-camera, 2.35 TB
# dataset keeps many AV1 decoders open); the local map-style leg is light. Optional ``num_workers`` /
# ``cpus`` override the CLI defaults for that leg.
# ``mmap_local_maxworkers``: map-style decode is CPU-bound and each worker decodes its cameras on
# parallel threads, so the saturation point is ~num_cpus / num_cameras workers (~90 concurrent decode
# threads). The 96-core H100 nodes here schedule at most 92 cpus/task, so we take 92 cpus / 30 workers.
SCENARIOS = {
    "mmap_local": {"kind": "map", "nodes": 1, "mem": "64G", "time": "01:00:00"},
    "mmap_local_maxworkers": {
        "kind": "map",
        "nodes": 1,
        "mem": "128G",
        "time": "01:00:00",
        "num_workers": 30,
        "cpus": 92,
    },
    "stream_hub": {"kind": "stream", "nodes": 1, "mem": "250G", "time": "03:00:00"},
    "stream_bucket": {"kind": "stream", "nodes": 1, "mem": "250G", "time": "03:00:00"},
    "stream_bucket_2node": {"kind": "stream", "nodes": 2, "mem": "250G", "time": "03:00:00"},
}


def _tree_rss_bytes() -> int:
    """Sum RSS of this process and all descendants via /proc (DataLoader workers are separate procs)."""
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
    """Background thread tracking peak process-tree RSS for the duration of the ``with`` block."""

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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[k]


class _TimedStreaming(StreamingLeRobotDataset):
    """StreamingLeRobotDataset that times the fetch stage (parquet/network row) separately from the
    decode stage (video decode + torch conversion in ``_finalize_sample``), so a single-process pass
    can attribute per-sample cost to fetch vs decode. Timing lives here in the benchmark, not in the
    library, to keep the dataset itself instrumentation-free."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fetch_s = 0.0
        self.decode_s = 0.0

    def __iter__(self):
        self._in_flight_epoch = self._epoch
        self._pipeline.set_epoch(self._in_flight_epoch)
        self._epoch += 1
        self.video_decoder_cache = self._make_video_decoder_cache()
        iterator = iter(self._pipeline)
        while True:
            t0 = time.perf_counter()
            try:
                row = next(iterator)
            except StopIteration:
                return
            t1 = time.perf_counter()
            sample = self._finalize_sample(row)
            t2 = time.perf_counter()
            self.fetch_s += t1 - t0
            self.decode_s += t2 - t1
            yield sample


def select_node_episodes(
    meta: LeRobotDatasetMetadata, num_partitions: int, index: int, cap: int
) -> list[int]:
    """This node's episode share, mirroring lerobot_train ``--data_partition=node``: group episodes by
    shared video files, LPT-balance the groups by frame count, take this node's bin (capped)."""
    episodes = list(range(meta.total_episodes))
    from_idx = meta.episodes["dataset_from_index"]
    to_idx = meta.episodes["dataset_to_index"]
    lengths = [int(to_idx[ep] - from_idx[ep]) for ep in episodes]
    if meta.video_keys:
        file_columns = {
            key: (meta.episodes[f"videos/{key}/chunk_index"], meta.episodes[f"videos/{key}/file_index"])
            for key in meta.video_keys
        }
    else:
        file_columns = {"data": (meta.episodes["data/chunk_index"], meta.episodes["data/file_index"])}
    episode_file_ids = [
        [(key, chunks[ep], files[ep]) for key, (chunks, files) in file_columns.items()] for ep in episodes
    ]
    groups = group_episodes_by_files(episode_file_ids)
    if len(groups) < num_partitions:
        groups = [[i] for i in range(len(episodes))]
    group_lengths = [sum(lengths[i] for i in g) for g in groups]
    bins = partition_episodes(group_lengths, num_partitions)
    chosen = sorted(episodes[i] for g in bins[index] for i in groups[g])
    return chosen[:cap] if cap and len(chosen) > cap else chosen


def build_dataset(scenario: str, args: argparse.Namespace):
    """Return (dataset, meta, is_map_style, info) for the scenario; single-frame (no delta windows)."""
    if scenario.startswith("mmap_local"):
        if not args.local_root:
            raise SystemExit("mmap_local needs --local_root pointing at a local LeRobotDataset copy.")
        meta = LeRobotDatasetMetadata(ROBOCASA_REPO, root=args.local_root)
        episodes = select_node_episodes(meta, args.num_partitions, args.partition_index, args.max_episodes)
        dataset = LeRobotDataset(ROBOCASA_REPO, root=args.local_root, episodes=episodes, tolerance_s=1e-3)
        return dataset, meta, True, {"loaded_episodes": len(episodes)}

    data_files_root = MOLMO_BUCKET if scenario.startswith("stream_bucket") else None
    meta = LeRobotDatasetMetadata(MOLMO_REPO, revision=MOLMO_REVISION)
    dataset = _TimedStreaming(
        MOLMO_REPO,
        revision=MOLMO_REVISION,
        data_files_root=data_files_root,
        episode_pool_size=args.episode_pool_size,
        max_buffer_input_shards=args.max_buffer_input_shards,
        video_decoder_cache_size=args.video_decoder_cache_size,
        video_fetch_workers=args.video_fetch_workers,
        tolerance_s=1e-3,
        # Throughput benchmark: don't gate on the one-row-group-per-episode invariant (a public
        # dataset may be collapsed); reshard() still yields per-episode shards where it holds.
        validate_row_groups=False,
    )
    return dataset, meta, False, {"num_shards": dataset.num_shards, "data_files_root": data_files_root}


def _split(fetch_s: float, decode_s: float, getitem_s: float, n_probe: int) -> dict:
    stage = fetch_s + decode_s
    return {
        "single_proc_samples_per_s": round(n_probe / getitem_s, 2) if getitem_s else None,
        "fetch_pct": round(100 * fetch_s / stage, 1) if stage else None,
        "decode_pct": round(100 * decode_s / stage, 1) if stage else None,
    }


def measure_fetch_decode_stream(dataset: _TimedStreaming, n_probe: int, warmup: int) -> dict:
    """Single-process pass attributing per-sample time to fetch (parquet/network row) vs decode (video)."""
    it = iter(dataset)
    for _ in range(warmup):  # exclude the cold shuffle-buffer fill from the ratio
        next(it)
    dataset.fetch_s = dataset.decode_s = 0.0
    t0 = time.perf_counter()
    for _ in range(n_probe):
        next(it)
    return _split(dataset.fetch_s, dataset.decode_s, time.perf_counter() - t0, n_probe)


def measure_fetch_decode_map(dataset: LeRobotDataset, n_probe: int, warmup: int) -> dict:
    """Same split for the map-style loader: fetch = raw tabular row (``get_raw_item``), decode = the rest
    of ``__getitem__`` (video decode + transforms). Local reads make fetch tiny and decode dominant.

    Random frames are resampled past any that torchcodec fails to decode, so a single flaky frame can't
    abort the whole benchmark (the parallel DataLoader pass draws its own fresh random frames)."""
    rng = random.Random(0)
    n = len(dataset)
    fetch_s = getitem_s = 0.0
    warmed = measured = skipped = attempts = 0
    while measured < n_probe and attempts < (warmup + n_probe) * 10:
        attempts += 1
        i = rng.randrange(n)
        try:
            t0 = time.perf_counter()
            dataset.get_raw_item(i)
            t1 = time.perf_counter()
            dataset[i]
            t2 = time.perf_counter()
        except Exception:
            skipped += 1
            continue
        if warmed < warmup:
            warmed += 1
            continue
        fetch_s += t1 - t0
        getitem_s += t2 - t1
        measured += 1
    if skipped:
        print(f"map fetch/decode probe skipped {skipped} undecodable frame(s)", flush=True)
    return _split(fetch_s, max(0.0, getitem_s - fetch_s), getitem_s, measured)


def run_scenario(scenario: str, args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(args.device)

    dataset, meta, is_map_style, info = build_dataset(scenario, args)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=is_map_style,  # map-style: global random shuffle; streaming: shuffled inside the dataset
        pin_memory=device.type == "cuda",
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    sample_latencies_ms: list[float] = []
    episodes_per_batch: list[int] = []
    samples = 0
    first_batch_latency_s = None
    steady_start = None

    t_start = time.perf_counter()
    t_prev = t_start
    with PeakRSSSampler() as rss:
        for i, batch in enumerate(loader):
            for value in batch.values():
                if torch.is_tensor(value):
                    value.to(device, non_blocking=device.type == "cuda")
            now = time.perf_counter()
            if first_batch_latency_s is None:
                first_batch_latency_s = now - t_start
            if i == args.warmup_batches:
                steady_start = now
            elif i > args.warmup_batches:
                sample_latencies_ms.append((now - t_prev) / args.batch_size * 1000.0)
                samples += args.batch_size
                ep = batch.get("episode_index")
                if torch.is_tensor(ep):
                    episodes_per_batch.append(int(torch.unique(ep).numel()))
            t_prev = now
            # Measure throughput over a fixed wall-clock window (after warmup) so every scenario is
            # compared over the same duration regardless of its speed; num_batches is only a safety cap.
            if steady_start is not None and (now - steady_start) >= args.duration_s:
                break
            if i + 1 >= args.num_batches:
                break
    peak_rss_gb = round(rss.peak_bytes / 1e9, 2) if rss.peak_bytes else None

    now = time.perf_counter()
    elapsed = now - t_start
    steady_elapsed_s = (now - steady_start) if steady_start is not None else elapsed

    if samples == 0:
        raise SystemExit(
            f"FAILED: 0 samples in {args.duration_s}s for scenario={scenario} "
            "(inspect worker logs; try --num_workers 0 to surface the exception)."
        )

    # Single-process fetch/decode split + single-proc throughput. Run AFTER the DataLoader pass: this
    # decodes video in the main process, which must stay decode-clean until the workers have forked
    # (decoding before fork corrupts the workers' torchcodec state).
    del loader
    if is_map_style:
        fetch_decode = measure_fetch_decode_map(dataset, args.probe_samples, args.probe_warmup)
    else:
        fetch_decode = measure_fetch_decode_stream(dataset, args.probe_samples, args.probe_warmup)

    image_shape = list(meta.features[meta.video_keys[0]]["shape"]) if meta.video_keys else None
    num_cameras = len(meta.video_keys)
    results = {
        "scenario": scenario,
        "rank": rank,
        "world_size": world_size,
        "loader": "map_style" if is_map_style else "streaming",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "episode_pool_size": None if is_map_style else args.episode_pool_size,
        "max_buffer_input_shards": None
        if is_map_style
        else (args.max_buffer_input_shards or args.episode_pool_size),
        **info,
        "num_cameras": num_cameras,
        "image_shape": image_shape,
        "fps": meta.fps,
        "peak_rss_gb": peak_rss_gb,
        "samples_measured": samples,
        "steady_window_s": round(steady_elapsed_s, 2),
        "first_batch_latency_s": round(first_batch_latency_s or float("nan"), 3),
        # Parallel throughput over the steady window (excludes warmup + the prefetch queue it filled).
        # A sample is one timestep (one dataset item); it decodes num_cameras video frames.
        "samples_per_s": round(samples / steady_elapsed_s, 2) if steady_elapsed_s else 0.0,
        "decoded_frames_per_s": round(samples / steady_elapsed_s * num_cameras, 2)
        if steady_elapsed_s
        else 0.0,
        **fetch_decode,
        # Distinct episodes per batch / batch size: ~1.0 ≈ map-style uniform, low ≈ correlated samples.
        "shuffle_randomness_frac": round(statistics.mean(episodes_per_batch) / args.batch_size, 3)
        if episodes_per_batch
        else None,
        "p50_sample_latency_ms": round(statistics.median(sample_latencies_ms), 3)
        if sample_latencies_ms
        else None,
        "p95_sample_latency_ms": round(percentile(sample_latencies_ms, 95), 3),
        "p99_sample_latency_ms": round(percentile(sample_latencies_ms, 99), 3),
        "total_time_s": round(elapsed, 2),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{scenario}_bs{args.batch_size}_w{args.num_workers}_r{rank}of{world_size}"
    (out_dir / f"{tag}.json").write_text(json.dumps(results, indent=2))
    flat = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in results.items()}
    with open(out_dir / f"{tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    print(json.dumps(results, indent=2), flush=True)
    print(f"Wrote {out_dir / tag}.json and .csv", flush=True)


def submit_chain(args: argparse.Namespace) -> None:
    """Submit every scenario as a serial sbatch chain (one network-bound job at a time).

    Bodies are passed to ``sbatch --wrap`` as a single argv (no outer shell), so ``$SLURM_PROCID`` /
    ``$SLURM_NTASKS`` stay literal and expand at job runtime, not at submit time.
    """
    this_file = Path(__file__).resolve()
    repo_dir = str(this_file.parents[2])  # <repo>/examples/scaling/<this file>
    logs = Path(repo_dir) / "logs"
    logs.mkdir(exist_ok=True)
    run = f"conda run --no-capture-output -n {args.conda_env} python"
    common = (
        f"--batch_size {args.batch_size} "
        f"--prefetch_factor {args.prefetch_factor} --episode_pool_size {args.episode_pool_size} "
        f"--video_decoder_cache_size {args.video_decoder_cache_size} --duration_s {args.duration_s} "
        f"--num_batches {args.num_batches} --out_dir {args.out_dir}"
    )
    if args.max_buffer_input_shards is not None:
        common += f" --max_buffer_input_shards {args.max_buffer_input_shards}"
    if args.local_root:
        common += f" --local_root {args.local_root}"
    env_prefix = "export TOKENIZERS_PARALLELISM=false"
    sched = []
    for opt, env in (("--account", "ACCOUNT"), ("--partition", "PARTITION"), ("--qos", "QOS")):
        if os.environ.get(env):
            sched.append(f"{opt}={os.environ[env]}")

    selected = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
    prev = ""
    for scenario in selected:
        cfg = SCENARIOS[scenario]
        nw = cfg.get("num_workers", args.num_workers)
        cpus = cfg.get("cpus", nw + 4)
        worker = f"{run} {this_file} --scenario {scenario} --num_workers {nw} {common}"
        if cfg["nodes"] > 1:
            # One task per node; each exports RANK/WORLD_SIZE so the stream splits shards per node.
            inner = f"export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS && cd {repo_dir} && {env_prefix} && {worker}"
            body = f"srun --export=ALL bash -c '{inner}'"
            node_flags = [f"--nodes={cfg['nodes']}", "--ntasks-per-node=1", "--gpus-per-node=1"]
        else:
            body = f"cd {repo_dir} && {env_prefix} && {worker}"
            node_flags = ["--nodes=1", "--ntasks=1", "--gpus=1"]
        cmd = [
            "sbatch",
            "--parsable",
            f"--job-name=dlbench_{scenario}",
            *node_flags,
            f"--cpus-per-task={cpus}",
            f"--mem={cfg['mem']}",
            f"--time={cfg['time']}",
            f"--output={logs}/%x-%j.out",
            *sched,
        ]
        if prev:
            cmd.append(f"--dependency=afterany:{prev}")
        cmd += ["--wrap", body]
        jid = subprocess.check_output(cmd, text=True).strip().split(";")[0]
        print(f"submitted {jid}  dlbench_{scenario}{f'  (after {prev})' if prev else ''}", flush=True)
        prev = jid

    print(f"\nSubmitted {len(selected)} jobs as a serial chain. Results: {args.out_dir}/*.json", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scenario",
        choices=list(SCENARIOS),
        default=None,
        help="Run ONE scenario (worker mode). Omit to submit the whole chain (orchestrator mode).",
    )
    p.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Orchestrator only: comma-separated subset of scenarios to submit (default: all).",
    )
    p.add_argument("--local_root", type=str, default=None, help="Local LeRobotDataset copy for mmap_local.")
    p.add_argument(
        "--num_partitions", type=int, default=8, help="Node count for mmap_local episode partition."
    )
    p.add_argument("--partition_index", type=int, default=0)
    p.add_argument(
        "--max_episodes", type=int, default=512, help="Cap mmap_local episodes to the local share."
    )
    p.add_argument(
        "--video_fetch_workers",
        type=int,
        default=16,
        help="Concurrent byte-range fetch threads per consumer (the fetch-throughput knob; was 4).",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument(
        "--episode_pool_size", type=int, default=1024, help="Streaming shuffle pool (randomness knob)."
    )
    p.add_argument(
        "--max_buffer_input_shards",
        type=int,
        default=None,
        help="Concurrently-live random episodes feeding the pool after reshard() "
        "(default: episode_pool_size). The frac knob; set >= batch_size for frac->1.",
    )
    p.add_argument(
        "--video_decoder_cache_size", type=int, default=32, help="Max open video decoders (bounds RAM)."
    )
    p.add_argument(
        "--duration_s", type=float, default=60.0, help="Steady-state measurement window (seconds)."
    )
    p.add_argument(
        "--num_batches", type=int, default=1_000_000, help="Safety cap; duration_s governs the window."
    )
    p.add_argument("--warmup_batches", type=int, default=5, help="Excluded from steady-state throughput.")
    p.add_argument(
        "--probe_samples", type=int, default=100, help="Single-process samples for fetch/decode split."
    )
    p.add_argument(
        "--probe_warmup", type=int, default=10, help="Samples skipped before the fetch/decode probe."
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--conda_env", type=str, default="lerobot", help="Conda env the chained jobs run in.")
    p.add_argument("--out_dir", type=str, default="benchmarks/streaming/results_dataloading")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenario is None:
        if torch.cuda.is_available():
            print(
                "NOTE: no --scenario given, submitting the SLURM chain. This benchmark is meant to run on a "
                "compute cluster; run from a login node with ACCOUNT/PARTITION/QOS set.",
                file=sys.stderr,
            )
        submit_chain(args)
    else:
        run_scenario(args.scenario, args)


if __name__ == "__main__":
    main()
