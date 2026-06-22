#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import os
import random
import resource
import tempfile
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.episode_video_streaming import (
    EpisodeByteCache,
    EpisodeVideoManifest,
    NativeHTTPRangeFetcher,
    assert_hf_hub_range_cache_branch,
)
from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames_torchcodec

DEFAULT_REPO = "allenai/MolmoAct2-BimanualYAM-Dataset"
DEFAULT_REVISION = "e9f21ae15074330839f2ac25ed4b49d76dfa1f9c"
DEFAULT_DATA_ROOT = "hf://buckets/pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket"
SIDECAR_CACHE_DIR = Path(tempfile.gettempdir()) / "lerobot-sidecars"
FULL_SIDECAR_NAME = "molmoact2-full.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark episode-level streaming mini-MP4 cache.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--strategy",
        choices=("both", "full", "indexed", "remote-decoder", "native-http"),
        default="both",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--range-backend",
        choices=("fsspec", "native-http"),
        default="fsspec",
        help="Range reader used by indexed/full episode-pool fetch tracks.",
    )
    parser.add_argument("--num-episodes", type=int, default=512)
    parser.add_argument(
        "--manifest-episodes",
        type=int,
        default=None,
        help="Limit manifest construction to the first N episodes for local smoke tests.",
    )
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--native-http-connections",
        type=int,
        default=None,
        help="Max HTTP connections for --range-backend native-http. Defaults to --workers.",
    )
    parser.add_argument(
        "--native-http-retries",
        type=int,
        default=8,
        help="Retries per native HTTP range request.",
    )
    parser.add_argument(
        "--native-http-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for native HTTP requests.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=10.0,
        help="Print episode-pool fill progress every N seconds. Set 0 to disable.",
    )
    parser.add_argument(
        "--http-failure-log",
        default=None,
        help="Optional JSONL file for failed/retried HTTP range attempts.",
    )
    parser.add_argument(
        "--include-decode",
        action="store_true",
        help="Also run decoder-opening/frame-decode comparison tracks. Fetch-only is the default.",
    )
    parser.add_argument("--include-pool-sampling", action="store_true")
    parser.add_argument("--pool-random-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-samples-s", type=float, default=500.0)
    parser.add_argument("--stream-samples", type=int, default=4096)
    parser.add_argument("--pool-samples-per-episode", type=int, default=160)
    parser.add_argument("--stream-prefetch-episodes", type=int, default=16)
    parser.add_argument("--decode-workers", type=int, default=1)
    parser.add_argument("--prefetch-ahead", type=int, default=8)
    parser.add_argument("--frames-per-episode", type=int, default=16)
    parser.add_argument("--max-probe-mb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--byte-budget-gb", type=float, default=80)
    parser.add_argument(
        "--in-memory", action="store_true", help="Accepted for compatibility; manifest is always in memory."
    )
    parser.add_argument("--no-hub-branch-assert", action="store_true")
    return parser.parse_args()


def _episode_pool(total: int, requested: int, pool_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    upper = min(total, requested)
    if pool_size > upper:
        raise ValueError(f"pool-size={pool_size} exceeds available episodes={upper}")
    return rng.sample(range(upper), pool_size)


def _timestamps(manifest: EpisodeVideoManifest, episodes: Sequence[int], frames_per_episode: int, seed: int):
    rng = random.Random(seed)
    out: dict[tuple[int, str], list[float]] = {}
    for ep in episodes:
        for camera_key in manifest.video_keys:
            span = manifest.lookup(ep, camera_key)
            lo = span.first_pts
            hi = max(span.last_pts, lo)
            out[(ep, camera_key)] = sorted(rng.uniform(lo, hi) for _ in range(frames_per_episode))
    return out


def _timestamps_from_meta(
    meta: LeRobotDatasetMetadata, episodes: Sequence[int], frames_per_episode: int, seed: int
) -> dict[tuple[int, str], list[float]]:
    rng = random.Random(seed)
    out: dict[tuple[int, str], list[float]] = {}
    for ep in episodes:
        row = meta.episodes[ep]
        for camera_key in meta.video_keys:
            lo = float(row[f"videos/{camera_key}/from_timestamp"])
            hi = max(float(row[f"videos/{camera_key}/to_timestamp"]), lo)
            out[(ep, camera_key)] = sorted(rng.uniform(lo, hi) for _ in range(frames_per_episode))
    return out


def _bytes_for(manifest: EpisodeVideoManifest, episodes: Sequence[int]) -> int:
    total = 0
    for ep in episodes:
        for camera_key in manifest.video_keys:
            total += manifest.lookup(ep, camera_key).mdat_length
    return total


def _random_training_samples(episodes: Sequence[int], count: int, seed: int) -> list[tuple[int, float]]:
    rng = random.Random(seed)
    out = []
    for _ in range(count):
        ep = rng.choice(episodes)
        out.append((ep, rng.random()))
    return out


def _sampling_randomness(samples: Sequence[tuple[int, float]], *, batch_size: int) -> dict[str, float]:
    if not samples:
        return {
            "sample_count": 0.0,
            "unique_episodes": 0.0,
            "unique_episode_fraction": 0.0,
            "mean_samples_per_used_episode": 0.0,
            "max_samples_per_episode": 0.0,
            "mean_unique_episodes_per_batch": 0.0,
            "min_unique_episodes_per_batch": 0.0,
        }
    counts: dict[int, int] = {}
    for ep, _ts in samples:
        counts[ep] = counts.get(ep, 0) + 1
    batch_uniques = [
        len({ep for ep, _ts in samples[idx : idx + batch_size]})
        for idx in range(0, len(samples), batch_size)
        if samples[idx : idx + batch_size]
    ]
    return {
        "sample_count": float(len(samples)),
        "unique_episodes": float(len(counts)),
        "unique_episode_fraction": len(counts) / len(samples),
        "mean_samples_per_used_episode": len(samples) / len(counts),
        "max_samples_per_episode": float(max(counts.values())),
        "mean_unique_episodes_per_batch": float(np.mean(batch_uniques)),
        "min_unique_episodes_per_batch": float(min(batch_uniques)),
    }


def _decode_all(
    cache: EpisodeByteCache, timestamps: dict[tuple[int, str], list[float]], *, decode_workers: int
) -> float:
    start = time.perf_counter()
    items = list(timestamps.items())
    if decode_workers <= 1:
        for (ep, camera_key), ts in items:
            cache.get_frames(ep, camera_key, ts)
    else:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            futures = [pool.submit(cache.get_frames, ep, camera_key, ts) for (ep, camera_key), ts in items]
            for future in futures:
                future.result()
    return time.perf_counter() - start


def _decoder_locks(
    manifest: EpisodeVideoManifest, episodes: Sequence[int]
) -> dict[tuple[int, str], threading.Lock]:
    return {(ep, camera_key): threading.Lock() for ep in episodes for camera_key in manifest.video_keys}


def _open_resident_decoders(
    cache: EpisodeByteCache, episodes: Sequence[int], *, decode_workers: int
) -> tuple[float, int]:
    items = [(ep, camera_key) for ep in episodes for camera_key in cache.manifest.video_keys]
    start = time.perf_counter()
    if decode_workers <= 1:
        for ep, camera_key in items:
            cache.get_decoder(ep, camera_key)
    else:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            futures = [pool.submit(cache.get_decoder, ep, camera_key) for ep, camera_key in items]
            for future in futures:
                future.result()
    return time.perf_counter() - start, len(items)


def _decode_training_sample(
    cache: EpisodeByteCache,
    episode_index: int,
    relative_t: float,
    locks: dict[tuple[int, str], threading.Lock],
) -> None:
    for camera_key in cache.manifest.video_keys:
        span = cache.manifest.lookup(episode_index, camera_key)
        timestamp = span.first_pts + relative_t * max(span.last_pts - span.first_pts, 0.0)
        with locks[(episode_index, camera_key)]:
            cache.get_frames(episode_index, camera_key, [timestamp])


def run_pool_random_decode(
    cache: EpisodeByteCache,
    episodes: Sequence[int],
    *,
    sample_count: int,
    batch_size: int,
    decode_workers: int,
    seed: int,
) -> dict[str, float]:
    samples = _random_training_samples(episodes, sample_count, seed)
    touched_episodes = sorted({ep for ep, _ts in samples})
    decoder_open_s, decoder_count = _open_resident_decoders(
        cache, touched_episodes, decode_workers=decode_workers
    )
    locks = _decoder_locks(cache.manifest, touched_episodes)

    start = time.perf_counter()
    if decode_workers <= 1:
        for ep, ts in samples:
            _decode_training_sample(cache, ep, ts, locks)
    else:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            futures = [pool.submit(_decode_training_sample, cache, ep, ts, locks) for ep, ts in samples]
            for future in futures:
                future.result()
    decode_s = time.perf_counter() - start

    randomness = _sampling_randomness(samples, batch_size=batch_size)
    camera_frames = sample_count * len(cache.manifest.video_keys)
    result = {
        "decoder_open_s": decoder_open_s,
        "decoder_count": float(decoder_count),
        "decoder_open_ms": decoder_open_s * 1000 / max(decoder_count, 1),
        "decode_s": decode_s,
        "training_samples_s": sample_count / decode_s if decode_s > 0 else float("inf"),
        "camera_frames_s": camera_frames / decode_s if decode_s > 0 else float("inf"),
        "decode_ms_sample": decode_s * 1000 / max(sample_count, 1),
        "decode_ms_camera_frame": decode_s * 1000 / max(camera_frames, 1),
    }
    result.update(randomness)
    return result


def run_pool_stream_simulation(
    cache: EpisodeByteCache,
    resident_episodes: Sequence[int],
    *,
    dataset_episode_count: int,
    num_episodes: int,
    sample_count: int,
    target_samples_s: float,
    samples_per_episode: int,
    prefetch_episodes: int,
    batch_size: int,
    decode_workers: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    upper = min(dataset_episode_count, num_episodes)
    resident = list(resident_episodes)
    resident_set = set(resident)
    candidates = [ep for ep in range(upper) if ep not in resident_set]
    rng.shuffle(candidates)
    replacements = iter(candidates)
    pending: list[int] = []

    def schedule_one() -> bool:
        try:
            ep = next(replacements)
        except StopIteration:
            return False
        cache.submit_prefetch(ep)
        pending.append(ep)
        return True

    for _ in range(prefetch_episodes):
        if not schedule_one():
            break

    locks = _decoder_locks(cache.manifest, resident)
    sample_period = 1.0 / target_samples_s if target_samples_s > 0 else 0.0
    refill_wait_s = 0.0
    deadline_miss_s = 0.0
    replacement_count = 0
    decoded_samples: list[tuple[int, float]] = []
    start = time.perf_counter()
    next_deadline = start + sample_period

    for idx in range(sample_count):
        if idx > 0 and samples_per_episode > 0 and idx % samples_per_episode == 0 and pending:
            new_ep = pending.pop(0)
            wait_start = time.perf_counter()
            cache.ensure_ready(new_ep)
            for camera_key in cache.manifest.video_keys:
                locks[(new_ep, camera_key)] = threading.Lock()
                cache.get_decoder(new_ep, camera_key)
            refill_wait_s += time.perf_counter() - wait_start
            old_ep = resident.pop(0)
            resident_set.discard(old_ep)
            resident.append(new_ep)
            resident_set.add(new_ep)
            replacement_count += 1
            schedule_one()

        ep = rng.choice(resident)
        relative_t = rng.random()
        _decode_training_sample(cache, ep, relative_t, locks)
        decoded_samples.append((ep, relative_t))

        if sample_period > 0:
            now = time.perf_counter()
            if now < next_deadline:
                time.sleep(next_deadline - now)
            else:
                deadline_miss_s += now - next_deadline
            next_deadline += sample_period

    elapsed = time.perf_counter() - start
    result = {
        "target_samples_s": target_samples_s,
        "actual_samples_s": sample_count / elapsed if elapsed > 0 else float("inf"),
        "stream_wall_s": elapsed,
        "refill_wait_s": refill_wait_s,
        "deadline_miss_s": deadline_miss_s,
        "replacements": float(replacement_count),
        "replacement_episodes_s": replacement_count / elapsed if elapsed > 0 else 0.0,
        "samples_per_episode": float(samples_per_episode),
        "prefetch_episodes": float(prefetch_episodes),
        "kept_up": 1.0
        if sample_count / elapsed >= target_samples_s * 0.98 and deadline_miss_s < elapsed * 0.02
        else 0.0,
    }
    result.update(
        {
            f"stream_{key}": value
            for key, value in _sampling_randomness(decoded_samples, batch_size=batch_size).items()
        }
    )
    return result


def _fill_cache(
    cache: EpisodeByteCache, episodes: Sequence[int], *, progress_interval: float = 10.0
) -> float:
    start = time.perf_counter()
    for ep in episodes:
        cache.submit_prefetch(ep)
    last_progress = start
    for idx, ep in enumerate(episodes, start=1):
        cache.ensure_ready(ep)
        now = time.perf_counter()
        if progress_interval > 0 and now - last_progress >= progress_interval:
            timings = cache.timing_summary()
            byte_count = timings.get("range_bytes", 0.0)
            elapsed = max(now - start, 1e-9)
            jobs = timings.get("jobs", 0.0)
            total_jobs = len(episodes) * len(cache.manifest.video_keys)
            _log(
                "fill_progress: "
                f"episodes_ready={idx}/{len(episodes)} "
                f"camera_jobs={jobs:.0f}/{total_jobs} "
                f"fetched={byte_count / 1024**3:.2f} GiB "
                f"fetch={byte_count / elapsed / 1024**2:.1f} MiB/s "
                f"elapsed={_format_duration(elapsed)}"
            )
            last_progress = now
    return time.perf_counter() - start


def _samples_per_s(elapsed_s: float, episodes: Sequence[int], frames_per_episode: int) -> float:
    if elapsed_s <= 0:
        return float("inf")
    return len(episodes) * frames_per_episode / elapsed_s


def _log(message: str) -> None:
    print(message, flush=True)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _current_rss_mib() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024
    return None


def _peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    if rss > 10**8:
        return rss / 1024**2
    return rss / 1024


def _memory_snapshot() -> dict[str, float | None]:
    return {"rss_mib": _current_rss_mib(), "peak_rss_mib": _peak_rss_mib()}


def _print_memory_summary(start: dict[str, float | None], end: dict[str, float | None]) -> None:
    start_rss = start["rss_mib"]
    end_rss = end["rss_mib"]
    delta = None if start_rss is None or end_rss is None else end_rss - start_rss
    print()
    print("| Memory | MiB |")
    print("|---|---:|")
    if start_rss is not None:
        print(f"| rss start | {start_rss:.1f} |")
    if end_rss is not None:
        print(f"| rss end | {end_rss:.1f} |")
    if delta is not None:
        print(f"| rss delta | {delta:.1f} |")
    print(f"| peak rss | {end['peak_rss_mib']:.1f} |")


def _root_join(data_root: str, relative_path: str) -> str:
    if data_root.startswith("hf://"):
        return f"{data_root.rstrip('/')}/{relative_path}"
    return str(Path(data_root) / relative_path)


def _find_or_download_sidecar(data_root: str, manifest_episode_count: int) -> Path | None:
    _ = manifest_episode_count
    local = SIDECAR_CACHE_DIR / FULL_SIDECAR_NAME
    if _valid_sidecar(local):
        return local
    if local.exists():
        print(f"mp4_sidecar_invalid_local: {local}")
        local.unlink()
    remote_relative = f"meta/mp4-sidecars/{FULL_SIDECAR_NAME}"
    remote = _root_join(data_root, remote_relative)
    protocol = "hf" if data_root.startswith("hf://") else "file"
    fs = fsspec.filesystem(protocol)
    if not fs.exists(remote):
        return None
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading_mp4_sidecar: {remote} -> {local}")
    if data_root.startswith("hf://"):
        _download_sidecar_native_http(data_root, remote_relative, local)
    else:
        fs.get(remote, str(local))
    return local


def _valid_sidecar(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return "manifest_json" in data
    except Exception:
        return False


def _download_sidecar_native_http(data_root: str, relative_path: str, local: Path) -> None:
    fetcher = NativeHTTPRangeFetcher(data_root, max_connections=16)
    tmp = local.with_suffix(local.suffix + ".tmp")
    try:
        size = fetcher.info_size(relative_path)
        chunk_size = 16 * 1024 * 1024
        ranges = [(offset, min(chunk_size, size - offset)) for offset in range(0, size, chunk_size)]
        with tmp.open("wb") as out_file:
            out_file.truncate(size)

        def read_chunk(offset_length: tuple[int, int]) -> tuple[int, bytes]:
            offset, length = offset_length
            return offset, fetcher.read_range(relative_path, offset, length)

        start = time.perf_counter()
        done = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(read_chunk, item) for item in ranges]
            with tmp.open("r+b") as rw_file:
                for future in futures:
                    offset, data = future.result()
                    rw_file.seek(offset)
                    rw_file.write(data)
                    done += len(data)
                    elapsed = max(time.perf_counter() - start, 1e-9)
                    print(
                        f"sidecar_download: {done / 1024**2:.1f}/{size / 1024**2:.1f} MiB "
                        f"({done / elapsed / 1024**2:.1f} MiB/s)",
                        flush=True,
                    )
        tmp.replace(local)
    finally:
        fetcher.close()


class EpisodeParquetReader:
    def __init__(self, meta: LeRobotDatasetMetadata, data_root: str):
        self.meta = meta
        self.data_root = data_root
        protocol = "hf" if data_root.startswith("hf://") else "file"
        self.fs = fsspec.filesystem(protocol)
        self._episode_row_groups = self._build_episode_row_groups()
        self._table_cache: dict[str, pa.Table] = {}
        self._cache_lock = threading.Lock()

    def read_episode(self, episode_index: int) -> None:
        relative_path = str(self.meta.get_data_file_path(episode_index))
        table = self._read_table(relative_path)
        table.filter(pc.equal(table["episode_index"], episode_index))

    def _read_table(self, relative_path: str) -> pa.Table:
        with self._cache_lock:
            table = self._table_cache.get(relative_path)
        if table is not None:
            return table
        with self.fs.open(
            _root_join(self.data_root, relative_path), "rb", block_size=2**20, cache_type="none"
        ) as f:
            table = pq.ParquetFile(f).read()
        with self._cache_lock:
            return self._table_cache.setdefault(relative_path, table)

    def submit_read_episode(self, pool: ThreadPoolExecutor, episode_index: int):
        return pool.submit(self.read_episode, episode_index)

    def read_episodes(self, episodes: Sequence[int], *, workers: int) -> float:
        start = time.perf_counter()
        if workers <= 1:
            for ep in episodes:
                self.read_episode(ep)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self.read_episode, ep) for ep in episodes]
                for future in futures:
                    future.result()
        return time.perf_counter() - start

    def _build_episode_row_groups(self) -> dict[int, int]:
        counts: dict[tuple[int, int], int] = {}
        row_groups = {}
        for ep_idx in range(int(self.meta.total_episodes)):
            ep = self.meta.episodes[ep_idx]
            key = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
            row_groups[ep_idx] = counts.get(key, 0)
            counts[key] = row_groups[ep_idx] + 1
        return row_groups


def run_fetch_pool(
    manifest: EpisodeVideoManifest,
    data_root: str,
    episodes: Sequence[int],
    dataset_episode_count: int,
    benchmark_episode_count: int,
    byte_budget: int,
    workers: int,
    range_backend: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    with EpisodeByteCache(
        manifest,
        data_root,
        byte_budget=byte_budget,
        workers=workers,
        range_backend=range_backend,
        native_http_connections=args.native_http_connections,
        native_http_timeout=args.native_http_timeout,
        native_http_retries=args.native_http_retries,
        open_decoders=False,
    ) as cache:
        elapsed = _fill_cache(cache, episodes, progress_interval=args.progress_interval)
        timings = cache.timing_summary()
        random_decode = None
        stream_sim = None
        if args.include_pool_sampling:
            _log("pool_sampling: warming resident decoders and decoding random samples")
            random_decode = run_pool_random_decode(
                cache,
                episodes,
                sample_count=args.pool_random_samples,
                batch_size=args.batch_size,
                decode_workers=args.decode_workers,
                seed=args.seed + 3,
            )
            _log(
                f"pool_stream: consuming {args.target_samples_s:.1f} samples/s while prefetching replacements"
            )
            stream_sim = run_pool_stream_simulation(
                cache,
                episodes,
                dataset_episode_count=dataset_episode_count,
                num_episodes=benchmark_episode_count,
                sample_count=args.stream_samples,
                target_samples_s=args.target_samples_s,
                samples_per_episode=args.pool_samples_per_episode,
                prefetch_episodes=args.stream_prefetch_episodes,
                batch_size=args.batch_size,
                decode_workers=args.decode_workers,
                seed=args.seed + 4,
            )
    byte_count = _bytes_for(manifest, episodes)
    episode_mb = byte_count / len(episodes) / 1024**2
    job_count = max(timings["jobs"], 1.0)
    result = {
        "fetch_s": elapsed,
        "fetch_mbps": byte_count / elapsed / 1024**2,
        "fetch_episodes_s": len(episodes) / elapsed,
        "episode_mb": episode_mb,
        "avg_mb_miss": byte_count / (len(episodes) * len(manifest.video_keys)) / 1024**2,
        "jobs": timings["jobs"],
        "lookup_ms": timings["lookup_s"] * 1000 / job_count,
        "range_fetch_ms": timings["fetch_s"] * 1000 / job_count,
        "synthesize_ms": timings["synthesize_s"] * 1000 / job_count,
        "store_ms": timings["store_s"] * 1000 / job_count,
    }
    result.update({key: value for key, value in timings.items() if key.startswith("range_")})
    if random_decode is not None:
        result.update({f"pool_decode_{key}": value for key, value in random_decode.items()})
    if stream_sim is not None:
        result.update({f"pool_stream_{key}": value for key, value in stream_sim.items()})
    return result


def run_parallel(
    manifest: EpisodeVideoManifest,
    data_root: str,
    episodes: Sequence[int],
    timestamps: dict[tuple[int, str], list[float]],
    byte_budget: int,
    workers: int,
    decode_workers: int,
    frames_per_episode: int,
    parquet_reader: EpisodeParquetReader,
    range_backend: str,
) -> dict[str, float]:
    with EpisodeByteCache(
        manifest,
        data_root,
        byte_budget=byte_budget,
        workers=workers,
        range_backend=range_backend,
        open_decoders=False,
    ) as cache:
        parquet_s = parquet_reader.read_episodes(episodes, workers=workers)
        fetch_s = _fill_cache(cache, episodes)
        decoder_start = time.perf_counter()
        for ep in episodes:
            for camera_key in manifest.video_keys:
                cache.get_decoder(ep, camera_key)
        decoder_s = time.perf_counter() - decoder_start
        decode_s = _decode_all(cache, timestamps, decode_workers=decode_workers)
    byte_count = _bytes_for(manifest, episodes)
    return {
        "fetch_s": fetch_s,
        "fetch_mbps": byte_count / fetch_s / 1024**2,
        "fetch_episodes_s": len(episodes) / fetch_s,
        "parquet_s": parquet_s,
        "decoder_ms_miss": decoder_s * 1000 / (len(episodes) * len(manifest.video_keys)),
        "decode_samples_s": _samples_per_s(decode_s, episodes, frames_per_episode),
    }


def run_overlapped(
    manifest: EpisodeVideoManifest,
    data_root: str,
    episodes: Sequence[int],
    timestamps: dict[tuple[int, str], list[float]],
    byte_budget: int,
    workers: int,
    decode_workers: int,
    frames_per_episode: int,
    prefetch_ahead: int,
    parquet_reader: EpisodeParquetReader,
    range_backend: str,
) -> dict[str, float]:
    with EpisodeByteCache(
        manifest,
        data_root,
        byte_budget=byte_budget,
        workers=workers,
        range_backend=range_backend,
        open_decoders=True,
    ) as cache:
        start = time.perf_counter()
        video_wait_decode_s = 0.0
        parquet_wait_s = 0.0
        parquet_pool = ThreadPoolExecutor(max_workers=max(1, min(workers, len(episodes))))
        parquet_futures = {
            ep: parquet_reader.submit_read_episode(parquet_pool, ep) for ep in episodes[:prefetch_ahead]
        }
        for ep in episodes[:prefetch_ahead]:
            cache.submit_prefetch(ep)
        try:
            for idx, ep in enumerate(episodes):
                next_idx = idx + prefetch_ahead
                if next_idx < len(episodes):
                    next_ep = episodes[next_idx]
                    cache.submit_prefetch(next_ep)
                    parquet_futures[next_ep] = parquet_reader.submit_read_episode(parquet_pool, next_ep)

                parquet_start = time.perf_counter()
                parquet_futures.pop(ep).result()
                parquet_wait_s += time.perf_counter() - parquet_start

                video_start = time.perf_counter()
                cache.ensure_ready(ep)
                if decode_workers <= 1:
                    for camera_key in manifest.video_keys:
                        cache.get_frames(ep, camera_key, timestamps[(ep, camera_key)])
                else:
                    with ThreadPoolExecutor(max_workers=decode_workers) as pool:
                        futures = [
                            pool.submit(cache.get_frames, ep, camera_key, timestamps[(ep, camera_key)])
                            for camera_key in manifest.video_keys
                        ]
                        for future in futures:
                            future.result()
                video_wait_decode_s += time.perf_counter() - video_start
        finally:
            parquet_pool.shutdown(wait=True)
        elapsed = time.perf_counter() - start
    return {
        "samples_s": _samples_per_s(elapsed, episodes, frames_per_episode),
        "video_samples_s": _samples_per_s(video_wait_decode_s, episodes, frames_per_episode),
        "parquet_samples_s": _samples_per_s(parquet_wait_s, episodes, frames_per_episode),
        "wall_s": elapsed,
        "video_wait_decode_s": video_wait_decode_s,
        "parquet_wait_s": parquet_wait_s,
    }


_remote_decoder_local = threading.local()


def _remote_decoder_cache() -> VideoDecoderCache:
    cache = getattr(_remote_decoder_local, "cache", None)
    if cache is None:
        cache = VideoDecoderCache(max_size=None)
        _remote_decoder_local.cache = cache
    return cache


def _decode_remote_source(
    meta: LeRobotDatasetMetadata,
    data_root: str,
    episode_index: int,
    camera_key: str,
    timestamps: list[float],
):
    video_path = _root_join(data_root, str(meta.get_video_file_path(episode_index, camera_key)))
    return decode_video_frames_torchcodec(
        video_path,
        timestamps,
        tolerance_s=1.0 / float(meta.fps),
        decoder_cache=_remote_decoder_cache(),
        return_uint8=True,
    )


def run_remote_decoder(
    meta: LeRobotDatasetMetadata,
    data_root: str,
    episodes: Sequence[int],
    timestamps: dict[tuple[int, str], list[float]],
    *,
    frames_per_episode: int,
    decode_workers: int,
    parquet_reader: EpisodeParquetReader,
) -> dict[str, float]:
    items = [
        (ep, camera_key, timestamps[(ep, camera_key)]) for ep in episodes for camera_key in meta.video_keys
    ]

    start = time.perf_counter()
    for ep, camera_key, ts in items:
        if camera_key == meta.video_keys[0]:
            parquet_reader.read_episode(ep)
        _decode_remote_source(meta, data_root, ep, camera_key, ts)
    sequential_s = time.perf_counter() - start

    start = time.perf_counter()
    if decode_workers <= 1:
        for ep, camera_key, ts in items:
            if camera_key == meta.video_keys[0]:
                parquet_reader.read_episode(ep)
            _decode_remote_source(meta, data_root, ep, camera_key, ts)
    else:
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            parquet_futures = [pool.submit(parquet_reader.read_episode, ep) for ep in episodes]
            futures = [
                pool.submit(_decode_remote_source, meta, data_root, ep, camera_key, ts)
                for ep, camera_key, ts in items
            ]
            for future in parquet_futures:
                future.result()
            for future in futures:
                future.result()
    parallel_s = time.perf_counter() - start

    return {
        "sequential_samples_s": _samples_per_s(sequential_s, episodes, frames_per_episode),
        "parallel_samples_s": _samples_per_s(parallel_s, episodes, frames_per_episode),
    }


def _print_range_timing_summary(fetch_pool: dict[str, float]) -> None:
    range_jobs = fetch_pool.get("range_jobs", 0.0)
    if range_jobs <= 0:
        return

    print()
    print("| Range Read Stage | avg ms/range |")
    print("|---|---:|")
    for key, label in (
        ("range_open_s", "fsspec handle open/lookup"),
        ("range_seek_s", "fsspec seek"),
        ("range_read_s", "fsspec read"),
        ("range_resolve_s", "http URL resolve"),
        ("range_header_s", "http response headers"),
        ("range_first_byte_s", "http first body byte"),
        ("range_body_s", "http body drain"),
        ("range_chunk_gap_s", "http chunk wait"),
        ("range_join_s", "join response chunks"),
        ("range_failed_attempt_s", "http failed attempts"),
        ("range_retry_sleep_s", "http retry sleep"),
    ):
        value = fetch_pool.get(key)
        if value is not None:
            print(f"| {label} | {value * 1000 / range_jobs:.3f} |")
    if "range_retry_attempts" in fetch_pool:
        print(f"| http retries | {fetch_pool['range_retry_attempts'] / range_jobs:.3f} |")
    if "range_exception_attempts" in fetch_pool:
        print(f"| http exceptions | {fetch_pool['range_exception_attempts'] / range_jobs:.3f} |")
        if fetch_pool["range_exception_attempts"] > 0:
            print(
                f"| http failed attempt avg s | "
                f"{fetch_pool.get('range_failed_attempt_s', 0.0) / fetch_pool['range_exception_attempts']:.1f} |"
            )
    if fetch_pool.get("range_failed_requests"):
        print(f"| http failed requests | {fetch_pool['range_failed_requests']:.0f} |")
    exception_counts = {
        key.removeprefix("range_exception_"): value
        for key, value in fetch_pool.items()
        if key.startswith("range_exception_") and key != "range_exception_attempts"
    }
    if exception_counts:
        summary = ", ".join(f"{name}={count:.0f}" for name, count in sorted(exception_counts.items()))
        print(f"| http exception counts | {summary} |")
    failed_status_counts = {
        key.removeprefix("range_failed_status_"): value
        for key, value in fetch_pool.items()
        if key.startswith("range_failed_status_")
    }
    if failed_status_counts:
        summary = ", ".join(f"{status}={count:.0f}" for status, count in sorted(failed_status_counts.items()))
        print(f"| http failed status counts | {summary} |")
    status_counts = {
        key.removeprefix("range_status_"): value
        for key, value in fetch_pool.items()
        if key.startswith("range_status_")
    }
    if status_counts:
        summary = ", ".join(f"{status}={count:.0f}" for status, count in sorted(status_counts.items()))
        print(f"| http status counts | {summary} |")
    for method in ("head", "get"):
        request_count = fetch_pool.get(f"range_hffs_{method}_requests", 0.0)
        if request_count <= 0:
            continue
        print(f"| hffs {method.upper()} requests/range | {request_count / range_jobs:.3f} |")
        print(
            f"| hffs {method.upper()} total | {fetch_pool[f'range_hffs_{method}_s'] * 1000 / range_jobs:.3f} |"
        )
        retries = fetch_pool.get(f"range_hffs_{method}_retries", 0.0)
        exceptions = fetch_pool.get(f"range_hffs_{method}_exception_attempts", 0.0)
        if retries:
            print(f"| hffs {method.upper()} retries/range | {retries / range_jobs:.3f} |")
            print(
                f"| hffs {method.upper()} retry sleep | "
                f"{fetch_pool.get(f'range_hffs_{method}_retry_sleep_s', 0.0) * 1000 / range_jobs:.3f} |"
            )
        if exceptions:
            print(f"| hffs {method.upper()} exceptions/range | {exceptions / range_jobs:.3f} |")
            print(
                f"| hffs {method.upper()} failed attempts | "
                f"{fetch_pool.get(f'range_hffs_{method}_failed_attempt_s', 0.0) * 1000 / range_jobs:.3f} |"
            )
        bytes_read = fetch_pool.get(f"range_hffs_{method}_bytes", 0.0)
        total_s = fetch_pool.get(f"range_hffs_{method}_s", 0.0)
        if bytes_read > 0 and total_s > 0:
            print(f"| hffs {method.upper()} MiB/s | {bytes_read / total_s / 1024**2:.1f} |")
        hffs_status_counts = {
            key.removeprefix(f"range_hffs_{method}_status_"): value
            for key, value in fetch_pool.items()
            if key.startswith(f"range_hffs_{method}_status_")
        }
        if hffs_status_counts:
            summary = ", ".join(
                f"{status}={count:.0f}" for status, count in sorted(hffs_status_counts.items())
            )
            print(f"| hffs {method.upper()} status counts | {summary} |")
        hffs_failed_status_counts = {
            key.removeprefix(f"range_hffs_{method}_failed_status_"): value
            for key, value in fetch_pool.items()
            if key.startswith(f"range_hffs_{method}_failed_status_")
        }
        if hffs_failed_status_counts:
            summary = ", ".join(
                f"{status}={count:.0f}" for status, count in sorted(hffs_failed_status_counts.items())
            )
            print(f"| hffs {method.upper()} failed status counts | {summary} |")
        hffs_exception_counts = {
            key.removeprefix(f"range_hffs_{method}_exception_"): value
            for key, value in fetch_pool.items()
            if key.startswith(f"range_hffs_{method}_exception_")
            and key != f"range_hffs_{method}_exception_attempts"
        }
        if hffs_exception_counts:
            summary = ", ".join(
                f"{name}={count:.0f}" for name, count in sorted(hffs_exception_counts.items())
            )
            print(f"| hffs {method.upper()} exception counts | {summary} |")
    chunks = fetch_pool.get("range_chunks", 0.0)
    if chunks > 0:
        bytes_read = fetch_pool.get("range_bytes", 0.0)
        body_s = fetch_pool.get("range_body_s", 0.0)
        print(f"| http chunks/range | {chunks / range_jobs:.1f} |")
        print(f"| http avg KiB/chunk | {bytes_read / chunks / 1024:.1f} |")
        if body_s > 0:
            print(f"| http body MiB/s | {bytes_read / body_s / 1024**2:.1f} |")
    print(f"| range reads | {range_jobs:.0f} |")
    print(f"| avg MiB/range | {fetch_pool.get('range_bytes', 0.0) / range_jobs / 1024**2:.1f} |")


def run_indexed_strategy(
    meta: LeRobotDatasetMetadata,
    data_root: str,
    args: argparse.Namespace,
    parquet_reader: EpisodeParquetReader,
    *,
    range_backend: str = "fsspec",
    label: str = "indexed",
    sidecar_path: str | None = None,
) -> None:
    _log(f"starting_strategy: {label}")
    memory_start = _memory_snapshot()
    manifest_start = time.perf_counter()
    dataset_episode_count = int(meta.total_episodes)
    manifest_episode_count = args.manifest_episodes or dataset_episode_count
    manifest_episode_count = min(manifest_episode_count, dataset_episode_count, args.num_episodes)
    manifest = EpisodeVideoManifest.build(
        meta,
        data_root,
        episode_indices=range(manifest_episode_count),
        range_backend=range_backend,
        workers=args.workers,
        max_probe_bytes=args.max_probe_mb * 1024 * 1024,
        sidecar_path=sidecar_path,
    )
    manifest_s = time.perf_counter() - manifest_start
    _log(f"{label}: manifest_build_s={manifest_s:.2f}")

    benchmark_episode_count = min(dataset_episode_count, args.num_episodes)
    episodes = _episode_pool(dataset_episode_count, args.num_episodes, args.pool_size, args.seed)
    byte_budget = int(args.byte_budget_gb * 1024**3)
    byte_count = _bytes_for(manifest, episodes)
    _log(
        f"{label}: planned_video_fetch={byte_count / 1024**3:.2f} GiB per fetch track "
        f"({byte_count / len(episodes) / 1024**2:.1f} MiB/episode)"
    )

    _log(f"{label}: filling episode byte cache with {args.workers} workers")
    fetch_pool = run_fetch_pool(
        manifest,
        data_root,
        episodes,
        dataset_episode_count,
        benchmark_episode_count,
        byte_budget,
        args.workers,
        range_backend,
        args,
    )
    estimated_dataset_s = dataset_episode_count / fetch_pool["fetch_episodes_s"]
    estimated_benchmark_s = benchmark_episode_count / fetch_pool["fetch_episodes_s"]

    print(f"manifest_build_s: {manifest_s:.2f}")
    print(f"strategy: {label}")
    print(f"range_backend: {range_backend}")
    print(f"mp4_sidecar: {sidecar_path or 'none'}")
    print(f"data_root: {data_root}")
    print(f"dataset_episodes: {dataset_episode_count}")
    print(f"benchmark_episodes: {benchmark_episode_count}")
    print(f"pool_episodes: {len(episodes)}")
    print(f"sampled_episodes: {episodes}")
    print(f"cameras: {manifest.video_keys}")
    print()
    print(
        "| Track | fetch MB/s | fetch eps/s | wall s | est benchmark | est full dataset | avg MB/camera | notes |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    print(
        f"| EPISODE POOL FETCH | {fetch_pool['fetch_mbps']:.1f} | "
        f"{fetch_pool['fetch_episodes_s']:.2f} | {fetch_pool['fetch_s']:.2f} | "
        f"{_format_duration(estimated_benchmark_s)} | {_format_duration(estimated_dataset_s)} | "
        f"{fetch_pool['avg_mb_miss']:.1f} | {args.workers} workers, no decoder open/frame decode |"
    )
    print()
    print("| Camera Job Stage | avg ms/job |")
    print("|---|---:|")
    print(f"| manifest lookup | {fetch_pool['lookup_ms']:.3f} |")
    print(f"| remote byte-range fetch | {fetch_pool['range_fetch_ms']:.3f} |")
    print(f"| synthesize mini-MP4 | {fetch_pool['synthesize_ms']:.3f} |")
    print(f"| store in shared cache | {fetch_pool['store_ms']:.3f} |")
    print(f"| camera jobs | {fetch_pool['jobs']:.0f} |")
    _print_range_timing_summary(fetch_pool)
    if args.include_pool_sampling:
        print()
        print("| Resident Pool Decode | value |")
        print("|---|---:|")
        print(f"| random training samples | {fetch_pool['pool_decode_sample_count']:.0f} |")
        print(f"| decoder opens | {fetch_pool['pool_decode_decoder_count']:.0f} |")
        print(f"| decoder open ms/episode-camera | {fetch_pool['pool_decode_decoder_open_ms']:.3f} |")
        print(f"| decode wall s | {fetch_pool['pool_decode_decode_s']:.3f} |")
        print(f"| training samples/s | {fetch_pool['pool_decode_training_samples_s']:.1f} |")
        print(f"| camera frames/s | {fetch_pool['pool_decode_camera_frames_s']:.1f} |")
        print(f"| decode ms/training sample | {fetch_pool['pool_decode_decode_ms_sample']:.3f} |")
        print(f"| decode ms/camera frame | {fetch_pool['pool_decode_decode_ms_camera_frame']:.3f} |")
        print()
        print("| Resident Pool Randomness | value |")
        print("|---|---:|")
        print(f"| pool episodes | {len(episodes)} |")
        print(f"| batch size | {args.batch_size} |")
        print(f"| unique episodes sampled | {fetch_pool['pool_decode_unique_episodes']:.0f} |")
        print(
            f"| mean unique episodes/batch | {fetch_pool['pool_decode_mean_unique_episodes_per_batch']:.1f} |"
        )
        print(
            f"| min unique episodes/batch | {fetch_pool['pool_decode_min_unique_episodes_per_batch']:.0f} |"
        )
        print(
            f"| mean samples/used episode | {fetch_pool['pool_decode_mean_samples_per_used_episode']:.2f} |"
        )
        print(f"| max samples/episode | {fetch_pool['pool_decode_max_samples_per_episode']:.0f} |")
        print()
        print("| Streaming Keep-Up Simulation | value |")
        print("|---|---:|")
        print(f"| target samples/s | {fetch_pool['pool_stream_target_samples_s']:.1f} |")
        print(f"| actual samples/s | {fetch_pool['pool_stream_actual_samples_s']:.1f} |")
        print(f"| kept up | {'yes' if fetch_pool['pool_stream_kept_up'] else 'no'} |")
        print(f"| stream wall s | {fetch_pool['pool_stream_stream_wall_s']:.3f} |")
        print(f"| refill wait s | {fetch_pool['pool_stream_refill_wait_s']:.3f} |")
        print(f"| deadline miss s | {fetch_pool['pool_stream_deadline_miss_s']:.3f} |")
        print(f"| replacement episodes | {fetch_pool['pool_stream_replacements']:.0f} |")
        print(f"| replacement episodes/s | {fetch_pool['pool_stream_replacement_episodes_s']:.2f} |")
        print(f"| samples per replacement episode | {fetch_pool['pool_stream_samples_per_episode']:.0f} |")
        print(f"| prefetch replacement episodes | {fetch_pool['pool_stream_prefetch_episodes']:.0f} |")
        print(
            f"| stream mean unique episodes/batch | "
            f"{fetch_pool['pool_stream_stream_mean_unique_episodes_per_batch']:.1f} |"
        )
        print(
            f"| stream min unique episodes/batch | "
            f"{fetch_pool['pool_stream_stream_min_unique_episodes_per_batch']:.0f} |"
        )
    _print_memory_summary(memory_start, _memory_snapshot())

    if args.include_decode:
        timestamps = _timestamps(manifest, episodes, args.frames_per_episode, args.seed + 1)
        _log(f"{label}: running parallel video fetch + decode-only")
        parallel = run_parallel(
            manifest,
            data_root,
            episodes,
            timestamps,
            byte_budget,
            args.workers,
            args.decode_workers,
            args.frames_per_episode,
            parquet_reader,
            range_backend,
        )
        _log(f"{label}: running overlapped end-to-end")
        overlapped = run_overlapped(
            manifest,
            data_root,
            episodes,
            timestamps,
            byte_budget,
            args.workers,
            args.decode_workers,
            args.frames_per_episode,
            args.prefetch_ahead,
            parquet_reader,
            range_backend,
        )
        print(
            f"| DECODE COMPARISON | {parallel['fetch_mbps']:.1f} | {parallel['fetch_episodes_s']:.2f} | "
            f"{parallel['fetch_s']:.2f} | "
            f"{_format_duration(benchmark_episode_count / parallel['fetch_episodes_s'])} | "
            f"{_format_duration(dataset_episode_count / parallel['fetch_episodes_s'])} | "
            f"{fetch_pool['avg_mb_miss']:.1f} | "
            f"decoder open {parallel['decoder_ms_miss']:.1f} ms/miss, "
            f"decode {parallel['decode_samples_s']:.1f} samples/s, parquet {parallel['parquet_s']:.2f}s |"
        )
        print(
            f"| OVERLAPPED E2E | - | - | {overlapped['wall_s']:.2f} | - | - | "
            f"{fetch_pool['avg_mb_miss']:.1f} | "
            f"{overlapped['samples_s']:.1f} samples/s; video+decode "
            f"{overlapped['video_wait_decode_s']:.2f}s, parquet {overlapped['parquet_wait_s']:.2f}s |"
        )


def run_remote_strategy(
    meta: LeRobotDatasetMetadata,
    data_root: str,
    args: argparse.Namespace,
    parquet_reader: EpisodeParquetReader,
) -> None:
    _log("starting_strategy: remote-decoder")
    episodes = _episode_pool(int(meta.total_episodes), args.num_episodes, args.pool_size, args.seed)
    timestamps = _timestamps_from_meta(meta, episodes, args.frames_per_episode, args.seed + 1)
    _log("remote-decoder: running direct source MP4 decoder")
    result = run_remote_decoder(
        meta,
        data_root,
        episodes,
        timestamps,
        frames_per_episode=args.frames_per_episode,
        decode_workers=args.decode_workers,
        parquet_reader=parquet_reader,
    )
    print("strategy: remote-decoder")
    print(f"data_root: {data_root}")
    print(f"episodes: {episodes}")
    print(f"cameras: {list(meta.video_keys)}")
    print()
    print("| Track | samples/s | notes |")
    print("|---|---:|---|")
    print(f"| REMOTE SEQUENTIAL | {result['sequential_samples_s']:.1f} | direct source MP4 decoder |")
    print(
        f"| REMOTE PARALLEL | {result['parallel_samples_s']:.1f} | "
        f"direct source MP4 decoder, {args.decode_workers} workers |"
    )


def main() -> None:
    args = parse_args()
    if args.strategy == "full":
        args.strategy = "both"
    if args.strategy == "native-http":
        args.range_backend = "native-http"
    if args.http_failure_log:
        os.environ["LEROBOT_HTTP_FAILURE_LOG"] = args.http_failure_log
        print(f"http_failure_log: {args.http_failure_log}")
    data_root = args.data_root
    if data_root.startswith("hf://") and not args.no_hub_branch_assert:
        assert_hf_hub_range_cache_branch()

    meta = LeRobotDatasetMetadata(args.repo_id, revision=args.revision)
    meta.ensure_readable()
    parquet_reader = EpisodeParquetReader(meta, data_root)
    manifest_episode_count = args.manifest_episodes or int(meta.total_episodes)
    manifest_episode_count = min(manifest_episode_count, int(meta.total_episodes), args.num_episodes)
    sidecar_path = _find_or_download_sidecar(data_root, manifest_episode_count)

    if sidecar_path is not None:
        print(f"using_mp4_sidecar: {sidecar_path}")

    if sidecar_path is not None and args.strategy == "both":
        if args.include_decode:
            run_remote_strategy(meta, data_root, args, parquet_reader)
            print()
        run_indexed_strategy(
            meta,
            data_root,
            args,
            parquet_reader,
            range_backend=args.range_backend,
            label=f"indexed-sidecar-{args.range_backend}",
            sidecar_path=str(sidecar_path),
        )
        return
    if sidecar_path is not None and args.strategy == "indexed":
        run_indexed_strategy(
            meta,
            data_root,
            args,
            parquet_reader,
            range_backend=args.range_backend,
            label=f"indexed-sidecar-{args.range_backend}",
            sidecar_path=str(sidecar_path),
        )
        return
    if sidecar_path is not None and args.strategy == "native-http":
        run_indexed_strategy(
            meta,
            data_root,
            args,
            parquet_reader,
            range_backend="native-http",
            label="indexed-sidecar-native-http",
            sidecar_path=str(sidecar_path),
        )
        return
    if args.strategy == "both":
        expected_sidecar = SIDECAR_CACHE_DIR / FULL_SIDECAR_NAME
        expected_remote = _root_join(data_root, f"meta/mp4-sidecars/{FULL_SIDECAR_NAME}")
        print(f"mp4_sidecar_missing_local: {expected_sidecar}")
        print(f"mp4_sidecar_missing_remote: {expected_remote}")
        print(
            "build_mp4_sidecar: "
            "uv run --no-sync python scripts/build_mp4_sidecar.py "
            f"--workers {args.workers} --range-backend native-http --output {expected_sidecar}"
        )
        print("running_without_mp4_sidecar: indexed variants will build MP4 indexes online")
        print()

    if args.strategy in ("both", "indexed"):
        run_indexed_strategy(
            meta,
            data_root,
            args,
            parquet_reader,
            range_backend="fsspec",
            label="indexed",
            sidecar_path=None,
        )
    if args.strategy == "both":
        print()
    if args.strategy == "remote-decoder" or (args.strategy == "both" and args.include_decode):
        run_remote_strategy(meta, data_root, args, parquet_reader)
    if args.strategy == "both" and args.include_decode:
        print()
    if args.strategy in ("both", "native-http"):
        run_indexed_strategy(
            meta,
            data_root,
            args,
            parquet_reader,
            range_backend="native-http",
            label="indexed-native-http",
            sidecar_path=None,
        )


if __name__ == "__main__":
    main()
