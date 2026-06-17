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
import random
import tempfile
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
    make_range_fetcher,
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
        choices=("both", "full", "indexed", "remote-decoder", "native-http", "random-frames"),
        default="both",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--num-episodes", type=int, default=512)
    parser.add_argument(
        "--manifest-episodes",
        type=int,
        default=None,
        help="Limit manifest construction to the first N episodes for local smoke tests.",
    )
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument(
        "--frame-pool-size",
        type=int,
        default=4096,
        help="Number of random frame/camera targets for --strategy random-frames.",
    )
    parser.add_argument(
        "--coalesce-gap-kb",
        type=int,
        default=256,
        help="Merge random-frame byte windows separated by at most this many KiB.",
    )
    parser.add_argument(
        "--random-frame-backend",
        choices=("fsspec", "native-http"),
        default="native-http",
        help="Range backend for --strategy random-frames.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--include-decode",
        action="store_true",
        help="Also run decoder-opening/frame-decode comparison tracks. Fetch-only is the default.",
    )
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


@dataclass(frozen=True)
class FrameByteWindow:
    file_id: int
    file_path: str
    byte_offset: int
    byte_length: int
    useful_bytes: int
    sample_lo: int
    sample_hi: int
    target_sample: int


@dataclass(frozen=True)
class CoalescedByteRange:
    file_id: int
    file_path: str
    byte_offset: int
    byte_length: int
    windows: int
    useful_bytes: int


def _previous_sync_sample(sync_samples: np.ndarray, target_sample: int) -> int:
    prev = sync_samples[sync_samples <= target_sample]
    if len(prev):
        return int(prev[-1])
    if len(sync_samples):
        return int(sync_samples[0])
    return target_sample


def _frame_window_for_sample(
    manifest: EpisodeVideoManifest, episode_index: int, camera_key: str, ts: float
) -> FrameByteWindow:
    span = manifest.lookup(episode_index, camera_key)
    file_record = manifest.file_lookup(span.file_id)
    mp4 = file_record.mp4
    if len(mp4.sample_pts) == 0:
        raise ValueError(f"{file_record.file_path} has no indexed samples")
    target = int(np.searchsorted(mp4.sample_pts, ts, side="left"))
    target = min(max(target, 0), len(mp4.sample_pts) - 1)
    lo = _previous_sync_sample(mp4.sync_samples, target)
    hi = max(target, lo)
    offsets = mp4.sample_offsets[lo : hi + 1]
    sizes = mp4.sample_sizes[lo : hi + 1]
    byte_offset = int(offsets.min())
    byte_end = int((offsets + sizes).max())
    return FrameByteWindow(
        file_id=span.file_id,
        file_path=file_record.file_path,
        byte_offset=byte_offset,
        byte_length=byte_end - byte_offset,
        useful_bytes=int(sizes.sum()),
        sample_lo=lo,
        sample_hi=hi,
        target_sample=target,
    )


def _sample_frame_windows(
    manifest: EpisodeVideoManifest,
    *,
    benchmark_episode_count: int,
    frame_pool_size: int,
    seed: int,
) -> list[FrameByteWindow]:
    rng = random.Random(seed)
    windows = []
    for _ in range(frame_pool_size):
        ep = rng.randrange(benchmark_episode_count)
        camera_key = rng.choice(manifest.video_keys)
        span = manifest.lookup(ep, camera_key)
        ts = rng.uniform(span.first_pts, max(span.last_pts, span.first_pts))
        windows.append(_frame_window_for_sample(manifest, ep, camera_key, ts))
    return windows


def _coalesce_windows(windows: Sequence[FrameByteWindow], gap_bytes: int) -> list[CoalescedByteRange]:
    by_file: dict[int, list[FrameByteWindow]] = {}
    for window in windows:
        by_file.setdefault(window.file_id, []).append(window)

    ranges = []
    for file_id, file_windows in by_file.items():
        ordered = sorted(file_windows, key=lambda w: w.byte_offset)
        current_start = ordered[0].byte_offset
        current_end = ordered[0].byte_offset + ordered[0].byte_length
        current_path = ordered[0].file_path
        current_windows = 1
        current_useful_bytes = ordered[0].useful_bytes
        for window in ordered[1:]:
            start = window.byte_offset
            end = window.byte_offset + window.byte_length
            if start <= current_end + gap_bytes:
                current_end = max(current_end, end)
                current_windows += 1
                current_useful_bytes += window.useful_bytes
                continue
            ranges.append(
                CoalescedByteRange(
                    file_id=file_id,
                    file_path=current_path,
                    byte_offset=current_start,
                    byte_length=current_end - current_start,
                    windows=current_windows,
                    useful_bytes=current_useful_bytes,
                )
            )
            current_start = start
            current_end = end
            current_path = window.file_path
            current_windows = 1
            current_useful_bytes = window.useful_bytes
        ranges.append(
            CoalescedByteRange(
                file_id=file_id,
                file_path=current_path,
                byte_offset=current_start,
                byte_length=current_end - current_start,
                windows=current_windows,
                useful_bytes=current_useful_bytes,
            )
        )
    return ranges


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


def _fill_cache(cache: EpisodeByteCache, episodes: Sequence[int]) -> float:
    start = time.perf_counter()
    for ep in episodes:
        cache.submit_prefetch(ep)
    for ep in episodes:
        cache.ensure_ready(ep)
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
    byte_budget: int,
    workers: int,
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
        elapsed = _fill_cache(cache, episodes)
        timings = cache.timing_summary()
    byte_count = _bytes_for(manifest, episodes)
    episode_mb = byte_count / len(episodes) / 1024**2
    job_count = max(timings["jobs"], 1.0)
    return {
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
    fetch_pool = run_fetch_pool(manifest, data_root, episodes, byte_budget, args.workers, range_backend)
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


def run_random_frame_strategy(
    meta: LeRobotDatasetMetadata,
    data_root: str,
    args: argparse.Namespace,
    *,
    range_backend: str = "native-http",
    label: str = "random-frames",
    sidecar_path: str | None = None,
) -> None:
    if args.frame_pool_size <= 0:
        raise ValueError(f"frame-pool-size must be > 0, got {args.frame_pool_size}")
    _log(f"starting_strategy: {label}")
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
    window_start = time.perf_counter()
    windows = _sample_frame_windows(
        manifest,
        benchmark_episode_count=benchmark_episode_count,
        frame_pool_size=args.frame_pool_size,
        seed=args.seed,
    )
    window_s = time.perf_counter() - window_start
    raw_bytes = sum(window.byte_length for window in windows)
    useful_bytes = sum(window.useful_bytes for window in windows)
    avg_decode_samples = sum(window.sample_hi - window.sample_lo + 1 for window in windows) / len(windows)

    coalesce_start = time.perf_counter()
    coalesced = _coalesce_windows(windows, args.coalesce_gap_kb * 1024)
    coalesce_s = time.perf_counter() - coalesce_start
    coalesced_bytes = sum(item.byte_length for item in coalesced)

    _log(
        f"{label}: fetching {len(coalesced)} coalesced ranges for {len(windows)} random frame targets "
        f"({coalesced_bytes / 1024**2:.1f} MiB)"
    )
    fetcher = make_range_fetcher(data_root, range_backend=range_backend, workers=args.workers)

    def read_range(item: CoalescedByteRange) -> int:
        payload = fetcher.read_range(item.file_path, item.byte_offset, item.byte_length)
        if len(payload) != item.byte_length:
            raise OSError(f"Short read for {item.file_path}: expected {item.byte_length}, got {len(payload)}")
        return len(payload)

    fetch_start = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            fetched_bytes = sum(pool.map(read_range, coalesced))
    finally:
        fetcher.close()
    fetch_s = time.perf_counter() - fetch_start

    print(f"manifest_build_s: {manifest_s:.2f}")
    print(f"strategy: {label}")
    print(f"range_backend: {range_backend}")
    print(f"mp4_sidecar: {sidecar_path or 'none'}")
    print(f"data_root: {data_root}")
    print(f"dataset_episodes: {dataset_episode_count}")
    print(f"benchmark_episodes: {benchmark_episode_count}")
    print(f"frame_targets: {len(windows)}")
    print(f"cameras: {manifest.video_keys}")
    print(f"coalesce_gap_kb: {args.coalesce_gap_kb}")
    print()
    print(
        "| Track | fetch MB/s | frame targets/s | wall s | raw MiB | coalesced MiB | "
        "ranges | avg KiB/range | avg KiB/frame | notes |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    print(
        f"| RANDOM FRAME WINDOWS | {fetched_bytes / fetch_s / 1024**2:.1f} | "
        f"{len(windows) / fetch_s:.1f} | {fetch_s:.2f} | {raw_bytes / 1024**2:.1f} | "
        f"{coalesced_bytes / 1024**2:.1f} | {len(coalesced)} | "
        f"{coalesced_bytes / max(len(coalesced), 1) / 1024:.1f} | "
        f"{coalesced_bytes / len(windows) / 1024:.1f} | "
        f"{args.workers} workers, fetch-only, avg decode window {avg_decode_samples:.2f} samples |"
    )
    print()
    print("| Local Stage | wall s |")
    print("|---|---:|")
    print(f"| compute frame windows from sidecar | {window_s:.3f} |")
    print(f"| coalesce byte windows | {coalesce_s:.3f} |")
    print(f"| raw byte windows | {len(windows)} |")
    print(f"| coalesced byte ranges | {len(coalesced)} |")
    print(f"| useful sample MiB | {useful_bytes / 1024**2:.1f} |")


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
            range_backend="native-http",
            label="indexed-native-http-sidecar",
            sidecar_path=str(sidecar_path),
        )
        print()
        run_indexed_strategy(
            meta,
            data_root,
            args,
            parquet_reader,
            range_backend="fsspec",
            label="indexed-sidecar",
            sidecar_path=str(sidecar_path),
        )
        print()
        run_random_frame_strategy(
            meta,
            data_root,
            args,
            range_backend=args.random_frame_backend,
            label=f"random-frames-{args.random_frame_backend}-sidecar",
            sidecar_path=str(sidecar_path),
        )
        return
    if sidecar_path is not None and args.strategy == "indexed":
        run_indexed_strategy(
            meta,
            data_root,
            args,
            parquet_reader,
            range_backend="fsspec",
            label="indexed-sidecar",
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
            label="indexed-native-http-sidecar",
            sidecar_path=str(sidecar_path),
        )
        return
    if sidecar_path is not None and args.strategy == "random-frames":
        run_random_frame_strategy(
            meta,
            data_root,
            args,
            range_backend=args.random_frame_backend,
            label=f"random-frames-{args.random_frame_backend}-sidecar",
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
    if args.strategy == "both":
        print()
    if args.strategy in ("both", "random-frames"):
        run_random_frame_strategy(
            meta,
            data_root,
            args,
            range_backend=args.random_frame_backend,
            label=f"random-frames-{args.random_frame_backend}",
            sidecar_path=None,
        )


if __name__ == "__main__":
    main()
