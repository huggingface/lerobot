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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        choices=("both", "full", "indexed", "remote-decoder", "native-http", "gop-window"),
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
    parser.add_argument(
        "--include-gop-window",
        action="store_true",
        help="Also benchmark random frame GOP/window byte-range fetches from the MP4 sidecar.",
    )
    parser.add_argument(
        "--gop-window-post-frames",
        type=int,
        default=0,
        help="Extra compressed samples after each target frame to include in GOP/window ranges.",
    )
    parser.add_argument(
        "--gop-window-merge-gap-kb",
        type=int,
        default=0,
        help="Merge GOP/window ranges from the same MP4 when the byte gap is at most this many KiB.",
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
class GopWindowRange:
    file_path: str
    offset: int
    length: int
    target_frames: int
    covered_samples: int


def _sample_bounds_for_episode(manifest: EpisodeVideoManifest, episode_index: int, camera_key: str):
    span = manifest.lookup(episode_index, camera_key)
    mp4 = manifest.file_lookup(span.file_id).mp4
    sample_count = len(mp4.sample_pts)
    if sample_count == 0:
        raise ValueError(f"{mp4.file_path} contains no indexed samples")
    lo = int(np.searchsorted(mp4.sample_pts, span.first_pts, side="left"))
    hi = int(np.searchsorted(mp4.sample_pts, span.last_pts, side="right")) - 1
    lo = min(max(lo, 0), sample_count - 1)
    hi = min(max(hi, lo), sample_count - 1)
    return span, mp4, lo, hi


def _byte_range_for_samples(mp4, sample_lo: int, sample_hi: int, *, file_size: int) -> tuple[int, int]:
    offsets = mp4.sample_offsets[sample_lo : sample_hi + 1]
    sizes = mp4.sample_sizes[sample_lo : sample_hi + 1]
    byte_lo = int(offsets.min())
    byte_hi = int((offsets + sizes).max())
    byte_hi = min(byte_hi, file_size)
    return byte_lo, byte_hi - byte_lo


def _gop_window_for_target_sample(
    manifest: EpisodeVideoManifest,
    episode_index: int,
    camera_key: str,
    target_sample: int,
    *,
    post_frames: int,
) -> GopWindowRange:
    span = manifest.lookup(episode_index, camera_key)
    file_record = manifest.file_lookup(span.file_id)
    mp4 = file_record.mp4
    sync = mp4.sync_samples[mp4.sync_samples <= target_sample]
    sample_lo = int(sync[-1]) if len(sync) else 0
    sample_hi = min(max(target_sample + post_frames, sample_lo), span.sample_hi, len(mp4.sample_pts) - 1)
    offset, length = _byte_range_for_samples(mp4, sample_lo, sample_hi, file_size=file_record.file_size)
    return GopWindowRange(
        file_path=file_record.file_path,
        offset=offset,
        length=length,
        target_frames=1,
        covered_samples=sample_hi - sample_lo + 1,
    )


def _gop_window_ranges(
    manifest: EpisodeVideoManifest,
    episodes: Sequence[int],
    *,
    frames_per_episode: int,
    seed: int,
    post_frames: int,
    merge_gap_bytes: int,
) -> tuple[list[GopWindowRange], int, int, int]:
    rng = random.Random(seed)
    raw: list[GopWindowRange] = []
    compressed_target_bytes = 0
    covered_samples = 0
    for ep in episodes:
        for camera_key in manifest.video_keys:
            span, mp4, target_lo, target_hi = _sample_bounds_for_episode(manifest, ep, camera_key)
            for _ in range(frames_per_episode):
                ts = rng.uniform(span.first_pts, max(span.last_pts, span.first_pts))
                target = int(np.searchsorted(mp4.sample_pts, ts, side="left"))
                target = min(max(target, target_lo), target_hi)
                compressed_target_bytes += int(mp4.sample_sizes[target])
                window = _gop_window_for_target_sample(
                    manifest,
                    ep,
                    camera_key,
                    target,
                    post_frames=post_frames,
                )
                covered_samples += window.covered_samples
                raw.append(window)

    merged = _merge_gop_window_ranges(raw, merge_gap_bytes)
    return merged, len(raw), compressed_target_bytes, covered_samples


def _merge_gop_window_ranges(ranges: Sequence[GopWindowRange], merge_gap_bytes: int) -> list[GopWindowRange]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: (item.file_path, item.offset, item.length))
    merged: list[GopWindowRange] = []
    current = ordered[0]
    for item in ordered[1:]:
        current_end = current.offset + current.length
        if item.file_path == current.file_path and item.offset <= current_end + merge_gap_bytes:
            new_end = max(current_end, item.offset + item.length)
            current = GopWindowRange(
                file_path=current.file_path,
                offset=current.offset,
                length=new_end - current.offset,
                target_frames=current.target_frames + item.target_frames,
                covered_samples=current.covered_samples + item.covered_samples,
            )
        else:
            merged.append(current)
            current = item
    merged.append(current)
    return merged


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
    return result


def run_gop_window_fetch(
    manifest: EpisodeVideoManifest,
    data_root: str,
    episodes: Sequence[int],
    workers: int,
    range_backend: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    merge_gap_bytes = int(args.gop_window_merge_gap_kb * 1024)
    windows, raw_windows, compressed_target_bytes, covered_samples = _gop_window_ranges(
        manifest,
        episodes,
        frames_per_episode=args.frames_per_episode,
        seed=args.seed + 2,
        post_frames=args.gop_window_post_frames,
        merge_gap_bytes=merge_gap_bytes,
    )
    if not windows:
        raise ValueError("No GOP/window ranges were planned")

    fetcher = make_range_fetcher(
        data_root,
        range_backend=range_backend,
        workers=workers,
        native_http_connections=args.native_http_connections,
        native_http_timeout=args.native_http_timeout,
        native_http_retries=args.native_http_retries,
    )

    def fetch_window(window: GopWindowRange) -> int:
        payload = fetcher.read_range(window.file_path, window.offset, window.length)
        if len(payload) != window.length:
            raise OSError(f"Short read for {window.file_path}: expected {window.length}, got {len(payload)}")
        return len(payload)

    byte_count = sum(window.length for window in windows)
    start = time.perf_counter()
    done = 0
    done_ranges = 0
    last_progress = start
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_window, window) for window in windows]
            for future in as_completed(futures):
                done += future.result()
                done_ranges += 1
                now = time.perf_counter()
                if args.progress_interval > 0 and now - last_progress >= args.progress_interval:
                    elapsed = max(now - start, 1e-9)
                    _log(
                        "gop_window_progress: "
                        f"ranges_done={done_ranges}/{len(windows)} "
                        f"fetched={done / 1024**3:.2f} GiB "
                        f"fetch={done / elapsed / 1024**2:.1f} MiB/s "
                        f"elapsed={_format_duration(elapsed)}"
                    )
                    last_progress = now
    finally:
        timings = fetcher.timing_summary() if hasattr(fetcher, "timing_summary") else {}
        fetcher.close()

    elapsed = time.perf_counter() - start
    result = {
        "fetch_s": elapsed,
        "fetch_mbps": byte_count / elapsed / 1024**2,
        "frame_windows_s": raw_windows / elapsed,
        "ranges_s": len(windows) / elapsed,
        "bytes": float(byte_count),
        "raw_windows": float(raw_windows),
        "merged_windows": float(len(windows)),
        "compressed_target_bytes": float(compressed_target_bytes),
        "covered_samples": float(covered_samples),
        "avg_mb_range": byte_count / len(windows) / 1024**2,
        "avg_kib_frame_window": byte_count / raw_windows / 1024,
        "avg_compressed_kib_target": compressed_target_bytes / raw_windows / 1024,
        "avg_covered_samples": covered_samples / raw_windows,
    }
    result.update({key: value for key, value in timings.items() if key.startswith("range_")})
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
    fetch_pool = run_fetch_pool(manifest, data_root, episodes, byte_budget, args.workers, range_backend, args)
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


def run_gop_window_strategy(
    meta: LeRobotDatasetMetadata,
    data_root: str,
    args: argparse.Namespace,
    *,
    range_backend: str = "fsspec",
    sidecar_path: str | None = None,
) -> None:
    _log("starting_strategy: gop-window")
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
    _log(f"gop-window: manifest_build_s={manifest_s:.2f}")

    benchmark_episode_count = min(dataset_episode_count, args.num_episodes)
    episodes = _episode_pool(dataset_episode_count, args.num_episodes, args.pool_size, args.seed)
    full_episode_bytes = _bytes_for(manifest, episodes)
    result = run_gop_window_fetch(manifest, data_root, episodes, args.workers, range_backend, args)
    estimated_benchmark_s = benchmark_episode_count * args.frames_per_episode / result["frame_windows_s"]
    estimated_dataset_s = dataset_episode_count * args.frames_per_episode / result["frame_windows_s"]

    print(f"manifest_build_s: {manifest_s:.2f}")
    print("strategy: gop-window")
    print(f"range_backend: {range_backend}")
    print(f"mp4_sidecar: {sidecar_path or 'none'}")
    print(f"data_root: {data_root}")
    print(f"dataset_episodes: {dataset_episode_count}")
    print(f"benchmark_episodes: {benchmark_episode_count}")
    print(f"pool_episodes: {len(episodes)}")
    print(f"frames_per_episode: {args.frames_per_episode}")
    print(f"gop_window_post_frames: {args.gop_window_post_frames}")
    print(f"gop_window_merge_gap_kb: {args.gop_window_merge_gap_kb}")
    print(f"sampled_episodes: {episodes}")
    print(f"cameras: {manifest.video_keys}")
    print()
    print(
        "| Track | fetch MB/s | frame windows/s | ranges/s | wall s | "
        "est benchmark | est full dataset | notes |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    print(
        f"| GOP/WINDOW FETCH | {result['fetch_mbps']:.1f} | {result['frame_windows_s']:.1f} | "
        f"{result['ranges_s']:.1f} | {result['fetch_s']:.2f} | "
        f"{_format_duration(estimated_benchmark_s)} | {_format_duration(estimated_dataset_s)} | "
        f"{args.workers} workers, fetch-and-drop, no decoder open/frame decode |"
    )
    print()
    print("| GOP Window Shape | value |")
    print("|---|---:|")
    print(f"| target frame windows | {result['raw_windows']:.0f} |")
    print(f"| fetched byte ranges | {result['merged_windows']:.0f} |")
    print(f"| fetched GiB | {result['bytes'] / 1024**3:.2f} |")
    print(f"| full episode-pool GiB | {full_episode_bytes / 1024**3:.2f} |")
    print(f"| fetched/full episode bytes | {result['bytes'] / full_episode_bytes:.3f} |")
    print(f"| avg MiB/range | {result['avg_mb_range']:.3f} |")
    print(f"| avg KiB/frame window | {result['avg_kib_frame_window']:.1f} |")
    print(f"| avg compressed KiB/target frame | {result['avg_compressed_kib_target']:.1f} |")
    print(f"| avg compressed samples/window | {result['avg_covered_samples']:.1f} |")
    _print_range_timing_summary(result)
    _print_memory_summary(memory_start, _memory_snapshot())


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
        if args.include_gop_window:
            print()
            run_gop_window_strategy(
                meta,
                data_root,
                args,
                range_backend=args.range_backend,
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
        if args.include_gop_window:
            print()
            run_gop_window_strategy(
                meta,
                data_root,
                args,
                range_backend=args.range_backend,
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
    if sidecar_path is not None and args.strategy == "gop-window":
        run_gop_window_strategy(
            meta,
            data_root,
            args,
            range_backend=args.range_backend,
            sidecar_path=str(sidecar_path),
        )
        return
    if args.strategy in ("both", "gop-window"):
        expected_sidecar = SIDECAR_CACHE_DIR / FULL_SIDECAR_NAME
        expected_remote = _root_join(data_root, f"meta/mp4-sidecars/{FULL_SIDECAR_NAME}")
        print(f"mp4_sidecar_missing_local: {expected_sidecar}")
        print(f"mp4_sidecar_missing_remote: {expected_remote}")
        print(
            "build_mp4_sidecar: "
            "uv run --no-sync python scripts/build_mp4_sidecar.py "
            f"--workers {args.workers} --range-backend native-http --output {expected_sidecar}"
        )
        if args.strategy == "gop-window":
            print("gop_window_requires_mp4_sidecar: existing per-sample MP4 index sidecar is required")
            return
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
