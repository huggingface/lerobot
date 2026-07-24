# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from __future__ import annotations

import io
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from lerobot.streaming.manifest import EpisodeVideoManifest
from lerobot.streaming.mp4 import Mp4SampleSlice, synthesize_mp4
from lerobot.streaming.range_fetch import make_range_fetcher


class EpisodeByteCache:
    def __init__(
        self,
        manifest: EpisodeVideoManifest,
        data_root: str | Path,
        *,
        byte_budget: int = 80 * 1024**3,
        workers: int = 8,
        range_backend: str = "fsspec",
        native_http_connections: int | None = None,
        native_http_timeout: float = 60.0,
        native_http_retries: int = 4,
        native_http_subranges: int = 1,
        open_decoders: bool = True,
        max_open_decoders: int = 64,
    ):
        if byte_budget <= 0:
            raise ValueError("byte_budget must be positive")
        if max_open_decoders <= 0:
            raise ValueError("max_open_decoders must be positive")
        self.manifest = manifest
        self.fetcher = make_range_fetcher(
            data_root,
            range_backend=range_backend,
            workers=workers,
            native_http_connections=native_http_connections,
            native_http_timeout=native_http_timeout,
            native_http_retries=native_http_retries,
            native_http_subranges=native_http_subranges,
        )
        self.byte_budget = byte_budget
        self.open_decoders = open_decoders
        self.max_open_decoders = max_open_decoders
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._cache: OrderedDict[tuple[int, str], dict[str, Any]] = OrderedDict()
        self._decoders: OrderedDict[tuple[int, str], Any] = OrderedDict()
        self._futures: dict[tuple[int, str], Future[dict[str, Any]]] = {}
        self._retained_episodes: dict[int, int] = {}
        self._bytes = 0
        self._lock = threading.Lock()
        self._timing_totals = {
            "lookup_s": 0.0,
            "fetch_s": 0.0,
            "synthesize_s": 0.0,
            "store_s": 0.0,
            "jobs": 0.0,
        }

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._cache.clear()
            self._decoders.clear()
            self._futures.clear()
            self._retained_episodes.clear()
            self._bytes = 0
        self.fetcher.close()

    def __enter__(self) -> EpisodeByteCache:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def submit_prefetch(self, episode_index: int) -> None:
        for camera_key in self.manifest.video_keys:
            self._submit(episode_index, camera_key)

    def retain_episode(self, episode_index: int) -> None:
        with self._lock:
            self._retained_episodes[episode_index] = self._retained_episodes.get(episode_index, 0) + 1

    def release_episode(self, episode_index: int) -> None:
        with self._lock:
            count = self._retained_episodes.get(episode_index, 0)
            if count <= 1:
                self._retained_episodes.pop(episode_index, None)
            else:
                self._retained_episodes[episode_index] = count - 1
            self._evict_locked()

    @property
    def resident_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @property
    def open_decoder_count(self) -> int:
        with self._lock:
            return len(self._decoders)

    def ensure_ready(self, episode_index: int) -> None:
        for camera_key in self.manifest.video_keys:
            self.get_bytes(episode_index, camera_key)
            if self.open_decoders:
                self.get_decoder(episode_index, camera_key)

    def is_ready(self, episode_index: int) -> bool:
        """Non-blocking: True when every camera of the episode is fetched (cached or future done).

        Lets a consumer swap in replacements only when they are already resident, instead of
        blocking the training hot path on a remote fetch (head-of-line stall).
        """
        for camera_key in self.manifest.video_keys:
            key = (episode_index, camera_key)
            with self._lock:
                if key in self._cache:
                    continue
                future = self._futures.get(key)
            if future is None or not future.done():
                return False
        return True

    def get_bytes(self, episode_index: int, camera_key: str) -> bytes:
        return self._get_entry(episode_index, camera_key)["bytes"]

    def get_decoder(self, episode_index: int, camera_key: str) -> Any:
        key = (episode_index, camera_key)
        entry = self._get_entry(episode_index, camera_key)
        with self._lock:
            decoder = self._decoders.get(key)
            if decoder is not None:
                self._decoders.move_to_end(key)
                return decoder

        decoder = open_video_decoder(io.BytesIO(entry["bytes"]))
        with self._lock:
            existing = self._decoders.get(key)
            if existing is not None:
                self._decoders.move_to_end(key)
                return existing
            self._decoders[key] = decoder
            while len(self._decoders) > self.max_open_decoders:
                self._decoders.popitem(last=False)
        return decoder

    def get_frames(self, episode_index: int, camera_key: str, timestamps: list[float]):
        span = self.manifest.lookup(episode_index, camera_key)
        local_ts = [ts - span.source_start_pts for ts in timestamps]
        decoder = self.get_decoder(episode_index, camera_key)
        metadata = decoder.metadata
        fps = getattr(metadata, "average_fps", None)
        if fps is None:
            duration = max(getattr(metadata, "end_stream_seconds", 0.0), 1e-9)
            fps = metadata.num_frames / duration
        return decoder.get_frames_at(indices=[round(ts * fps) for ts in local_ts]).data

    def timing_summary(self) -> dict[str, float]:
        with self._lock:
            summary = dict(self._timing_totals)
        fetcher_summary = getattr(self.fetcher, "timing_summary", None)
        if fetcher_summary is not None:
            summary.update(fetcher_summary())
        return summary

    def _submit(self, episode_index: int, camera_key: str) -> Future[dict[str, Any]]:
        key = (episode_index, camera_key)
        with self._lock:
            if key in self._cache:
                future: Future[dict[str, Any]] = Future()
                future.set_result(self._cache[key])
                return future
            future = self._futures.get(key)
            if future is None:
                future = self._pool.submit(self._fetch_and_synthesize, episode_index, camera_key)
                self._futures[key] = future
            return future

    def _get_entry(self, episode_index: int, camera_key: str) -> dict[str, Any]:
        key = (episode_index, camera_key)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
                return entry
        future = self._submit(episode_index, camera_key)
        entry = future.result()
        store_start = time.perf_counter()
        with self._lock:
            self._futures.pop(key, None)
            existing = self._cache.get(key)
            if existing is not None:
                self._cache.move_to_end(key)
                return existing
            self._cache[key] = entry
            self._bytes += len(entry["bytes"])
            try:
                self._evict_locked()
            except MemoryError:
                failed_entry = self._cache.pop(key, None)
                if failed_entry is not None:
                    self._bytes -= len(failed_entry["bytes"])
                raise
            timings = entry.pop("_timings", None)
            if timings is not None:
                self._timing_totals["lookup_s"] += timings["lookup_s"]
                self._timing_totals["fetch_s"] += timings["fetch_s"]
                self._timing_totals["synthesize_s"] += timings["synthesize_s"]
                self._timing_totals["store_s"] += time.perf_counter() - store_start
                self._timing_totals["jobs"] += 1
            return entry

    def _evict_locked(self) -> None:
        while self._bytes > self.byte_budget:
            key = next(
                (candidate for candidate in self._cache if candidate[0] not in self._retained_episodes),
                None,
            )
            if key is None:
                raise MemoryError(
                    f"Retained episode bytes exceed byte budget ({self._bytes} > {self.byte_budget})"
                )
            entry = self._cache.pop(key)
            self._bytes -= len(entry["bytes"])
            self._decoders.pop(key, None)

    def _fetch_and_synthesize(self, episode_index: int, camera_key: str) -> dict[str, Any]:
        lookup_start = time.perf_counter()
        span = self.manifest.lookup(episode_index, camera_key)
        file_record = self.manifest.file_lookup(span.file_id)
        sample_slice = Mp4SampleSlice(
            sample_lo=span.sample_lo,
            sample_hi=span.sample_hi,
            byte_offset=span.mdat_offset,
            byte_length=span.mdat_length,
            source_start_pts=span.source_start_pts,
        )
        lookup_s = time.perf_counter() - lookup_start
        fetch_start = time.perf_counter()
        payload = self.fetcher.read_range(file_record.file_path, span.mdat_offset, span.mdat_length)
        fetch_s = time.perf_counter() - fetch_start
        if len(payload) != span.mdat_length:
            raise OSError(
                f"Short read for {file_record.file_path}: expected {span.mdat_length}, got {len(payload)}"
            )
        synthesize_start = time.perf_counter()
        mp4_bytes = synthesize_mp4(file_record.mp4, sample_slice, payload)
        synthesize_s = time.perf_counter() - synthesize_start
        entry: dict[str, Any] = {
            "bytes": mp4_bytes,
            "_timings": {
                "lookup_s": lookup_s,
                "fetch_s": fetch_s,
                "synthesize_s": synthesize_s,
            },
        }
        return entry


def open_video_decoder(file_like_or_bytesio, frame_mappings=None):
    if frame_mappings is not None:
        raise ValueError("Synthesized episode videos use a local timeline; pass frame_mappings=None.")
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(file_like_or_bytesio, seek_mode="approximate")
