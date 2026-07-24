# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from __future__ import annotations

import io
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from lerobot.streaming.manifest import EpisodeVideoManifest
from lerobot.streaming.mp4 import Mp4SampleSlice, synthesize_mp4
from lerobot.streaming.range_fetch import make_range_fetcher

logger = logging.getLogger(__name__)


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
        video_backend: str = "torchcodec",
        tolerance_s: float = 1e-4,
    ):
        if byte_budget <= 0:
            raise ValueError("byte_budget must be positive")
        if max_open_decoders <= 0:
            raise ValueError("max_open_decoders must be positive")
        if video_backend == "video_reader":
            video_backend = "pyav"
        if video_backend not in {"torchcodec", "pyav"}:
            raise ValueError(f"Unsupported video backend: {video_backend}")
        if tolerance_s <= 0:
            raise ValueError("tolerance_s must be positive")
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
        self.video_backend = video_backend
        self.tolerance_s = tolerance_s
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._cache: OrderedDict[tuple[int, str], dict[str, Any]] = OrderedDict()
        self._decoders: OrderedDict[tuple[int, str], Any] = OrderedDict()
        self._futures: dict[tuple[int, str], Future[dict[str, Any]]] = {}
        self._retained_episodes: dict[int, int] = {}
        self._decoder_fallback_count = 0
        self._fallback_decoders: set[tuple[int, str]] = set()
        self._fallback_warning_emitted = False
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
            decoders = list(self._decoders.values())
            self._cache.clear()
            self._decoders.clear()
            self._futures.clear()
            self._retained_episodes.clear()
            self._fallback_decoders.clear()
            self._bytes = 0
        for decoder in decoders:
            _close_decoder(decoder)
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

    @property
    def decoder_fallback_count(self) -> int:
        with self._lock:
            return self._decoder_fallback_count

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

        decoder = self._open_decoder(key, entry["bytes"])
        with self._lock:
            existing = self._decoders.get(key)
            if existing is not None:
                self._decoders.move_to_end(key)
                _close_decoder(decoder)
                return existing
            self._decoders[key] = decoder
            while len(self._decoders) > self.max_open_decoders:
                evicted_key, evicted_decoder = self._decoders.popitem(last=False)
                self._fallback_decoders.discard(evicted_key)
                _close_decoder(evicted_decoder)
        return decoder

    def _open_decoder(self, key: tuple[int, str], data: bytes) -> Any:
        try:
            if self.video_backend == "torchcodec":
                return open_video_decoder(io.BytesIO(data))
            return open_video_decoder(io.BytesIO(data), backend=self.video_backend)
        except Exception as primary_error:
            if self.video_backend != "torchcodec":
                raise
            try:
                decoder = open_video_decoder(io.BytesIO(data), backend="pyav")
            except Exception as fallback_error:
                raise RuntimeError(
                    "Both TorchCodec and PyAV rejected synthesized episode video "
                    f"{key}: TorchCodec error: {primary_error}"
                ) from fallback_error
            with self._lock:
                self._decoder_fallback_count += 1
                self._fallback_decoders.add(key)
                should_warn = not self._fallback_warning_emitted
                self._fallback_warning_emitted = True
            if should_warn:
                logger.warning(
                    "TorchCodec rejected a synthesized episode MP4; using the bounded PyAV "
                    "decoder fallback for affected videos. First error: %s",
                    primary_error,
                )
            else:
                logger.debug("Using PyAV decoder fallback for synthesized episode video %s", key)
            return decoder

    def get_frames(self, episode_index: int, camera_key: str, timestamps: list[float]):
        key = (episode_index, camera_key)
        span = self.manifest.lookup(episode_index, camera_key)
        local_ts = [ts - span.source_start_pts for ts in timestamps]
        decoder, release = self._decoder_for_frames(episode_index, camera_key)
        with self._lock:
            uses_pyav_timestamps = self.video_backend == "pyav" and key not in self._fallback_decoders
        try:
            if uses_pyav_timestamps:
                return decoder.get_frames_played_at(local_ts, tolerance_s=self.tolerance_s).data
            metadata = decoder.metadata
            fps = getattr(metadata, "average_fps", None)
            if fps is None:
                duration = max(getattr(metadata, "end_stream_seconds", 0.0), 1e-9)
                fps = metadata.num_frames / duration
            return decoder.get_frames_at(indices=[round(ts * fps) for ts in local_ts]).data
        finally:
            if release is not None:
                release()

    def _decoder_for_frames(
        self, episode_index: int, camera_key: str
    ) -> tuple[Any, Callable[[], None] | None]:
        key = (episode_index, camera_key)
        while True:
            decoder = self.get_decoder(episode_index, camera_key)
            acquire = getattr(decoder, "acquire", None)
            if acquire is None:
                return decoder, None
            try:
                acquire()
            except RuntimeError:
                # The decoder was evicted between lookup and lease acquisition. Remove a stale
                # cached reference if it raced with close, then retry with a fresh decoder.
                with self._lock:
                    if self._decoders.get(key) is decoder:
                        self._decoders.pop(key)
                        self._fallback_decoders.discard(key)
                continue
            return decoder, decoder.release

    def timing_summary(self) -> dict[str, float]:
        with self._lock:
            summary = dict(self._timing_totals)
            summary["decoder_fallbacks"] = float(self._decoder_fallback_count)
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
            decoder = self._decoders.pop(key, None)
            self._fallback_decoders.discard(key)
            if decoder is not None:
                _close_decoder(decoder)

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


class _PyAVVideoDecoder:
    """Small seekable PyAV adapter matching the TorchCodec calls used by the byte cache."""

    def __init__(self, file_like_or_bytesio: Any):
        import av

        self._source = file_like_or_bytesio
        self._container = av.open(file_like_or_bytesio)
        self._stream = self._container.streams.video[0]
        average_rate = self._stream.average_rate
        if average_rate is None:
            raise ValueError("PyAV video stream does not expose an average frame rate")
        self._fps = float(average_rate)
        duration = (
            float(self._stream.duration * self._stream.time_base)
            if self._stream.duration is not None
            else 0.0
        )
        self.metadata = SimpleNamespace(
            average_fps=self._fps,
            num_frames=int(self._stream.frames or round(duration * self._fps)),
            begin_stream_seconds=0.0,
            end_stream_seconds=duration,
        )
        self._decode_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._users = 0
        self._close_requested = False
        self._closed = False

    def acquire(self) -> None:
        with self._state_lock:
            if self._close_requested or self._closed:
                raise RuntimeError("PyAV decoder is closing")
            self._users += 1

    def release(self) -> None:
        with self._state_lock:
            self._users -= 1
            if self._users < 0:
                raise RuntimeError("Unbalanced PyAV decoder release")
            if self._users == 0 and self._close_requested:
                self._close_resources()

    def get_frames_at(self, *, indices: list[int]) -> SimpleNamespace:
        if not indices:
            import torch

            return SimpleNamespace(data=torch.empty((0, 3, 0, 0), dtype=torch.uint8))
        timestamps = [index / self._fps for index in indices]
        return self._get_frames_played_at(timestamps, tolerance_s=0.5 / self._fps + 1e-6)

    def get_frames_played_at(
        self,
        timestamps: list[float],
        *,
        tolerance_s: float,
    ) -> SimpleNamespace:
        return self._get_frames_played_at(timestamps, tolerance_s=tolerance_s)

    def _get_frames_played_at(
        self,
        timestamps: list[float],
        *,
        tolerance_s: float,
    ) -> SimpleNamespace:
        import torch

        first_ts = min(timestamps)
        last_ts = max(timestamps)
        loaded_frames: list[torch.Tensor] = []
        loaded_ts: list[float] = []
        with self._decode_lock:
            self._container.seek(
                round(first_ts / self._stream.time_base) - 1,
                backward=True,
                any_frame=False,
                stream=self._stream,
            )
            for frame in self._container.decode(self._stream):
                if frame.pts is None:
                    continue
                current_ts = float(frame.pts * self._stream.time_base)
                array = frame.to_ndarray(format="rgb24")
                loaded_frames.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
                loaded_ts.append(current_ts)
                if current_ts >= last_ts:
                    break

        if not loaded_frames:
            raise ValueError(f"PyAV decoded no frames for timestamps {timestamps}")
        query_ts = torch.tensor(timestamps)
        loaded_ts_tensor = torch.tensor(loaded_ts)
        distances = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
        minimum, closest = distances.min(1)
        if not (minimum <= tolerance_s).all():
            raise ValueError(
                f"PyAV frame timestamps exceed tolerance {tolerance_s}: "
                f"queries={query_ts}, decoded={loaded_ts_tensor}"
            )
        return SimpleNamespace(data=torch.stack([loaded_frames[index] for index in closest]))

    def close(self) -> None:
        with self._state_lock:
            self._close_requested = True
            if self._users == 0:
                self._close_resources()

    def _close_resources(self) -> None:
        if self._closed:
            return
        self._container.close()
        close = getattr(self._source, "close", None)
        if close is not None:
            close()
        self._closed = True


def _close_decoder(decoder: Any) -> None:
    close = getattr(decoder, "close", None)
    if close is not None:
        try:
            close()
        except Exception:
            logger.debug("Failed to close video decoder", exc_info=True)


def open_video_decoder(file_like_or_bytesio, frame_mappings=None, *, backend: str = "torchcodec"):
    if frame_mappings is not None:
        raise ValueError("Synthesized episode videos use a local timeline; pass frame_mappings=None.")
    if backend == "pyav":
        return _PyAVVideoDecoder(file_like_or_bytesio)
    if backend != "torchcodec":
        raise ValueError(f"Unsupported video backend: {backend}")
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(file_like_or_bytesio, seek_mode="approximate")
