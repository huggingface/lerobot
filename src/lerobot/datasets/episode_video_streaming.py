# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import contextlib
import io
import json
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import fsspec
import httpx
import numpy as np
from huggingface_hub import HfApi, HfFileSystem, constants
from huggingface_hub.utils import hf_raise_for_status

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.mp4 import Mp4Index, Mp4SampleSlice, fetch_mp4_index, synthesize_mp4


@dataclass(frozen=True)
class EpisodeVideoSpan:
    file_id: int
    mdat_offset: int
    mdat_length: int
    first_pts: float
    last_pts: float
    frame_count: int
    sample_lo: int
    sample_hi: int
    source_start_pts: float


@dataclass(frozen=True)
class VideoFileRecord:
    file_path: str
    file_size: int
    mp4: Mp4Index


class ThreadLocalRangeFetcher:
    """Range reader that gives each worker thread independent file handles."""

    def __init__(self, data_root: str | Path, *, block_size: int = 2**20, cache_type: str = "none"):
        self.data_root = str(data_root).rstrip("/")
        protocol = "hf" if self.data_root.startswith("hf://") else "file"
        self.fs = fsspec.filesystem(protocol)
        self.block_size = block_size
        self.cache_type = cache_type
        self._local = threading.local()
        self._timing_lock = threading.Lock()
        self._timing_totals = {
            "range_jobs": 0.0,
            "range_bytes": 0.0,
            "range_open_s": 0.0,
            "range_seek_s": 0.0,
            "range_read_s": 0.0,
        }

    def _url(self, relative_path: str) -> str:
        if self.data_root.startswith("hf://"):
            return f"{self.data_root}/{relative_path}"
        return str(Path(self.data_root) / relative_path)

    def _handle(self, relative_path: str):
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = {}
            self._local.handles = handles
        handle = handles.get(relative_path)
        if handle is None or getattr(handle, "closed", False):
            handle = self.fs.open(
                self._url(relative_path), "rb", block_size=self.block_size, cache_type=self.cache_type
            )
            handles[relative_path] = handle
        return handle

    def info_size(self, relative_path: str) -> int:
        return int(self.fs.info(self._url(relative_path))["size"])

    def read_range(self, relative_path: str, offset: int, length: int) -> bytes:
        open_start = time.perf_counter()
        handle = self._handle(relative_path)
        open_s = time.perf_counter() - open_start
        seek_start = time.perf_counter()
        handle.seek(offset)
        seek_s = time.perf_counter() - seek_start
        read_start = time.perf_counter()
        data = handle.read(length)
        read_s = time.perf_counter() - read_start
        self._record_timing(
            range_jobs=1.0,
            range_bytes=float(len(data)),
            range_open_s=open_s,
            range_seek_s=seek_s,
            range_read_s=read_s,
        )
        return data

    def _record_timing(self, **kwargs: float) -> None:
        with self._timing_lock:
            for key, value in kwargs.items():
                self._timing_totals[key] = self._timing_totals.get(key, 0.0) + value

    def timing_summary(self) -> dict[str, float]:
        with self._timing_lock:
            return dict(self._timing_totals)

    def close(self) -> None:
        handles = getattr(self._local, "handles", None)
        if handles is None:
            return
        for handle in handles.values():
            with contextlib.suppress(Exception):
                handle.close()
        handles.clear()


class NativeHTTPRangeFetcher:
    """Direct pooled HTTP range reader for hf:// paths."""

    _GLOBAL_SOURCE_URLS: dict[tuple[str, str], str] = {}
    _GLOBAL_RESOLVED_URLS: dict[tuple[str, str], str] = {}
    _GLOBAL_SIZES: dict[tuple[str, str], int] = {}
    _GLOBAL_LOCK = threading.Lock()

    _RETRYABLE_EXCEPTIONS = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadError,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
    )

    def __init__(
        self,
        data_root: str | Path,
        *,
        max_connections: int = 32,
        timeout: float = 60.0,
        max_retries: int = 4,
    ):
        self.data_root = str(data_root).rstrip("/")
        if not self.data_root.startswith("hf://"):
            raise ValueError("NativeHTTPRangeFetcher only supports hf:// roots")
        self.max_retries = max_retries
        self.api = HfApi()
        self.fs: HfFileSystem | None = None
        self._bucket_id: str | None = None
        self._bucket_prefix = ""
        if self.data_root.startswith("hf://buckets/"):
            bucket_root = self.data_root.removeprefix("hf://buckets/")
            parts = bucket_root.split("/", 2)
            if len(parts) < 2:
                raise ValueError(f"Invalid bucket root: {self.data_root}")
            self._bucket_id = f"{parts[0]}/{parts[1]}"
            self._bucket_prefix = parts[2].strip("/") if len(parts) == 3 else ""
        else:
            self.fs = HfFileSystem()
        self.client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            follow_redirects=False,
        )
        self._resolved_urls: dict[str, str] = {}
        self._source_urls: dict[str, str] = {}
        self._sizes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._timing_lock = threading.Lock()
        self._timing_totals = {
            "range_jobs": 0.0,
            "range_bytes": 0.0,
            "range_resolve_s": 0.0,
            "range_header_s": 0.0,
            "range_first_byte_s": 0.0,
            "range_body_s": 0.0,
            "range_retry_attempts": 0.0,
            "range_retry_sleep_s": 0.0,
            "range_failed_requests": 0.0,
        }

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.request(method, url, **kwargs)
            except self._RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(0.5 * 2**attempt, 5.0))
        if last_exc is None:
            raise RuntimeError("HTTP request failed without an exception")
        raise last_exc

    def _cache_key(self, relative_path: str) -> tuple[str, str]:
        return self.data_root, relative_path

    def _path(self, relative_path: str) -> str:
        return f"{self.data_root}/{relative_path}"

    def _bucket_path(self, relative_path: str) -> str:
        if self._bucket_prefix:
            return f"{self._bucket_prefix}/{relative_path}"
        return relative_path

    def _headers_for(self, request_url: str, source_url: str) -> dict[str, str]:
        headers = self.api._build_hf_headers()
        if urlparse(request_url).netloc != urlparse(source_url).netloc:
            headers.pop("authorization", None)
            headers.pop("Authorization", None)
        return headers

    def _source_url(self, relative_path: str) -> str:
        with self._lock:
            source = self._source_urls.get(relative_path)
            if source is not None:
                return source
        key = self._cache_key(relative_path)
        with self._GLOBAL_LOCK:
            source = self._GLOBAL_SOURCE_URLS.get(key)
        if source is None:
            if self._bucket_id is not None:
                source = (
                    f"{constants.ENDPOINT}/buckets/{self._bucket_id}/resolve/"
                    f"{quote(self._bucket_path(relative_path))}"
                )
            else:
                if self.fs is None:
                    raise RuntimeError("HfFileSystem fallback was not initialized")
                source = self.fs.url(self._path(relative_path))
            with self._GLOBAL_LOCK:
                self._GLOBAL_SOURCE_URLS[key] = source
        with self._lock:
            self._source_urls[relative_path] = source
            return source

    def _resolve_url(self, relative_path: str, *, refresh: bool = False) -> str:
        with self._lock:
            if not refresh and relative_path in self._resolved_urls:
                return self._resolved_urls[relative_path]
        key = self._cache_key(relative_path)
        if not refresh:
            with self._GLOBAL_LOCK:
                resolved = self._GLOBAL_RESOLVED_URLS.get(key)
                size = self._GLOBAL_SIZES.get(key)
            if resolved is not None:
                with self._lock:
                    self._resolved_urls[relative_path] = resolved
                    if size is not None:
                        self._sizes[relative_path] = size
                return resolved

        source = self._source_url(relative_path)
        response = self._request("HEAD", source, headers=self.api._build_hf_headers(), follow_redirects=False)
        try:
            hf_raise_for_status(response)
            location = response.headers.get("Location")
            resolved = urljoin(source, location) if location else source
            with self._lock:
                self._resolved_urls[relative_path] = resolved
                if "Content-Length" in response.headers:
                    self._sizes[relative_path] = int(response.headers["Content-Length"])
            with self._GLOBAL_LOCK:
                self._GLOBAL_RESOLVED_URLS[key] = resolved
                if "Content-Length" in response.headers:
                    self._GLOBAL_SIZES[key] = int(response.headers["Content-Length"])
            return resolved
        finally:
            response.close()

    def info_size(self, relative_path: str) -> int:
        with self._lock:
            size = self._sizes.get(relative_path)
            if size is not None:
                return size
        key = self._cache_key(relative_path)
        with self._GLOBAL_LOCK:
            size = self._GLOBAL_SIZES.get(key)
        if size is not None:
            with self._lock:
                self._sizes[relative_path] = size
            return size

        resolved = self._resolve_url(relative_path)
        source = self._source_url(relative_path)
        response = self._request(
            "HEAD", resolved, headers=self._headers_for(resolved, source), follow_redirects=True
        )
        try:
            hf_raise_for_status(response)
            size = int(response.headers["Content-Length"])
            with self._lock:
                self._sizes[relative_path] = size
            with self._GLOBAL_LOCK:
                self._GLOBAL_SIZES[key] = size
            return size
        finally:
            response.close()

    def read_range(self, relative_path: str, offset: int, length: int) -> bytes:
        resolve_start = time.perf_counter()
        resolved = self._resolve_url(relative_path)
        source = self._source_url(relative_path)
        resolve_s = time.perf_counter() - resolve_start
        headers = self._headers_for(resolved, source)
        headers["Range"] = f"bytes={offset}-{offset + length - 1}"
        payload, status_code, timings = self._read_range_response(resolved, headers)
        if status_code == 403:
            refresh_start = time.perf_counter()
            resolved = self._resolve_url(relative_path, refresh=True)
            resolve_s += time.perf_counter() - refresh_start
            headers = self._headers_for(resolved, source)
            headers["Range"] = f"bytes={offset}-{offset + length - 1}"
            payload, status_code, retry_timings = self._read_range_response(resolved, headers)
            for key, value in retry_timings.items():
                timings[key] += value
        if status_code == 403:
            raise PermissionError(f"HTTP range request returned 403 after URL refresh: {relative_path}")
        self._record_timing(
            range_jobs=1.0,
            range_bytes=float(len(payload)),
            range_resolve_s=resolve_s,
            **{f"range_status_{status_code}": 1.0},
            **timings,
        )
        return payload

    def _read_range_response(self, url: str, headers: dict[str, str]) -> tuple[bytes, int, dict[str, float]]:
        last_exc: Exception | None = None
        retry_attempts = 0.0
        retry_sleep_s = 0.0
        for attempt in range(self.max_retries + 1):
            try:
                payload, status_code, timings = self._read_range_response_once(url, headers)
                timings["range_retry_attempts"] = retry_attempts
                timings["range_retry_sleep_s"] = retry_sleep_s
                return payload, status_code, timings
            except self._RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                retry_attempts += 1.0
                sleep_s = min(0.5 * 2**attempt, 5.0)
                retry_sleep_s += sleep_s
                time.sleep(sleep_s)
        self._record_timing(
            range_failed_requests=1.0,
            range_retry_attempts=retry_attempts,
            range_retry_sleep_s=retry_sleep_s,
        )
        if last_exc is None:
            raise RuntimeError("HTTP range request failed without an exception")
        raise last_exc

    def _read_range_response_once(
        self, url: str, headers: dict[str, str]
    ) -> tuple[bytes, int, dict[str, float]]:
        header_start = time.perf_counter()
        with self.client.stream("GET", url, headers=headers) as response:
            header_s = time.perf_counter() - header_start
            if response.status_code == 403:
                return (
                    b"",
                    response.status_code,
                    {
                        "range_header_s": header_s,
                        "range_first_byte_s": 0.0,
                        "range_body_s": 0.0,
                    },
                )
            hf_raise_for_status(response)
            chunks = []
            first_byte_s = 0.0
            first_chunk = True
            chunk_gap_s = 0.0
            chunk_count = 0.0
            previous_chunk_at = body_start = time.perf_counter()
            for chunk in response.iter_bytes():
                now = time.perf_counter()
                if first_chunk:
                    first_byte_s = now - body_start
                    first_chunk = False
                chunk_gap_s += now - previous_chunk_at
                previous_chunk_at = now
                chunk_count += 1.0
                chunks.append(chunk)
            body_s = time.perf_counter() - body_start
            join_start = time.perf_counter()
            payload = b"".join(chunks)
            join_s = time.perf_counter() - join_start
            return (
                payload,
                response.status_code,
                {
                    "range_header_s": header_s,
                    "range_first_byte_s": first_byte_s,
                    "range_body_s": body_s,
                    "range_join_s": join_s,
                    "range_chunks": chunk_count,
                    "range_chunk_gap_s": chunk_gap_s,
                },
            )

    def _record_timing(self, **kwargs: float) -> None:
        with self._timing_lock:
            for key, value in kwargs.items():
                self._timing_totals[key] = self._timing_totals.get(key, 0.0) + value

    def timing_summary(self) -> dict[str, float]:
        with self._timing_lock:
            return dict(self._timing_totals)

    def close(self) -> None:
        self.client.close()


def make_range_fetcher(
    data_root: str | Path,
    *,
    range_backend: str,
    workers: int,
    native_http_connections: int | None = None,
    native_http_timeout: float = 60.0,
    native_http_retries: int = 4,
):
    if range_backend == "fsspec":
        return ThreadLocalRangeFetcher(data_root)
    if range_backend == "native-http":
        max_connections = native_http_connections or max(8, workers)
        return NativeHTTPRangeFetcher(
            data_root,
            max_connections=max_connections,
            timeout=native_http_timeout,
            max_retries=native_http_retries,
        )
    raise ValueError(f"Unknown range backend: {range_backend}")


class EpisodeVideoManifest:
    _FILE_SIDECAR_CACHE: dict[str, dict[str, VideoFileRecord]] = {}
    _FILE_SIDECAR_CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        *,
        video_keys: list[str],
        files: list[VideoFileRecord],
        spans: dict[str, np.ndarray],
    ):
        self.video_keys = list(video_keys)
        self._camera_to_id = {key: idx for idx, key in enumerate(self.video_keys)}
        self.files = files
        self.spans = spans

    @classmethod
    def build(
        cls,
        meta: LeRobotDatasetMetadata,
        data_root: str | Path,
        *,
        episode_indices: list[int] | range | None = None,
        range_backend: str = "fsspec",
        workers: int = 8,
        header_probe_bytes: int = 4 * 1024 * 1024,
        max_probe_bytes: int = 64 * 1024 * 1024,
        keyframe_pad_s: float = 0.1,
        keyframe_pad_fraction: float = 0.05,
        sidecar_path: str | Path | None = None,
    ) -> EpisodeVideoManifest:
        meta.ensure_readable()
        video_keys = list(meta.video_keys)
        if episode_indices is None:
            episode_indices = range(int(meta.total_episodes))
        rel_paths = sorted(
            {str(meta.get_video_file_path(ep_idx, key)) for ep_idx in episode_indices for key in video_keys}
        )
        path_to_id = {path: idx for idx, path in enumerate(rel_paths)}
        if sidecar_path is None:
            files = cls._build_file_records(
                rel_paths,
                data_root,
                range_backend=range_backend,
                workers=workers,
                header_probe_bytes=header_probe_bytes,
                max_probe_bytes=max_probe_bytes,
            )
        else:
            records = cls.load_file_sidecar(sidecar_path)
            missing = [path for path in rel_paths if path not in records]
            if missing:
                raise ValueError(
                    f"Sidecar {sidecar_path} is missing {len(missing)} files, first: {missing[0]}"
                )
            files = [records[path] for path in rel_paths]

        total = int(meta.total_episodes)
        num_cameras = len(video_keys)
        spans: dict[str, np.ndarray] = {
            "file_id": np.zeros((total, num_cameras), dtype=np.int32),
            "mdat_offset": np.zeros((total, num_cameras), dtype=np.int64),
            "mdat_length": np.zeros((total, num_cameras), dtype=np.int64),
            "first_pts": np.zeros((total, num_cameras), dtype=np.float64),
            "last_pts": np.zeros((total, num_cameras), dtype=np.float64),
            "frame_count": np.zeros((total, num_cameras), dtype=np.int32),
            "sample_lo": np.zeros((total, num_cameras), dtype=np.int32),
            "sample_hi": np.zeros((total, num_cameras), dtype=np.int32),
            "source_start_pts": np.zeros((total, num_cameras), dtype=np.float64),
        }

        for ep_idx in episode_indices:
            ep = meta.episodes[ep_idx]
            for cam_idx, key in enumerate(video_keys):
                rel_path = str(meta.get_video_file_path(ep_idx, key))
                file_id = path_to_id[rel_path]
                mp4 = files[file_id].mp4
                from_ts = float(ep[f"videos/{key}/from_timestamp"])
                to_ts = float(ep[f"videos/{key}/to_timestamp"])
                sample_slice = mp4.sample_slice(
                    from_ts,
                    to_ts,
                    keyframe_pad_s=keyframe_pad_s,
                    keyframe_pad_fraction=keyframe_pad_fraction,
                    file_size=files[file_id].file_size,
                )
                spans["file_id"][ep_idx, cam_idx] = file_id
                spans["mdat_offset"][ep_idx, cam_idx] = sample_slice.byte_offset
                spans["mdat_length"][ep_idx, cam_idx] = sample_slice.byte_length
                spans["first_pts"][ep_idx, cam_idx] = from_ts
                spans["last_pts"][ep_idx, cam_idx] = to_ts
                spans["frame_count"][ep_idx, cam_idx] = sample_slice.sample_hi - sample_slice.sample_lo + 1
                spans["sample_lo"][ep_idx, cam_idx] = sample_slice.sample_lo
                spans["sample_hi"][ep_idx, cam_idx] = sample_slice.sample_hi
                spans["source_start_pts"][ep_idx, cam_idx] = sample_slice.source_start_pts

        return cls(video_keys=video_keys, files=files, spans=spans)

    @staticmethod
    def _build_file_records(
        rel_paths: list[str],
        data_root: str | Path,
        *,
        range_backend: str,
        workers: int,
        header_probe_bytes: int,
        max_probe_bytes: int,
    ) -> list[VideoFileRecord]:
        fetcher = make_range_fetcher(data_root, range_backend=range_backend, workers=workers)

        def build_file(path: str) -> VideoFileRecord:
            file_size = fetcher.info_size(path)
            mp4 = fetch_mp4_index(
                path,
                fetcher.read_range,
                file_size=file_size,
                header_probe_bytes=header_probe_bytes,
                max_probe_bytes=max_probe_bytes,
            )
            return VideoFileRecord(path, file_size, mp4)

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(build_file, rel_paths))
        finally:
            fetcher.close()

    @classmethod
    def write_file_sidecar(
        cls,
        sidecar_path: str | Path,
        rel_paths: list[str],
        data_root: str | Path,
        *,
        range_backend: str = "native-http",
        workers: int = 8,
        header_probe_bytes: int = 4 * 1024 * 1024,
        max_probe_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        records = cls._build_file_records(
            sorted(set(rel_paths)),
            data_root,
            range_backend=range_backend,
            workers=workers,
            header_probe_bytes=header_probe_bytes,
            max_probe_bytes=max_probe_bytes,
        )
        cls.save_file_sidecar(sidecar_path, records)

    @staticmethod
    def save_file_sidecar(sidecar_path: str | Path, records: list[VideoFileRecord]) -> None:
        sidecar_path = Path(sidecar_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "files": [
                {"file_path": record.file_path, "file_size": record.file_size, "mp4": record.mp4.to_dict()}
                for record in records
            ],
        }
        arrays = {}
        for file_idx, record in enumerate(records):
            arrays[f"{file_idx}/sample_pts"] = record.mp4.sample_pts
            arrays[f"{file_idx}/sample_durations"] = record.mp4.sample_durations
            arrays[f"{file_idx}/sample_sizes"] = record.mp4.sample_sizes
            arrays[f"{file_idx}/sample_offsets"] = record.mp4.sample_offsets
            arrays[f"{file_idx}/sync_samples"] = record.mp4.sync_samples
        np.savez_compressed(sidecar_path, manifest_json=json.dumps(payload).encode("utf-8"), **arrays)

    @staticmethod
    def load_file_sidecar(sidecar_path: str | Path) -> dict[str, VideoFileRecord]:
        cache_key = str(Path(sidecar_path).expanduser())
        with EpisodeVideoManifest._FILE_SIDECAR_CACHE_LOCK:
            cached = EpisodeVideoManifest._FILE_SIDECAR_CACHE.get(cache_key)
        if cached is not None:
            return cached

        with np.load(sidecar_path, allow_pickle=False) as data:
            payload = json.loads(bytes(data["manifest_json"]).decode("utf-8"))
            records = {}
            for file_idx, item in enumerate(payload["files"]):
                arrays = {
                    name: data[f"{file_idx}/{name}"]
                    for name in [
                        "sample_pts",
                        "sample_durations",
                        "sample_sizes",
                        "sample_offsets",
                        "sync_samples",
                    ]
                }
                mp4 = Mp4Index.from_dict(item["mp4"], arrays)
                records[item["file_path"]] = VideoFileRecord(item["file_path"], int(item["file_size"]), mp4)
        with EpisodeVideoManifest._FILE_SIDECAR_CACHE_LOCK:
            EpisodeVideoManifest._FILE_SIDECAR_CACHE[cache_key] = records
        return records

    def camera_id(self, camera_key: str) -> int:
        return self._camera_to_id[camera_key]

    def lookup(self, episode_index: int, camera_key: str) -> EpisodeVideoSpan:
        cam = self.camera_id(camera_key)
        return EpisodeVideoSpan(
            file_id=int(self.spans["file_id"][episode_index, cam]),
            mdat_offset=int(self.spans["mdat_offset"][episode_index, cam]),
            mdat_length=int(self.spans["mdat_length"][episode_index, cam]),
            first_pts=float(self.spans["first_pts"][episode_index, cam]),
            last_pts=float(self.spans["last_pts"][episode_index, cam]),
            frame_count=int(self.spans["frame_count"][episode_index, cam]),
            sample_lo=int(self.spans["sample_lo"][episode_index, cam]),
            sample_hi=int(self.spans["sample_hi"][episode_index, cam]),
            source_start_pts=float(self.spans["source_start_pts"][episode_index, cam]),
        )

    def file_lookup(self, file_id: int) -> VideoFileRecord:
        return self.files[file_id]

    def mp4_index(self, episode_index: int, camera_key: str) -> Mp4Index:
        return self.files[self.lookup(episode_index, camera_key).file_id].mp4

    def sample_slice(self, episode_index: int, camera_key: str) -> Mp4SampleSlice:
        span = self.lookup(episode_index, camera_key)
        return Mp4SampleSlice(
            sample_lo=span.sample_lo,
            sample_hi=span.sample_hi,
            byte_offset=span.mdat_offset,
            byte_length=span.mdat_length,
            source_start_pts=span.source_start_pts,
        )


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
        open_decoders: bool = True,
    ):
        self.manifest = manifest
        self.fetcher = make_range_fetcher(
            data_root,
            range_backend=range_backend,
            workers=workers,
            native_http_connections=native_http_connections,
            native_http_timeout=native_http_timeout,
            native_http_retries=native_http_retries,
        )
        self.byte_budget = byte_budget
        self.open_decoders = open_decoders
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._cache: OrderedDict[tuple[int, str], dict[str, Any]] = OrderedDict()
        self._futures: dict[tuple[int, str], Future[dict[str, Any]]] = {}
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
        self._pool.shutdown(wait=True)
        with self._lock:
            self._cache.clear()
            self._futures.clear()
            self._bytes = 0
        self.fetcher.close()

    def __enter__(self) -> EpisodeByteCache:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def submit_prefetch(self, episode_index: int) -> None:
        for camera_key in self.manifest.video_keys:
            self._submit(episode_index, camera_key)

    def ensure_ready(self, episode_index: int) -> None:
        for camera_key in self.manifest.video_keys:
            self.get_bytes(episode_index, camera_key)

    def get_bytes(self, episode_index: int, camera_key: str) -> bytes:
        return self._get_entry(episode_index, camera_key)["bytes"]

    def get_decoder(self, episode_index: int, camera_key: str):
        entry = self._get_entry(episode_index, camera_key)
        decoder = entry.get("decoder")
        if decoder is None:
            decoder = open_video_decoder(io.BytesIO(entry["bytes"]))
            entry["decoder"] = decoder
        return decoder

    def get_frames(self, episode_index: int, camera_key: str, timestamps: list[float]):
        span = self.manifest.lookup(episode_index, camera_key)
        local_ts = [ts - span.source_start_pts for ts in timestamps]
        decoder = self.get_decoder(episode_index, camera_key)
        if hasattr(decoder, "get_frames_played_at"):
            return decoder.get_frames_played_at(local_ts).data
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
            self._evict_locked()
            timings = entry.pop("_timings", None)
            if timings is not None:
                self._timing_totals["lookup_s"] += timings["lookup_s"]
                self._timing_totals["fetch_s"] += timings["fetch_s"]
                self._timing_totals["synthesize_s"] += timings["synthesize_s"]
                self._timing_totals["store_s"] += time.perf_counter() - store_start
                self._timing_totals["jobs"] += 1
            return entry

    def _evict_locked(self) -> None:
        while self._bytes > self.byte_budget and self._cache:
            _key, entry = self._cache.popitem(last=False)
            self._bytes -= len(entry["bytes"])

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
            "decoder": None,
            "_timings": {
                "lookup_s": lookup_s,
                "fetch_s": fetch_s,
                "synthesize_s": synthesize_s,
            },
        }
        if self.open_decoders:
            entry["decoder"] = open_video_decoder(io.BytesIO(mp4_bytes))
        return entry


def open_video_decoder(file_like_or_bytesio, frame_mappings=None):
    if frame_mappings is not None:
        raise ValueError("Synthesized episode videos use a local timeline; pass frame_mappings=None.")
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder(file_like_or_bytesio, seek_mode="approximate")


def assert_hf_hub_range_cache_branch() -> None:
    """Fail unless huggingface_hub was installed from the required range-cache branch."""

    try:
        dist = metadata.distribution("huggingface_hub")
    except metadata.PackageNotFoundError as exc:
        raise AssertionError("huggingface_hub is not installed") from exc

    candidates = []
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        candidates.append(direct_url)
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(direct_url)
            candidates.append(str(parsed.get("url", "")))
            candidates.append(str(parsed.get("vcs_info", {}).get("requested_revision", "")))
            candidates.append(str(parsed.get("vcs_info", {}).get("commit_id", "")))

    text = "\n".join(candidates)
    if "feat/hffs-cache-cdn-range-reads" not in text:
        raise AssertionError(
            "huggingface_hub must be installed from "
            "git+https://github.com/huggingface/huggingface_hub.git@feat/hffs-cache-cdn-range-reads"
        )


@dataclass
class StageTimer:
    fetch_ms: float = 0.0
    decode_ms: float = 0.0
    bytes_read: int = 0
    misses: int = 0

    def record_fetch(self, start: float, byte_count: int) -> None:
        self.fetch_ms += (time.perf_counter() - start) * 1000
        self.bytes_read += byte_count
        self.misses += 1
