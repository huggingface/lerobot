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
import logging
import os
import posixpath
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urljoin, urlparse
from uuid import uuid4

import fsspec
import httpx
import numpy as np
from huggingface_hub import HfApi, HfFileSystem, constants
from huggingface_hub.utils import get_session, hf_raise_for_status

from lerobot.streaming.mp4 import (
    Mp4Index,
    Mp4SampleSlice,
    fetch_mp4_index,
    synthesize_mp4,
    synthesized_mp4_size,
)

if TYPE_CHECKING:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.streaming.sidecar import SidecarSpec

_HTTP_FAILURE_LOG_LOCK = threading.Lock()


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


def _get_header(headers: Any, name: str) -> str | None:
    if hasattr(headers, "get"):
        return headers.get(name)
    lower_name = name.lower()
    for key, value in headers.items():
        if key.lower() == lower_name:
            return value
    return None


def _ensure_request_id(headers: dict[str, str]) -> str:
    request_id = _get_header(headers, "X-Amzn-Trace-Id") or _get_header(headers, "X-Request-Id")
    if request_id is None:
        request_id = str(uuid4())
        headers["X-Amzn-Trace-Id"] = request_id
    return request_id


def _log_http_failure(
    *,
    backend: str,
    method: str,
    url: str,
    headers: dict[str, str],
    elapsed_s: float,
    status_code: int | None = None,
    exception: Exception | None = None,
    attempt: int | None = None,
    response_headers: Any | None = None,
) -> None:
    log_path = os.environ.get("LEROBOT_HTTP_FAILURE_LOG")
    if not log_path:
        return
    parsed = urlparse(url)
    record = {
        "ts": datetime.now(UTC).isoformat(),
        "backend": backend,
        "method": method,
        "host": parsed.netloc,
        "path": parsed.path,
        "range": _get_header(headers, "Range") or _get_header(headers, "range"),
        "request_id": _get_header(headers, "X-Amzn-Trace-Id") or _get_header(headers, "X-Request-Id"),
        "elapsed_s": round(elapsed_s, 6),
    }
    if attempt is not None:
        record["attempt"] = attempt
    if status_code is not None:
        record["status_code"] = status_code
    if exception is not None:
        record["exception_type"] = type(exception).__name__
        record["exception"] = str(exception)
    if response_headers is not None:
        record["response_request_id"] = (
            _get_header(response_headers, "x-request-id")
            or _get_header(response_headers, "x-amz-cf-id")
            or _get_header(response_headers, "x-amz-request-id")
        )
        record["cache_status"] = (
            _get_header(response_headers, "x-cache")
            or _get_header(response_headers, "cf-cache-status")
            or _get_header(response_headers, "x-hf-cache")
        )
        record["content_range"] = _get_header(response_headers, "content-range")
        record["content_length"] = _get_header(response_headers, "content-length")

    path = Path(log_path).expanduser()
    with _HTTP_FAILURE_LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as out:
            out.write(json.dumps(record, sort_keys=True) + "\n")


class ThreadLocalRangeFetcher:
    """Range reader that gives each worker thread independent file handles."""

    def __init__(self, data_root: str | Path, *, block_size: int = 2**20, cache_type: str = "none"):
        self.data_root = str(data_root).rstrip("/")
        self.fs, self._root_path = fsspec.core.url_to_fs(self.data_root)
        self._is_local = self.fs.protocol in ("file", "local") or (
            isinstance(self.fs.protocol, tuple) and "file" in self.fs.protocol
        )
        self.block_size = block_size
        self.cache_type = cache_type
        self._local = threading.local()
        self._handles_lock = threading.Lock()
        self._all_handles: dict[int, Any] = {}
        self._timing_lock = threading.Lock()
        self._timing_totals = {
            "range_jobs": 0.0,
            "range_bytes": 0.0,
            "range_open_s": 0.0,
            "range_seek_s": 0.0,
            "range_read_s": 0.0,
        }

    def _url(self, relative_path: str) -> str:
        if self._is_local:
            return str(Path(self._root_path) / relative_path)
        return posixpath.join(self._root_path.rstrip("/"), relative_path.lstrip("/"))

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
            self._instrument_hf_handle(handle)
            handles[relative_path] = handle
            with self._handles_lock:
                self._all_handles[id(handle)] = handle
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

    def _instrument_hf_handle(self, handle: Any) -> None:
        if getattr(handle, "_lerobot_range_timing", False):
            return
        if not hasattr(handle, "_request_with_retry"):
            return

        def request_with_retry(
            handle_self,
            method: str,
            url: str,
            *,
            headers: dict[str, str],
            follow_redirects: bool | None = None,
            max_retries: int = 5,
        ) -> httpx.Response:
            from huggingface_hub.hf_file_system import _RANGE_RETRY_EXCEPTIONS, _RANGE_RETRY_STATUS_CODES

            method_key = method.lower()
            sleep_time = 1.0
            retry_attempts = 0.0
            retry_sleep_s = 0.0
            failed_attempt_s = 0.0
            exception_attempts = 0.0
            extra_counts: dict[str, float] = {}
            call_start = time.perf_counter()
            request_kwargs: dict[str, Any] = {
                "headers": headers,
                "timeout": constants.HF_HUB_DOWNLOAD_TIMEOUT,
            }
            _ensure_request_id(headers)
            if follow_redirects is not None:
                request_kwargs["follow_redirects"] = follow_redirects

            for attempt in range(max_retries + 1):
                attempt_start = time.perf_counter()
                try:
                    response = get_session().request(method, url, **request_kwargs)
                except _RANGE_RETRY_EXCEPTIONS as exc:
                    attempt_s = time.perf_counter() - attempt_start
                    failed_attempt_s += attempt_s
                    exception_attempts += 1.0
                    key = f"range_hffs_{method_key}_exception_{type(exc).__name__}"
                    extra_counts[key] = extra_counts.get(key, 0.0) + 1.0
                    _log_http_failure(
                        backend="hffs",
                        method=method,
                        url=url,
                        headers=headers,
                        elapsed_s=attempt_s,
                        exception=exc,
                        attempt=attempt,
                    )
                    if attempt == max_retries:
                        self._record_hffs_request_timing(
                            method_key,
                            time.perf_counter() - call_start,
                            retry_attempts,
                            retry_sleep_s,
                            failed_attempt_s,
                            exception_attempts,
                            None,
                            0,
                            extra_counts,
                        )
                        raise
                else:
                    elapsed = time.perf_counter() - attempt_start
                    if response.status_code not in _RANGE_RETRY_STATUS_CODES or attempt == max_retries:
                        self._record_hffs_request_timing(
                            method_key,
                            time.perf_counter() - call_start,
                            retry_attempts,
                            retry_sleep_s,
                            failed_attempt_s,
                            exception_attempts,
                            response.status_code,
                            len(response.content),
                            extra_counts,
                        )
                        return response

                    failed_attempt_s += elapsed
                    key = f"range_hffs_{method_key}_failed_status_{response.status_code}"
                    extra_counts[key] = extra_counts.get(key, 0.0) + 1.0
                    response.close()

                    _log_http_failure(
                        backend="hffs",
                        method=method,
                        url=url,
                        headers=headers,
                        elapsed_s=elapsed,
                        status_code=response.status_code,
                        attempt=attempt,
                        response_headers=response.headers,
                    )

                time.sleep(sleep_time)
                retry_attempts += 1.0
                retry_sleep_s += sleep_time
                sleep_time = min(8.0, sleep_time * 2)

            raise RuntimeError("unreachable")

        handle._request_with_retry = MethodType(request_with_retry, handle)
        handle._lerobot_range_timing = True

    def _record_hffs_request_timing(
        self,
        method_key: str,
        total_s: float,
        retry_attempts: float,
        retry_sleep_s: float,
        failed_attempt_s: float,
        exception_attempts: float,
        status_code: int | None,
        byte_count: int,
        extra_counts: dict[str, float],
    ) -> None:
        timings = {
            f"range_hffs_{method_key}_requests": 1.0,
            f"range_hffs_{method_key}_s": total_s,
            f"range_hffs_{method_key}_retries": retry_attempts,
            f"range_hffs_{method_key}_retry_sleep_s": retry_sleep_s,
            f"range_hffs_{method_key}_failed_attempt_s": failed_attempt_s,
            f"range_hffs_{method_key}_exception_attempts": exception_attempts,
            f"range_hffs_{method_key}_bytes": float(byte_count),
        }
        if status_code is not None:
            timings[f"range_hffs_{method_key}_status_{status_code}"] = 1.0
        timings.update(extra_counts)
        self._record_timing(**timings)

    def timing_summary(self) -> dict[str, float]:
        with self._timing_lock:
            return dict(self._timing_totals)

    def close(self) -> None:
        with self._handles_lock:
            handles = list(self._all_handles.values())
            self._all_handles.clear()
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.close()
        local_handles = getattr(self._local, "handles", None)
        if local_handles is not None:
            local_handles.clear()


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
    _RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        data_root: str | Path,
        *,
        max_connections: int = 32,
        timeout: float = 60.0,
        max_retries: int = 4,
        subrange_parts: int = 1,
        subrange_min_bytes: int = 8 * 1024 * 1024,
    ):
        self.data_root = str(data_root).rstrip("/")
        if not self.data_root.startswith("hf://"):
            raise ValueError("NativeHTTPRangeFetcher only supports hf:// roots")
        self.max_retries = max_retries
        # Sub-range parallelism: split one large GET into `subrange_parts` concurrent GETs.
        # Under a per-host throughput ceiling this adds no aggregate bandwidth, but divides
        # per-request latency by ~parts - keep (in-flight jobs x parts) near the ceiling's
        # connection sweet spot (~64 on the observed HF bucket path) rather than raising both.
        self.subrange_parts = max(1, subrange_parts)
        self.subrange_min_bytes = max(1, subrange_min_bytes)
        self._subrange_pool = (
            ThreadPoolExecutor(max_workers=max_connections, thread_name_prefix="subrange")
            if self.subrange_parts > 1
            else None
        )
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
        parts = self.subrange_parts
        if self._subrange_pool is None or parts <= 1 or length < 2 * self.subrange_min_bytes:
            return self._read_range_single(relative_path, offset, length)
        parts = min(parts, max(1, length // self.subrange_min_bytes))
        if parts <= 1:
            return self._read_range_single(relative_path, offset, length)
        step = (length + parts - 1) // parts
        spans = [(offset + i * step, min(step, length - i * step)) for i in range(parts)]
        futures = [
            self._subrange_pool.submit(self._read_range_single, relative_path, span_off, span_len)
            for span_off, span_len in spans
        ]
        return b"".join(future.result() for future in futures)

    def _read_range_single(self, relative_path: str, offset: int, length: int) -> bytes:
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
        if status_code != 206:
            raise RuntimeError(f"HTTP range request returned {status_code} after retries: {relative_path}")
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
        failed_attempt_s = 0.0
        exception_attempts = 0.0
        exception_counts: dict[str, float] = {}
        _ensure_request_id(headers)
        for attempt in range(self.max_retries + 1):
            attempt_start = time.perf_counter()
            try:
                payload, status_code, timings = self._read_range_response_once(url, headers)
                if status_code in self._RETRYABLE_STATUS_CODES:
                    attempt_s = time.perf_counter() - attempt_start
                    failed_attempt_s += attempt_s
                    exception_attempts += 1.0
                    status_key = f"range_failed_status_{status_code}"
                    exception_counts[status_key] = exception_counts.get(status_key, 0.0) + 1.0
                    _log_http_failure(
                        backend="native-http",
                        method="GET",
                        url=url,
                        headers=headers,
                        elapsed_s=attempt_s,
                        status_code=status_code,
                        attempt=attempt,
                    )
                    if attempt >= self.max_retries:
                        timings["range_retry_attempts"] = retry_attempts
                        timings["range_retry_sleep_s"] = retry_sleep_s
                        timings["range_failed_attempt_s"] = failed_attempt_s
                        timings["range_exception_attempts"] = exception_attempts
                        timings.update(exception_counts)
                        return payload, status_code, timings
                    retry_attempts += 1.0
                    sleep_s = min(0.5 * 2**attempt, 5.0)
                    retry_sleep_s += sleep_s
                    time.sleep(sleep_s)
                    continue
                timings["range_retry_attempts"] = retry_attempts
                timings["range_retry_sleep_s"] = retry_sleep_s
                timings["range_failed_attempt_s"] = failed_attempt_s
                timings["range_exception_attempts"] = exception_attempts
                timings.update(exception_counts)
                return payload, status_code, timings
            except self._RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                attempt_s = time.perf_counter() - attempt_start
                failed_attempt_s += attempt_s
                exception_attempts += 1.0
                exception_key = f"range_exception_{type(exc).__name__}"
                exception_counts[exception_key] = exception_counts.get(exception_key, 0.0) + 1.0
                _log_http_failure(
                    backend="native-http",
                    method="GET",
                    url=url,
                    headers=headers,
                    elapsed_s=attempt_s,
                    exception=exc,
                    attempt=attempt,
                )
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
            range_failed_attempt_s=failed_attempt_s,
            range_exception_attempts=exception_attempts,
            **exception_counts,
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
            if response.status_code == 403 or response.status_code in self._RETRYABLE_STATUS_CODES:
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
        if self._subrange_pool is not None:
            self._subrange_pool.shutdown(wait=False, cancel_futures=True)
        self.client.close()


def make_range_fetcher(
    data_root: str | Path,
    *,
    range_backend: str,
    workers: int,
    native_http_connections: int | None = None,
    native_http_timeout: float = 60.0,
    native_http_retries: int = 4,
    native_http_subranges: int = 1,
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
            subrange_parts=native_http_subranges,
        )
    raise ValueError(f"Unknown range backend: {range_backend}")


class EpisodeVideoManifest:
    _FILE_SIDECAR_CACHE: dict[str, tuple[tuple[int, int], dict[str, VideoFileRecord]]] = {}
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
                futures = {pool.submit(build_file, path): path for path in rel_paths}
                records = []
                progress_interval = max(1, len(futures) // 20)
                for completed, future in enumerate(as_completed(futures), start=1):
                    records.append(future.result())
                    if completed == len(futures) or completed % progress_interval == 0:
                        logging.info("Indexed %d/%d MP4 files for streaming sidecar", completed, len(futures))
                return sorted(records, key=lambda record: record.file_path)
        finally:
            fetcher.close()

    @classmethod
    def write_file_sidecar(
        cls,
        sidecar_path: str | Path,
        rel_paths: list[str],
        data_root: str | Path,
        *,
        spec: SidecarSpec,
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
        cls.save_file_sidecar(sidecar_path, records, spec=spec)

    @staticmethod
    def save_file_sidecar(
        sidecar_path: str | Path,
        records: list[VideoFileRecord],
        *,
        spec: SidecarSpec,
    ) -> None:
        sidecar_path = Path(sidecar_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "sidecar": spec.with_source_files(
                tuple((record.file_path, record.file_size) for record in records)
            ).to_dict(),
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
        cache_key = str(sidecar_path.expanduser())
        with EpisodeVideoManifest._FILE_SIDECAR_CACHE_LOCK:
            EpisodeVideoManifest._FILE_SIDECAR_CACHE.pop(cache_key, None)

    @staticmethod
    def load_file_sidecar_metadata(sidecar_path: str | Path) -> dict[str, Any]:
        with np.load(sidecar_path, allow_pickle=False) as data:
            payload = json.loads(bytes(data["manifest_json"]).decode("utf-8"))
        if payload.get("version") != 2 or not isinstance(payload.get("sidecar"), dict):
            raise ValueError(f"Unsupported MP4 sidecar schema in {sidecar_path}")
        return payload["sidecar"]

    @staticmethod
    def validate_file_sidecar(sidecar_path: str | Path, spec: SidecarSpec) -> bool:
        try:
            from lerobot.streaming.sidecar import SidecarSpec

            candidate = SidecarSpec.from_dict(EpisodeVideoManifest.load_file_sidecar_metadata(sidecar_path))
            if not spec.matches(candidate):
                return False
            records = EpisodeVideoManifest.load_file_sidecar(sidecar_path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

        expected = dict(candidate.source_files)
        actual = {path: record.file_size for path, record in records.items()}
        return actual == expected

    @staticmethod
    def load_file_sidecar(sidecar_path: str | Path) -> dict[str, VideoFileRecord]:
        path = Path(sidecar_path).expanduser()
        cache_key = str(path)
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with EpisodeVideoManifest._FILE_SIDECAR_CACHE_LOCK:
            cached = EpisodeVideoManifest._FILE_SIDECAR_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

        with np.load(path, allow_pickle=False) as data:
            payload = json.loads(bytes(data["manifest_json"]).decode("utf-8"))
            if payload.get("version") != 2:
                raise ValueError(f"Unsupported MP4 sidecar schema in {path}")
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
            EpisodeVideoManifest._FILE_SIDECAR_CACHE[cache_key] = (signature, records)
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

    def episode_byte_size(self, episode_index: int) -> int:
        """Exact synthesized video bytes retained while an episode is active."""
        return sum(
            synthesized_mp4_size(
                self.mp4_index(episode_index, camera_key),
                self.sample_slice(episode_index, camera_key),
            )
            for camera_key in self.video_keys
        )


class ExactCoveragePool:
    """Deterministic, exactly-once frame coverage over a byte-cache episode pool.

    A with-replacement pool never guarantees a full
    epoch: frames are drawn randomly and episodes rotate on a fixed cadence. This planner instead
    enumerates *every frame of every episode exactly once per epoch* while keeping at most
    ``pool_size`` episodes resident, so batch mixing stays high but coverage is complete and
    reproducible.

    Mechanics (this is the "evict only when all frames sampled" model):
      - Episodes are admitted in a seeded global permutation until either ``pool_size`` or the
        optional indexed-byte budget is reached.
      - Each resident episode carries a seeded shuffle of its own frame indices.
      - Each draw picks a resident episode with probability proportional to its *remaining* frames
        (i.e. a uniform draw over all remaining frames in the pool, the map-style ideal) and pops
        one frame.
      - An episode is evicted only when its last frame is emitted; a new episode is then admitted.
      - The epoch ends when the admission order is exhausted and every resident episode is drained.

    Newly admitted episodes are surfaced via :attr:`newly_admitted` (drain it to drive prefetch)
    and evictions via :attr:`evicted` (drain to release cache bytes). The planner does no I/O and
    is fully unit-testable. It yields ``(episode_index, frame_index)``; map to a decode timestamp
    with ``frame_index / max(frame_count - 1, 1)``.

    Determinism: the order is a pure function of ``(seed, epoch)``, the episode frame counts, and
    optional byte sizes/budget. Resume is a deterministic fast-forward: re-instantiate with the
    same inputs and skip ``n`` samples (tabular only, no decode).
    """

    def __init__(
        self,
        episode_frame_counts: Sequence[tuple[int, int]],
        pool_size: int,
        *,
        seed: int,
        epoch: int = 0,
        episode_byte_sizes: Mapping[int, int] | None = None,
        byte_budget: int | None = None,
    ):
        self._counts = {int(ep): int(n) for ep, n in episode_frame_counts if int(n) > 0}
        self._rng = np.random.default_rng([seed, epoch])
        order = np.array(sorted(self._counts), dtype=np.int64)
        self._rng.shuffle(order)
        self.pool_size = max(1, pool_size)
        self._byte_budget = byte_budget
        if byte_budget is not None and byte_budget <= 0:
            raise ValueError("byte_budget must be positive")
        if byte_budget is not None and episode_byte_sizes is None:
            raise ValueError("episode_byte_sizes are required when byte_budget is set")
        self._byte_sizes = {
            episode: int(episode_byte_sizes[episode]) if episode_byte_sizes is not None else 0
            for episode in self._counts
        }
        if any(size < 0 for size in self._byte_sizes.values()):
            raise ValueError("episode byte sizes must be non-negative")
        if byte_budget is not None:
            oversized = next(
                ((episode, size) for episode, size in self._byte_sizes.items() if size > byte_budget),
                None,
            )
            if oversized is not None:
                episode, size = oversized
                raise ValueError(
                    f"Episode {episode} requires {size} bytes, exceeding the byte budget {byte_budget}"
                )

        # Preserve the full seeded order for benchmark/tooling compatibility. Byte-aware admission
        # may temporarily skip an entry, but every episode remains in this deterministic frontier.
        self.admission_order: list[int] = order.tolist()
        self._pending: list[int] = list(self.admission_order)
        self._admitted_count = 0
        self._remaining: dict[int, tuple[np.ndarray, int]] = {}
        self._remaining_total = 0
        self._resident_bytes = 0
        self.newly_admitted: list[int] = []
        self.evicted: list[int] = []
        self._admit_available()

    def _admit_available(self) -> None:
        while len(self._remaining) < self.pool_size and self._pending:
            available_bytes = None if self._byte_budget is None else self._byte_budget - self._resident_bytes
            pending_index = next(
                (
                    index
                    for index, episode in enumerate(self._pending)
                    if available_bytes is None or self._byte_sizes[episode] <= available_bytes
                ),
                None,
            )
            if pending_index is None:
                return

            episode = self._pending.pop(pending_index)
            frame_count = self._counts[episode]
            frames = np.arange(frame_count, dtype=np.int64)
            self._rng.shuffle(frames)
            self._remaining[episode] = (frames, frame_count)
            self._remaining_total += frame_count
            self._resident_bytes += self._byte_sizes[episode]
            self._admitted_count += 1
            self.newly_admitted.append(episode)

    @property
    def remaining_total(self) -> int:
        return self._remaining_total

    @property
    def admitted_count(self) -> int:
        """Number of episodes pulled from the admission order so far (pool fills + rotations)."""
        return self._admitted_count

    @property
    def resident(self) -> list[int]:
        return list(self._remaining)

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    def prefetch_candidates(self, count: int) -> list[int]:
        """Return the next deterministic pending frontier without admitting it."""
        if count <= 0:
            return []
        return self._pending[:count]

    def __iter__(self) -> ExactCoveragePool:
        return self

    def __next__(self) -> tuple[int, int]:
        if self._remaining_total == 0:
            raise StopIteration
        # Uniform draw over all remaining frames in the pool: walk the residents by cumulative
        # remaining count. O(pool_size) per draw (~1024) -> negligible next to decode.
        target = int(self._rng.integers(self._remaining_total))
        chosen = None
        for ep, (_frames, remaining) in self._remaining.items():
            if target < remaining:
                chosen = ep
                break
            target -= remaining
        frames, remaining = self._remaining[chosen]
        remaining -= 1
        frame_index = int(frames[remaining])
        self._remaining_total -= 1
        if remaining == 0:
            del self._remaining[chosen]
            self.evicted.append(chosen)
            self._resident_bytes -= self._byte_sizes[chosen]
            self._admit_available()
        else:
            self._remaining[chosen] = (frames, remaining)
        return chosen, frame_index


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
