# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from __future__ import annotations

import contextlib
import json
import os
import posixpath
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any
from urllib.parse import quote, urljoin, urlparse
from uuid import uuid4

import fsspec
import httpx
from huggingface_hub import HfApi, HfFileSystem, constants
from huggingface_hub.utils import get_session, hf_raise_for_status

_HTTP_FAILURE_LOG_LOCK = threading.Lock()


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
