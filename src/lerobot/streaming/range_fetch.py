# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Thread-local fsspec and pooled native-HTTP byte-range readers."""

from __future__ import annotations

import contextlib
import posixpath
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote, urljoin, urlparse
from uuid import uuid4

import fsspec
import httpx
from huggingface_hub import HfApi, HfFileSystem, constants
from huggingface_hub.utils import hf_raise_for_status

from lerobot.streaming.location import LocationKind, StorageLocation


def _retry_delay_s(attempt: int) -> float:
    """Exponential backoff shared by HEAD and range retries."""
    return min(0.5 * 2**attempt, 5.0)


class ThreadLocalRangeFetcher:
    """Range reader that gives each worker thread independent file handles."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        block_size: int = 2**20,
        cache_type: str = "none",
        max_open_files: int = 8,
        token: str | bool | None = None,
    ) -> None:
        """Keep at most max_open_files source handles per fetch worker."""
        if max_open_files <= 0:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = max_open_files
        self.data_root = str(data_root).rstrip("/")
        storage_options = StorageLocation.parse(self.data_root).storage_options(token)
        self.fs, self._root_path = fsspec.core.url_to_fs(self.data_root, **storage_options)
        self._is_local = self.fs.protocol in ("file", "local") or (
            isinstance(self.fs.protocol, tuple) and "file" in self.fs.protocol
        )
        self.block_size = block_size
        self.cache_type = cache_type
        self._local = threading.local()
        self._handles_lock = threading.Lock()
        self._all_handles: dict[int, BinaryIO] = {}

    def _url(self, relative_path: str) -> str:
        """Resolve a dataset-relative path for the configured filesystem."""
        if self._is_local:
            return str(Path(self._root_path) / relative_path)
        return posixpath.join(self._root_path.rstrip("/"), relative_path.lstrip("/"))

    def _handle(self, relative_path: str) -> BinaryIO:
        """Reuse this thread's source handle and evict its least recently used handles."""
        handles: OrderedDict[str, BinaryIO] | None = getattr(self._local, "handles", None)
        if handles is None:
            handles = OrderedDict()
            self._local.handles = handles
        handle = handles.get(relative_path)
        if handle is None or getattr(handle, "closed", False):
            handle = self.fs.open(
                self._url(relative_path), "rb", block_size=self.block_size, cache_type=self.cache_type
            )
            handles[relative_path] = handle
            with self._handles_lock:
                self._all_handles[id(handle)] = handle
            while len(handles) > self.max_open_files:
                _, evicted = handles.popitem(last=False)
                try:
                    evicted.close()
                finally:
                    with self._handles_lock:
                        self._all_handles.pop(id(evicted), None)
        handles.move_to_end(relative_path)
        return handle

    def info_size(self, relative_path: str) -> int:
        """Return the source file size."""
        return int(self.fs.info(self._url(relative_path))["size"])

    def read_range(self, relative_path: str, offset: int, length: int) -> bytes:
        """Read an exact byte range through this thread's handle."""
        handle = self._handle(relative_path)
        handle.seek(offset)
        return handle.read(length)

    def close(self) -> None:
        """Close every thread-local source handle."""
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
        token: str | bool | None = None,
    ) -> None:
        """Configure direct pooled range requests for an HF object-store root."""
        self.data_root = str(data_root).rstrip("/")
        location = StorageLocation.parse(self.data_root)
        if not location.is_hf:
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
        self.api = HfApi(token=token)
        self.fs: HfFileSystem | None = None
        self._bucket_id: str | None = None
        self._bucket_prefix = ""
        if location.kind is LocationKind.HF_BUCKET:
            self._bucket_id = location.repo_id
            self._bucket_prefix = location.path_in_repo
        else:
            self.fs = HfFileSystem(token=token)
        self.client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            follow_redirects=False,
        )
        self._resolved_urls: dict[str, str] = {}
        self._source_urls: dict[str, str] = {}
        self._sizes: dict[str, int] = {}
        self._lock = threading.Lock()

    def _request(
        self, method: str, url: str, *, headers: Mapping[str, str], follow_redirects: bool
    ) -> httpx.Response:
        """Retry HEAD resolution failures, leaving the final response to its caller.

        Retryable statuses share the range request policy. Close intermediate
        responses before backoff; permanent errors return immediately.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.request(
                    method, url, headers=headers, follow_redirects=follow_redirects
                )
            except self._RETRYABLE_EXCEPTIONS:
                if attempt >= self.max_retries:
                    raise
            else:
                if response.status_code not in self._RETRYABLE_STATUS_CODES or attempt >= self.max_retries:
                    return response
                response.close()
            time.sleep(_retry_delay_s(attempt))
        raise RuntimeError("unreachable")

    def _path(self, relative_path: str) -> str:
        """Join the HF data root and a dataset-relative source path."""
        return f"{self.data_root}/{relative_path}"

    def _bucket_path(self, relative_path: str) -> str:
        """Include the configured bucket prefix in an object path."""
        if self._bucket_prefix:
            return f"{self._bucket_prefix}/{relative_path}"
        return relative_path

    def _headers_for(self, request_url: str, source_url: str) -> dict[str, str]:
        """Drop Hub authorization when a request targets a different host."""
        headers = self.api._build_hf_headers()
        if urlparse(request_url).netloc != urlparse(source_url).netloc:
            headers.pop("authorization", None)
            headers.pop("Authorization", None)
        return headers

    def _source_url(self, relative_path: str) -> str:
        """Cache the Hub URL used to resolve one source object."""
        with self._lock:
            source = self._source_urls.get(relative_path)
            if source is not None:
                return source
        if self._bucket_id is not None:
            source = (
                f"{constants.ENDPOINT}/buckets/{self._bucket_id}/resolve/"
                f"{quote(self._bucket_path(relative_path))}"
            )
        else:
            if self.fs is None:
                raise RuntimeError("HfFileSystem fallback was not initialized")
            source = self.fs.url(self._path(relative_path))
        with self._lock:
            self._source_urls[relative_path] = source
            return source

    def _resolve_url(self, relative_path: str, *, refresh: bool = False) -> str:
        """Resolve or refresh a source URL and cache its advertised size."""
        with self._lock:
            if not refresh and relative_path in self._resolved_urls:
                return self._resolved_urls[relative_path]
        source = self._source_url(relative_path)
        # A cached Hub redirect can contain the same expired signed URL. Refresh
        # the redirect as well as our local entry, without changing source identity.
        request_url = str(httpx.URL(source).copy_add_param("_refresh", str(uuid4()))) if refresh else source
        response = self._request(
            "HEAD", request_url, headers=self.api._build_hf_headers(), follow_redirects=False
        )
        try:
            hf_raise_for_status(response)
            location = response.headers.get("Location")
            resolved = urljoin(source, location) if location else source
            # A redirect's Content-Length describes its body, not the video object.
            size = response.headers.get("X-Linked-Size")
            if size is None and not response.is_redirect:
                size = response.headers.get("Content-Length")
            with self._lock:
                self._resolved_urls[relative_path] = resolved
                if size is not None:
                    self._sizes[relative_path] = int(size)
            return resolved
        finally:
            response.close()

    def info_size(self, relative_path: str) -> int:
        """Resolve and cache a remote source file size."""
        with self._lock:
            size = self._sizes.get(relative_path)
            if size is not None:
                return size
        resolved = self._resolve_url(relative_path)
        with self._lock:
            size = self._sizes.get(relative_path)
            if size is not None:
                return size
        source = self._source_url(relative_path)
        response = self._request(
            "HEAD", resolved, headers=self._headers_for(resolved, source), follow_redirects=True
        )
        try:
            hf_raise_for_status(response)
            size = int(response.headers["Content-Length"])
            with self._lock:
                self._sizes[relative_path] = size
            return size
        finally:
            response.close()

    def read_range(self, relative_path: str, offset: int, length: int) -> bytes:
        """Read a remote byte range, optionally split across pooled requests."""
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
        """Read one range, refreshing an expired URL once before failing."""
        source = self._source_url(relative_path)
        resolved = self._resolve_url(relative_path)
        payload, status_code = self._read_range_response(resolved, source, offset, length)
        if status_code in (401, 403):
            resolved = self._resolve_url(relative_path, refresh=True)
            payload, status_code = self._read_range_response(resolved, source, offset, length)
        if status_code in (401, 403):
            raise PermissionError(
                f"HTTP range request returned {status_code} after URL refresh: {relative_path}"
            )
        if status_code != 206:
            raise RuntimeError(f"HTTP range request returned {status_code} after retries: {relative_path}")
        return payload

    def _read_range_response(self, url: str, source: str, offset: int, length: int) -> tuple[bytes, int]:
        """Retry transient range failures and return the payload and final status."""
        headers = self._headers_for(url, source)
        headers["Range"] = f"bytes={offset}-{offset + length - 1}"
        for attempt in range(self.max_retries + 1):
            try:
                payload, status_code = self._read_range_response_once(url, headers)
            except self._RETRYABLE_EXCEPTIONS:
                if attempt >= self.max_retries:
                    raise
            else:
                if status_code not in self._RETRYABLE_STATUS_CODES or attempt >= self.max_retries:
                    return payload, status_code
            time.sleep(_retry_delay_s(attempt))
        raise RuntimeError("unreachable")

    def _read_range_response_once(self, url: str, headers: dict[str, str]) -> tuple[bytes, int]:
        """Read one HTTP response; denied and retryable statuses return an empty payload."""
        with self.client.stream("GET", url, headers=headers) as response:
            if response.status_code in (401, 403) or response.status_code in self._RETRYABLE_STATUS_CODES:
                return b"", response.status_code
            hf_raise_for_status(response)
            # Response.read() caches the body in HTTPX's response/stream cycle,
            # retaining a second video copy until cyclic GC runs. Consume the
            # chunks without populating that cache; context exit still closes it.
            return b"".join(response.iter_bytes()), response.status_code

    def close(self) -> None:
        """Close the HTTP client and subrange executor."""
        if self._subrange_pool is not None:
            self._subrange_pool.shutdown(wait=True, cancel_futures=True)
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
    token: str | bool | None = None,
) -> ThreadLocalRangeFetcher | NativeHTTPRangeFetcher:
    """Construct the configured local/fsspec or native-HTTP range reader."""
    if range_backend == "fsspec":
        return ThreadLocalRangeFetcher(data_root, token=token)
    if range_backend == "native-http":
        max_connections = native_http_connections or max(8, workers)
        return NativeHTTPRangeFetcher(
            data_root,
            max_connections=max_connections,
            timeout=native_http_timeout,
            max_retries=native_http_retries,
            subrange_parts=native_http_subranges,
            token=token,
        )
    raise ValueError(f"Unknown range backend: {range_backend}")
