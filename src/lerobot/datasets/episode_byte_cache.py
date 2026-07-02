"""Node-local LRU byte cache using precomputed byte-index manifest sidecars."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import fsspec

from .byte_index import EpisodeByteIndex, EpisodeSliceLookup
from .mp4_episode_slice import SparseMp4Reader

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    bytes_fetched: int = 0
    full_file_fallbacks: int = 0
    prefetch_submitted: int = 0
    prefetch_waits: int = 0
    mdat_slices: int = 0
    prefix_fetches: int = 0
    fetch_to_buffer_s: float = 0.0
    buffer_to_decoder_s: float = 0.0
    buffer_hit_decoder_s: float = 0.0
    decode_frame_s: float = 0.0
    decode_frames: int = 0

    def merge(self, other: CacheStats) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def stats_dict(self) -> dict[str, int | float]:
        avg_miss = self.bytes_fetched / max(1, self.misses)
        return {
            "byte_cache_hits": self.hits,
            "byte_cache_misses": self.misses,
            "byte_cache_bytes_fetched": self.bytes_fetched,
            "byte_cache_bytes_per_miss": avg_miss,
            "byte_cache_full_file_fallbacks": self.full_file_fallbacks,
            "byte_cache_prefetch_submitted": self.prefetch_submitted,
            "byte_cache_prefetch_waits": self.prefetch_waits,
            "byte_cache_mdat_slices": self.mdat_slices,
            "byte_cache_prefix_fetches": self.prefix_fetches,
            "fetch_to_buffer_ms_per_miss": 1000 * self.fetch_to_buffer_s / max(1, self.misses),
            "buffer_to_decoder_ms_per_miss": 1000 * self.buffer_to_decoder_s / max(1, self.misses),
            "buffer_hit_decoder_ms_per_hit": 1000 * self.buffer_hit_decoder_s / max(1, self.hits),
            "decode_ms_per_frame": 1000 * self.decode_frame_s / max(1, self.decode_frames),
        }


@dataclass
class _EpisodeEntry:
    decoders: dict[str, Any] = field(default_factory=dict)
    futures: dict[str, Future] = field(default_factory=dict)
    error: Exception | None = None


class RangeFetcher:
    """Byte-range GETs via fsspec, one request per range (no open/seek/read layering)."""

    def __init__(self, path: str):
        # Resolve any fsspec URL (hf://, s3://, gs://, plain local paths, ...), so S3-compatible
        # stores (e.g. Backblaze B2 via s3://) work identically to the Hub.
        self._fs, self.path = fsspec.core.url_to_fs(path)

    def fetch(self, lo: int, hi: int) -> bytes:
        if hi < lo:
            return b""
        # cat_file issues a single ranged GET (end-exclusive); fs.open would add a metadata
        # round-trip and buffered-read layering per fetch.
        return self._fs.cat_file(self.path, start=lo, end=hi + 1)


class EpisodeByteCache:
    """Manifest-driven episode MP4 fetch + in-memory sparse decode."""

    MAX_BYTES_PER_MISS = 25 * 1024 * 1024

    def __init__(
        self,
        byte_index: EpisodeByteIndex,
        max_bytes: int,
        *,
        data_root: str,
        max_prefetch_workers: int = 16,
    ):
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive; got {max_bytes}")
        self.byte_index = byte_index
        self.max_bytes = max_bytes
        self.data_root = data_root.rstrip("/")
        self._bytes_used = 0
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[Any, ...], tuple[Any, int]] = OrderedDict()
        self._header_cache: dict[int, bytes] = {}
        self._fetcher_cache: dict[int, RangeFetcher] = {}
        self._episodes: dict[int, _EpisodeEntry] = {}
        self._stats = CacheStats()
        self._executor = ThreadPoolExecutor(max_workers=max_prefetch_workers)

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(**{k: getattr(self._stats, k) for k in CacheStats.__dataclass_fields__})

    def submit_prefetch(self, ep_idx: int) -> None:
        # One future per (episode, camera): an episode's cameras fetch in parallel instead of
        # back-to-back on one thread, so the worker pool converts directly into concurrent
        # range GETs (the fetch throughput lever).
        with self._lock:
            if ep_idx in self._episodes:
                return
            entry = _EpisodeEntry()
            self._episodes[ep_idx] = entry
            self._stats.prefetch_submitted += 1
            for cam in self.byte_index.video_keys:
                entry.futures[cam] = self._executor.submit(self._prefetch_camera, ep_idx, cam, entry)

    def _prefetch_camera(self, ep_idx: int, cam: str, entry: _EpisodeEntry) -> None:
        try:
            entry.decoders[cam] = self._get_or_build_decoder(ep_idx, cam)
        except Exception as exc:
            entry.error = exc

    def ensure_ready(self, ep_idx: int) -> None:
        entry = self._episodes.get(ep_idx)
        if entry is None:
            raise KeyError(f"episode {ep_idx} not prefetched")
        pending = [f for f in entry.futures.values() if not f.done()]
        if pending:
            with self._lock:
                self._stats.prefetch_waits += 1
        for fut in entry.futures.values():
            fut.result()
        if entry.error is not None:
            raise entry.error

    def get_decoder(self, ep_idx: int, video_key: str) -> Any:
        entry = self._episodes[ep_idx]
        fut = entry.futures.get(video_key)
        if fut is not None:
            fut.result()
        if entry.error is not None:
            raise entry.error
        return entry.decoders[video_key]

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _get_or_build_decoder(self, ep_idx: int, cam: str) -> Any:
        key = (ep_idx, cam)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._stats.hits += 1
        if cached is not None:
            # Build the decoder outside the lock: self._lock is non-reentrant, and decoding
            # while holding it would also serialize every other fetch thread.
            payload, _ = cached
            t0 = time.perf_counter()
            dec = self._decoder_from_payload(payload, ep_idx, cam)
            with self._lock:
                self._stats.buffer_hit_decoder_s += time.perf_counter() - t0
            return dec

        payload, payload_bytes, dec = self._fetch_manifest_slice(ep_idx, cam)

        with self._lock:
            self._stats.misses += 1
            if payload_bytes > self.MAX_BYTES_PER_MISS:
                logger.warning(
                    "byte cache miss fetched %.1f MB (>25 MB) for ep=%s cam=%s",
                    payload_bytes / 1e6,
                    ep_idx,
                    cam,
                )
            self._evict_until(payload_bytes)
            self._cache[key] = (payload, payload_bytes)
            self._bytes_used += payload_bytes
        return dec

    def _fetch_manifest_slice(self, ep_idx: int, cam: str) -> tuple[SparseMp4Reader, int, Any]:
        lookup = self.byte_index.lookup(ep_idx, cam)
        file_info = self.byte_index.file_lookup(lookup.file_id)
        fetcher = self._get_fetcher(lookup.file_id, file_info.file_path)
        t_fetch = time.perf_counter()
        header = self._get_header_bytes(lookup.file_id, fetcher, file_info.header_length)
        lo = lookup.mdat_offset
        hi = lo + lookup.mdat_length - 1
        mdat = fetcher.fetch(lo, hi)
        fetch_s = time.perf_counter() - t_fetch
        nbytes = len(header) + len(mdat)
        with self._lock:
            self._stats.bytes_fetched += nbytes
            self._stats.mdat_slices += 1
            self._stats.fetch_to_buffer_s += fetch_s

        def lazy_fetch(pos: int, end: int) -> bytes:
            data = fetcher.fetch(pos, end)
            with self._lock:
                self._stats.bytes_fetched += len(data)
            return data

        reader = SparseMp4Reader(
            file_size=file_info.file_size,
            header=header,
            mdat_lo=lo,
            mdat_bytes=mdat,
            lazy_fetch=lazy_fetch,
        )
        t_init = time.perf_counter()
        dec = self._decoder_from_payload(reader, ep_idx, cam)
        self._validate_decoder(dec, lookup)
        init_s = time.perf_counter() - t_init
        with self._lock:
            self._stats.buffer_to_decoder_s += init_s
        self._rewind_payload(reader)
        return reader, nbytes, dec

    def _get_fetcher(self, file_id: int, rel_path: str) -> RangeFetcher:
        if file_id not in self._fetcher_cache:
            path = rel_path if rel_path.startswith("hf://") else f"{self.data_root}/{rel_path}"
            self._fetcher_cache[file_id] = RangeFetcher(path)
        return self._fetcher_cache[file_id]

    def _get_header_bytes(self, file_id: int, fetcher: RangeFetcher, header_length: int) -> bytes:
        if file_id in self._header_cache:
            return self._header_cache[file_id]
        hi = max(0, header_length - 1)
        header = fetcher.fetch(0, hi)
        with self._lock:
            self._header_cache[file_id] = header
            self._stats.bytes_fetched += len(header)
        return header

    def _decoder_from_payload(self, payload: SparseMp4Reader, ep_idx: int, cam: str) -> Any:
        # Lazy import: torchcodec_utils touches private torchcodec symbols that vary across
        # torchcodec versions; importing this module must not require them.
        from .torchcodec_utils import open_video_decoder

        payload.seek(0)
        mappings = self.byte_index.custom_frame_mappings(ep_idx, cam)
        return open_video_decoder(payload, frame_mappings=mappings)

    def _validate_decoder(self, dec: Any, lookup: EpisodeSliceLookup) -> None:
        begin = float(dec.metadata.begin_stream_seconds)
        end = float(dec.metadata.end_stream_seconds)
        duration = max(0.01, end - begin)
        for ts in (begin + 1e-3, begin + 0.5 * duration, end - 1e-3):
            _ = dec.get_frames_played_at([ts]).data

    def _rewind_payload(self, payload: SparseMp4Reader) -> None:
        payload.seek(0)

    def _evict_until(self, need: int) -> None:
        while self._bytes_used + need > self.max_bytes and self._cache:
            _, (_, size) = self._cache.popitem(last=False)
            self._bytes_used -= size
