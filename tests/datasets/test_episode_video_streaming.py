#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from lerobot.streaming.episode_video import (
    EpisodeByteCache,
    EpisodeVideoManifest,
    ThreadLocalRangeFetcher,
    _log_http_failure,
)
from lerobot.streaming.mp4 import (
    _box,
    _co64,
    _dinf,
    _hdlr,
    _mdhd,
    _mvhd,
    _stco,
    _stsc_one_sample_per_chunk,
    _stss,
    _stsz,
    _stts,
    _tkhd,
    _vmhd,
    parse_mp4_index,
    synthesize_mp4,
)


def _minimal_mp4(sample_offsets: list[int], *, use_co64: bool = False) -> bytes:
    ftyp = _box(b"ftyp", b"isom\0\0\2\0isomiso2mp41")
    sizes = np.array([10, 10, 10], dtype=np.int64)
    durations = np.array([1000, 1000, 1000], dtype=np.int64)
    stsd_body = struct.pack(">II", 0, 1) + struct.pack(">I4s", 16, b"avc1") + b"\0" * 8
    offsets = _co64(sample_offsets) if use_co64 else _stco(sample_offsets)
    stbl = _box(
        b"stbl",
        _box(b"stsd", stsd_body)
        + _stts(durations)
        + _stsc_one_sample_per_chunk(len(sizes))
        + _stsz(sizes)
        + offsets
        + _stss(np.array([1], dtype=np.int64)),
    )
    minf = _box(b"minf", _vmhd() + _dinf() + stbl)
    mdia = _box(b"mdia", _mdhd(1000, 3000) + _hdlr() + minf)
    trak = _box(b"trak", _tkhd(1, 3000, 64, 48) + mdia)
    moov = _box(b"moov", _mvhd(1000, 3000, 2) + trak)
    mdat_payload_start = 10_000
    free_size = mdat_payload_start - 8 - len(ftyp) - len(moov)
    assert free_size >= 8
    free = _box(b"free", b"\0" * (free_size - 8))
    return ftyp + moov + free + _box(b"mdat", b"x" * 128)


def test_episode_slice_uses_min_max_sample_offsets_for_reordered_chunks():
    mp4 = parse_mp4_index("test.mp4", _minimal_mp4([10_000, 10_050, 10_025]))

    sample_slice = mp4.sample_slice(0.0, 2.0, keyframe_pad_s=0, keyframe_pad_fraction=0)

    assert sample_slice.byte_offset == 10_000
    assert sample_slice.byte_length == 60
    assert sample_slice.sample_lo == 0
    assert sample_slice.sample_hi == 2


def test_synthesized_mp4_rebases_one_chunk_per_sample_offsets():
    mp4 = parse_mp4_index("test.mp4", _minimal_mp4([10_000, 10_050, 10_025]))
    sample_slice = mp4.sample_slice(0.0, 2.0, keyframe_pad_s=0, keyframe_pad_fraction=0)

    mini = synthesize_mp4(mp4, sample_slice, b"x" * sample_slice.byte_length)
    mini_index = parse_mp4_index("mini.mp4", mini)

    expected = np.array([0, 50, 25], dtype=np.int64) + mini_index.mdat_payload_offset
    np.testing.assert_array_equal(mini_index.sample_offsets, expected)
    np.testing.assert_array_equal(mini_index.sample_sizes, np.array([10, 10, 10]))


def test_parser_accepts_co64_chunk_offsets():
    mp4 = parse_mp4_index("test.mp4", _minimal_mp4([10_000, 10_050, 10_025], use_co64=True))

    np.testing.assert_array_equal(mp4.sample_offsets, np.array([10_000, 10_050, 10_025]))


def _fake_cache(monkeypatch, tmp_path, *, byte_budget=8, max_open_decoders=1):
    manifest = EpisodeVideoManifest(video_keys=["camera"], files=[], spans={})
    cache = EpisodeByteCache(
        manifest,
        tmp_path,
        byte_budget=byte_budget,
        workers=1,
        open_decoders=False,
        max_open_decoders=max_open_decoders,
    )
    monkeypatch.setattr(
        cache,
        "_fetch_and_synthesize",
        lambda episode_index, _camera_key: {"bytes": bytes([episode_index]) * 5, "_timings": None},
    )
    return cache


def test_byte_cache_does_not_evict_retained_episode(monkeypatch, tmp_path):
    with _fake_cache(monkeypatch, tmp_path, byte_budget=10) as cache:
        cache.retain_episode(0)
        cache.ensure_ready(0)
        cache.ensure_ready(1)
        cache.ensure_ready(2)

        assert (0, "camera") in cache._cache
        assert (1, "camera") not in cache._cache
        assert cache.resident_bytes <= cache.byte_budget


def test_byte_cache_rejects_retained_set_larger_than_budget(monkeypatch, tmp_path):
    with _fake_cache(monkeypatch, tmp_path, byte_budget=4) as cache:
        cache.retain_episode(0)

        with pytest.raises(MemoryError, match="byte budget"):
            cache.ensure_ready(0)


def test_decoder_count_has_independent_limit(monkeypatch, tmp_path):
    opened = []

    class FakeDecoder:
        pass

    def open_decoder(_data):
        decoder = FakeDecoder()
        opened.append(decoder)
        return decoder

    monkeypatch.setattr("lerobot.streaming.episode_video.open_video_decoder", open_decoder)
    with _fake_cache(monkeypatch, tmp_path, byte_budget=20, max_open_decoders=1) as cache:
        first = cache.get_decoder(0, "camera")
        second = cache.get_decoder(1, "camera")

        assert first is not second
        assert cache.open_decoder_count == 1


def test_releasing_episode_allows_immediate_eviction(monkeypatch, tmp_path):
    with _fake_cache(monkeypatch, tmp_path, byte_budget=5) as cache:
        cache.retain_episode(0)
        cache.ensure_ready(0)
        cache.release_episode(0)
        cache.ensure_ready(1)

        assert (0, "camera") not in cache._cache
        assert (1, "camera") in cache._cache


def test_range_fetcher_closes_handles_from_all_worker_threads(tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"0123456789")
    fetcher = ThreadLocalRangeFetcher(tmp_path)
    barrier = threading.Barrier(2)

    def read_from_worker(offset):
        barrier.wait()
        return fetcher.read_range("video.mp4", offset, 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(read_from_worker, offset) for offset in range(2)]
        assert [future.result() for future in futures] == [b"0", b"1"]

        handles = list(fetcher._all_handles.values())
    assert len(handles) == 2
    fetcher.close()

    assert not fetcher._all_handles
    assert all(handle.closed for handle in handles)


def test_http_failure_log_does_not_write_credentials(tmp_path, monkeypatch):
    log_path = tmp_path / "http-failures.jsonl"
    monkeypatch.setenv("LEROBOT_HTTP_FAILURE_LOG", str(log_path))

    _log_http_failure(
        backend="native-http",
        method="GET",
        url="https://cdn.example/private/video.mp4?token=url-secret",
        headers={
            "Authorization": "Bearer header-secret",
            "Range": "bytes=0-10",
            "X-Request-Id": "safe-request-id",
        },
        elapsed_s=0.1,
        status_code=403,
    )

    record = json.loads(log_path.read_text())
    assert record["host"] == "cdn.example"
    assert record["path"] == "/private/video.mp4"
    assert record["request_id"] == "safe-request-id"
    assert "secret" not in log_path.read_text()
