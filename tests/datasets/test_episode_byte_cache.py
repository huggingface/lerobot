# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for EpisodeByteCache fetch concurrency (decoder layer stubbed out)."""

import threading
import time
from dataclasses import dataclass

import pytest

from lerobot.datasets.episode_byte_cache import EpisodeByteCache, RangeFetcher


@dataclass
class _FakeLookup:
    file_id: int
    mdat_offset: int
    mdat_length: int


@dataclass
class _FakeFileInfo:
    file_path: str
    file_size: int
    header_length: int


class _FakeByteIndex:
    """Two cameras, one file per (episode, cam); fetch delay is injectable."""

    def __init__(self, tmp_path, num_episodes=4, mdat_len=64):
        self.video_keys = ["cam0", "cam1"]
        self._files = {}
        self._lookups = {}
        file_id = 0
        for ep in range(num_episodes):
            for cam in self.video_keys:
                path = tmp_path / f"ep{ep}_{cam}.bin"
                payload = bytes([ep]) * 16 + bytes(range(64)) * ((mdat_len + 63) // 64)
                path.write_bytes(payload)
                self._files[file_id] = _FakeFileInfo(str(path), len(payload), header_length=16)
                self._lookups[(ep, cam)] = _FakeLookup(file_id, mdat_offset=16, mdat_length=mdat_len)
                file_id += 1

    def lookup(self, ep_idx, cam):
        return self._lookups[(ep_idx, cam)]

    def file_lookup(self, file_id):
        return self._files[file_id]

    def custom_frame_mappings(self, ep_idx, cam):
        return None


class _SlowFetchCache(EpisodeByteCache):
    """Stub decode/validation; add a per-fetch delay to observe fetch parallelism."""

    fetch_delay_s = 0.15

    def _fetch_manifest_slice(self, ep_idx, cam):
        time.sleep(self.fetch_delay_s)
        return f"payload-{ep_idx}-{cam}", 32, f"decoder-{ep_idx}-{cam}"

    def _decoder_from_payload(self, payload, ep_idx, cam):
        return f"decoder-{ep_idx}-{cam}"


def _make_cache(tmp_path, **kwargs):
    index = _FakeByteIndex(tmp_path)
    return _SlowFetchCache(index, max_bytes=10_000_000, data_root=str(tmp_path), **kwargs)


def test_cameras_fetch_in_parallel(tmp_path):
    """An episode's cameras must not fetch back-to-back on one thread."""
    cache = _make_cache(tmp_path, max_prefetch_workers=8)
    start = time.perf_counter()
    for ep in range(4):
        cache.submit_prefetch(ep)
    for ep in range(4):
        cache.ensure_ready(ep)
    elapsed = time.perf_counter() - start
    # 4 episodes x 2 cams x 0.15s = 1.2s sequential; 8 workers -> one wave ~0.15s.
    assert elapsed < 0.6, f"fetches serialized: {elapsed:.2f}s for 8 fetches on 8 workers"
    assert cache.get_decoder(2, "cam1") == "decoder-2-cam1"
    cache.close()


def test_prefetch_error_propagates(tmp_path):
    cache = _make_cache(tmp_path, max_prefetch_workers=2)

    def boom(ep_idx, cam):
        raise RuntimeError("fetch failed")

    cache._fetch_manifest_slice = boom
    cache.submit_prefetch(0)
    with pytest.raises(RuntimeError, match="fetch failed"):
        cache.ensure_ready(0)
    with pytest.raises(RuntimeError, match="fetch failed"):
        cache.get_decoder(0, "cam0")
    cache.close()


def test_payload_cache_hit_does_not_deadlock(tmp_path):
    """Regression: the hit path used to re-acquire the non-reentrant lock (deadlock)."""
    cache = _make_cache(tmp_path, max_prefetch_workers=2)
    cache._cache[(0, "cam0")] = ("payload-0-cam0", 32)
    cache._bytes_used = 32

    result = {}

    def hit():
        result["dec"] = cache._get_or_build_decoder(0, "cam0")

    thread = threading.Thread(target=hit, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "cache-hit path deadlocked"
    assert result["dec"] == "decoder-0-cam0"
    assert cache.stats.hits == 1
    cache.close()


def test_range_fetcher_cat_file_correctness(tmp_path):
    payload = bytes(range(256)) * 4
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    fetcher = RangeFetcher(str(path))
    assert fetcher.fetch(0, 15) == payload[0:16]
    assert fetcher.fetch(100, 355) == payload[100:356]
    assert fetcher.fetch(len(payload) - 4, len(payload) - 1) == payload[-4:]
    assert fetcher.fetch(10, 9) == b""


def test_ensure_ready_unknown_episode_raises(tmp_path):
    cache = _make_cache(tmp_path)
    with pytest.raises(KeyError):
        cache.ensure_ready(99)
    cache.close()
