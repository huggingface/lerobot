#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Unit tests for ``lerobot.datasets.video_utils.PyavCache``.

Cover the LRU bounding + container-release behaviour that lets the pyav fallback
reuse open containers across calls (skipping ``av.open`` header parse + codec
init) without unbounded FD/RAM growth over datasets with many distinct files.
"""

import io
import types
from pathlib import Path

import pytest
import torch

import lerobot.datasets.video_utils as video_utils
from lerobot.datasets.video_utils import (
    DEFAULT_PYAV_CACHE_SIZE,
    PyavCache,
    decode_video_frames_pyav,
)

TEST_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "encoded_videos"
SRC_CLIP = TEST_ARTIFACTS_DIR / "clip_4frames.mp4"


class _FakeContainer:
    """Minimal stand-in for an av container: exposes a video stream, tracks close()."""

    def __init__(self):
        self.closed = False
        self.streams = types.SimpleNamespace(video=[object()])

    def close(self):
        self.closed = True


class _SpyOpen:
    """Context manager mimicking ``fsspec.open`` over in-memory bytes."""

    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return io.BytesIO(self._data)

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_av(monkeypatch):
    """Replace ``av.open`` with a fake returning fresh :class:`_FakeContainer` instances.

    Lets the LRU mechanics be tested without decoding real files.
    """
    monkeypatch.setattr(video_utils.av, "open", lambda *a, **k: _FakeContainer())


class TestPyavCacheBounded:
    def test_default_cache_is_bounded(self):
        """The default cache must have a finite ``max_size`` to bound FD/RSS growth."""
        cache = PyavCache()
        assert cache.max_size == DEFAULT_PYAV_CACHE_SIZE
        assert cache.max_size is not None and cache.max_size > 0

    def test_size_capped_and_evicts_lru(self, fake_av):
        """Re-accessing promotes to MRU; the LRU entry is evicted once over the cap."""
        cache = PyavCache(max_size=2)
        cache.get_container("a")
        cache.get_container("b")
        cache.get_container("a")  # promote "a"; "b" is now LRU
        cache.get_container("c")  # evict "b"
        assert cache.size() == 2
        assert "a" in cache and "c" in cache
        assert "b" not in cache

    def test_eviction_closes_container(self, fake_av):
        """Evicting an entry must close its container (otherwise we leak FDs)."""
        cache = PyavCache(max_size=1)
        cache.get_container("a")
        evicted = cache._cache["a"][1]
        assert evicted.closed is False
        cache.get_container("b")  # forces eviction of "a"
        assert evicted.closed is True

    def test_clear_closes_all_containers(self, fake_av):
        cache = PyavCache(max_size=10)
        for p in ("a", "b", "c"):
            cache.get_container(p)
        containers = [entry[1] for entry in cache._cache.values()]
        cache.clear()
        assert cache.size() == 0
        assert all(c.closed for c in containers)

    def test_hit_returns_same_and_does_not_evict(self, fake_av):
        cache = PyavCache(max_size=2)
        first = cache.get_container("a")
        second = cache.get_container("a")
        assert first is second
        assert cache.size() == 1

    def test_unbounded_when_max_size_none(self, fake_av):
        cache = PyavCache(max_size=None)
        for p in ("a", "b", "c", "d"):
            cache.get_container(p)
        assert cache.size() == 4

    def test_env_var_overrides_default(self, fake_av, monkeypatch):
        monkeypatch.setenv("LEROBOT_PYAV_CACHE_SIZE", "3")
        cache = PyavCache()
        assert cache.max_size == 3
        for p in ("a", "b", "c", "d", "e"):
            cache.get_container(p)
        assert cache.size() == 3


class TestPyavCacheByteBudget:
    def test_disabled_by_default(self):
        """Byte budget is opt-in: off unless configured."""
        assert PyavCache().byte_budget is None

    def test_remote_evicts_over_budget_but_keeps_one(self, fake_av, monkeypatch):
        """Remote clips buffered in RAM weigh against the budget; local ones don't."""
        data = SRC_CLIP.read_bytes()
        monkeypatch.setattr(video_utils.fsspec, "open", lambda *a, **k: _SpyOpen(data))

        cache = PyavCache(max_size=None, byte_budget=len(data))
        for i in range(4):
            cache.get_container(f"s3://bucket/clip_{i:04d}.mp4")
        assert cache.size() == 1

    def test_local_files_are_weightless(self, fake_av):
        cache = PyavCache(max_size=None, byte_budget=1)  # tiniest possible budget
        for p in ("a", "b", "c", "d"):
            cache.get_container(p)
        assert cache.size() == 4

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("LEROBOT_PYAV_CACHE_BYTES", "12345")
        assert PyavCache().byte_budget == 12345
        monkeypatch.setenv("LEROBOT_PYAV_CACHE_BYTES", "none")
        assert PyavCache().byte_budget is None


class TestPyavCacheDecode:
    """Real-decode tests guarding container reuse (seek-state) correctness."""

    def test_cached_matches_fresh_open_across_reuse(self):
        """Cached decode equals a fresh ad-hoc open, and reuse doesn't corrupt state."""
        timestamps = [0.0]
        tolerance_s = 1.0  # generous: nearest-frame match on a 4-frame clip

        # Fresh open via a raw file-like object hits the uncached (else) branch.
        with open(SRC_CLIP, "rb") as f:
            fresh = decode_video_frames_pyav(f, timestamps, tolerance_s)

        cache = PyavCache()
        first = decode_video_frames_pyav(str(SRC_CLIP), timestamps, tolerance_s, container_cache=cache)
        # Second call reuses the cached container (re-seek on a used container).
        second = decode_video_frames_pyav(str(SRC_CLIP), timestamps, tolerance_s, container_cache=cache)

        assert torch.equal(first, fresh)
        assert torch.equal(second, fresh)
        assert cache.size() == 1

    def test_reuses_cached_container_object(self):
        cache = PyavCache()
        decode_video_frames_pyav(str(SRC_CLIP), [0.0], 1.0, container_cache=cache)
        resource_first = cache._cache[str(SRC_CLIP)][0]
        decode_video_frames_pyav(str(SRC_CLIP), [0.0], 1.0, container_cache=cache)
        resource_second = cache._cache[str(SRC_CLIP)][0]
        assert resource_first is resource_second
