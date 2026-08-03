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

"""Unit tests for ``lerobot.datasets.video_utils.VideoDecoderCache``.

These cover the LRU bounding + file-handle release behaviour added to prevent
unbounded growth when iterating over datasets with many distinct video files
(observed: ~35 GB anon-rss per DataLoader worker on an 8 k-file dataset).
"""

import shutil
from pathlib import Path

import pytest

pytest.importorskip("torchcodec", reason="torchcodec is required (install lerobot[dataset])")

from lerobot.datasets.video_utils import VideoDecoderCache  # noqa: E402

TEST_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "encoded_videos"
SRC_CLIP = TEST_ARTIFACTS_DIR / "clip_4frames.mp4"


def _make_distinct_clips(tmp_path: Path, n: int) -> list[Path]:
    """Copy the small reference mp4 to ``n`` distinct paths.

    The cache keys on absolute path, so distinct paths force distinct cache entries
    even though the file contents are identical.
    """
    assert SRC_CLIP.exists(), f"missing test artifact {SRC_CLIP}"
    paths = []
    for i in range(n):
        dst = tmp_path / f"clip_{i:04d}.mp4"
        shutil.copyfile(SRC_CLIP, dst)
        paths.append(dst)
    return paths


class TestVideoDecoderCacheBounded:
    def test_default_cache_is_bounded(self):
        """The default cache must have a finite ``max_size`` to bound RSS growth."""
        cache = VideoDecoderCache()
        assert cache.max_size is not None, "default cache must be bounded"
        assert cache.max_size > 0

    def test_size_capped_at_max_size(self, tmp_path):
        """``get_decoder`` for >``max_size`` distinct paths must NOT grow without bound."""
        paths = _make_distinct_clips(tmp_path, n=5)
        cache = VideoDecoderCache(max_size=2)
        for p in paths:
            cache.get_decoder(p)
        assert cache.size() == 2

    def test_evicts_least_recently_used(self, tmp_path):
        """Re-accessing an entry must promote it; the LRU entry is the one evicted."""
        paths = _make_distinct_clips(tmp_path, n=3)
        cache = VideoDecoderCache(max_size=2)

        cache.get_decoder(paths[0])
        cache.get_decoder(paths[1])
        cache.get_decoder(paths[0])  # promote paths[0] to MRU; paths[1] is now LRU
        cache.get_decoder(paths[2])  # should evict paths[1]

        assert str(paths[0]) in cache  # MRU stays
        assert str(paths[1]) not in cache  # LRU evicted
        assert str(paths[2]) in cache  # newest stays

    def test_eviction_closes_file_handle(self, tmp_path):
        """Evicting an entry must close its fsspec file handle (otherwise we leak FDs)."""
        paths = _make_distinct_clips(tmp_path, n=2)
        cache = VideoDecoderCache(max_size=1)

        cache.get_decoder(paths[0])
        # Reach into the cache to capture the handle before it is evicted. This is
        # the only assertion in the suite that touches a private attribute, and it
        # is the most direct way to prove the file descriptor is actually released.
        evicted_handle = cache._cache[str(paths[0])][1]
        assert evicted_handle.closed is False

        cache.get_decoder(paths[1])  # forces eviction of paths[0]

        assert evicted_handle.closed is True

    def test_clear_closes_all_file_handles(self, tmp_path):
        """``clear()`` must close every cached file handle."""
        paths = _make_distinct_clips(tmp_path, n=3)
        cache = VideoDecoderCache(max_size=10)

        for p in paths:
            cache.get_decoder(p)
        handles = [entry[1] for entry in cache._cache.values()]
        assert all(not h.closed for h in handles)

        cache.clear()

        assert cache.size() == 0
        assert all(h.closed for h in handles)

    def test_hit_does_not_reopen_or_evict(self, tmp_path):
        """A cache hit must return the same decoder instance without touching the cap."""
        paths = _make_distinct_clips(tmp_path, n=1)
        cache = VideoDecoderCache(max_size=2)

        first = cache.get_decoder(paths[0])
        second = cache.get_decoder(paths[0])

        assert first is second
        assert cache.size() == 1

    def test_unbounded_when_max_size_none(self, tmp_path):
        """``max_size=None`` preserves the legacy unbounded behaviour."""
        paths = _make_distinct_clips(tmp_path, n=4)
        cache = VideoDecoderCache(max_size=None)
        for p in paths:
            cache.get_decoder(p)
        assert cache.size() == 4

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        """``LEROBOT_VIDEO_DECODER_CACHE_SIZE`` env var sets the default ``max_size``."""
        monkeypatch.setenv("LEROBOT_VIDEO_DECODER_CACHE_SIZE", "3")
        cache = VideoDecoderCache()
        assert cache.max_size == 3

        paths = _make_distinct_clips(tmp_path, n=5)
        for p in paths:
            cache.get_decoder(p)
        assert cache.size() == 3
