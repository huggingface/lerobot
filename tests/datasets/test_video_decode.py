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

"""Unit tests for video frame decoding in ``lerobot.datasets.video_utils``.

Covers two things:
- each backend (pyav, torchcodec) selects the correct frames, in order, verified against an
  oracle independent of the decoder: the committed ``indexed_clip_60frames.mp4`` artifact whose
  frame ``i`` is painted the constant value ``i``, so decoded content equals the known index;
- ``VideoDecoderCache`` LRU bounding + file-handle release (torchcodec decode path).
"""

import importlib.util
import shutil
from pathlib import Path

import pytest

pytest.importorskip("av", reason="av is required (install lerobot[dataset])")

import torch  # noqa: E402

from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames  # noqa: E402

FPS = 30

TEST_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "encoded_videos"
SRC_CLIP = TEST_ARTIFACTS_DIR / "clip_4frames.mp4"
# 60-frame 16x16 lossless-RGB clip (libx264rgb, qp=0, gop=2) with frame ``i`` painted the constant
# value ``i``. Lossless RGB avoids the YUV<->RGB range shift, so decoded pixels equal the frame index exactly.
INDEXED_CLIP = TEST_ARTIFACTS_DIR / "indexed_clip_60frames.mp4"

torchcodec_required = pytest.mark.skipif(
    importlib.util.find_spec("torchcodec") is None,
    reason="torchcodec not available",
)


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


@pytest.mark.parametrize(
    "backend",
    ["pyav", pytest.param("torchcodec", marks=torchcodec_required)],
)
@pytest.mark.parametrize(
    "indices",
    [
        [3, 50, 20, 58, 10],  # far apart, unsorted -> a seek per keyframe cluster
        list(range(20, 31)),  # contiguous window -> single forward pass
        [5, 6, 7, 40, 41, 42],  # two contiguous clusters far apart
        [58, 40, 20, 0],  # descending
        [30, 30, 31],  # duplicates
        [42],  # single frame
    ],
)
def test_decode_selects_correct_frames(backend, indices):
    """Each backend returns exactly the requested frames, in order.

    ``INDEXED_CLIP`` paints frame ``i`` the constant value ``i``, so the expected content is the
    frame index itself -- an oracle independent of the decoder under test.
    """
    tolerance_s = 1.0 / FPS
    frames = decode_video_frames(
        INDEXED_CLIP, [i / FPS for i in indices], tolerance_s, backend, return_uint8=True
    )
    expected = torch.tensor([i % 256 for i in indices], dtype=torch.uint8).view(-1, 1, 1, 1).expand_as(frames)
    assert frames.shape[0] == len(indices)
    assert torch.equal(frames, expected)


@torchcodec_required
class TestVideoDecoderCacheBounded:
    """LRU bounding + file-handle release, added to prevent unbounded growth when iterating over
    datasets with many distinct video files (observed: ~35 GB anon-rss per DataLoader worker on an
    8 k-file dataset)."""

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
