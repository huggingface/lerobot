# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.machinery import ModuleSpec
from threading import Barrier, Event
from types import ModuleType, SimpleNamespace

import pytest
import torch

from lerobot.datasets.video_utils import VideoDecoderCache, decode_video_frames_torchcodec


@pytest.fixture
def fake_decoders(monkeypatch):
    # These tests exercise cache synchronization, not native video decoding.
    package = ModuleType("torchcodec")
    package.__spec__ = ModuleSpec("torchcodec", loader=None, is_package=True)
    decoders = ModuleType("torchcodec.decoders")
    package.decoders = decoders
    monkeypatch.setitem(sys.modules, "torchcodec", package)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)
    return decoders


@pytest.mark.parametrize("operation", ["evict", "clear"])
def test_decode_keeps_file_open_until_read_finishes(tmp_path, monkeypatch, operation, fake_decoders):

    reading = Event()
    resume = Event()
    mutation_started = Event()
    mutation_done = Event()

    class Decoder:
        def __init__(self, handle, **kwargs):
            self.handle = handle
            self.metadata = SimpleNamespace(average_fps=30)

        def get_frames_at(self, indices):
            reading.set()
            if not resume.wait(5):
                raise TimeoutError("test did not release frame read")
            self.handle.seek(0)
            return SimpleNamespace(
                data=torch.zeros(len(indices), 3, 2, 2, dtype=torch.uint8),
                pts_seconds=torch.tensor(indices, dtype=torch.float64) / 30,
            )

    monkeypatch.setattr(fake_decoders, "VideoDecoder", Decoder, raising=False)
    paths = [tmp_path / name for name in ("a.mp4", "b.mp4")]
    for path in paths:
        path.write_bytes(b"video")
    cache = VideoDecoderCache(max_size=1)

    def mutate():
        mutation_started.set()
        if operation == "clear":
            cache.clear()
        else:
            cache.get_decoder(paths[1])
        mutation_done.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            read = pool.submit(decode_video_frames_torchcodec, paths[0], [0.0], 0.01, decoder_cache=cache)
            try:
                assert reading.wait(5)
                handle = cache._cache[str(paths[0])][1]
                mutation = pool.submit(mutate)
                assert mutation_started.wait(5)
                # Eviction and clear must not wait for the reader; they hand
                # the handle to it instead, so the mutation completes at once
                # while the file stays open until the read finishes.
                assert mutation_done.wait(5)
                assert not handle.closed
            finally:
                resume.set()
            assert read.result(timeout=5).shape == (1, 3, 2, 2)
            mutation.result(timeout=5)
        assert handle.closed
        assert cache.size() == (0 if operation == "clear" else 1)
    finally:
        resume.set()
        cache.clear()


def test_decode_error_releases_cache_lock(tmp_path, monkeypatch, fake_decoders):

    class Decoder:
        def __init__(self, handle, **kwargs):
            self.metadata = SimpleNamespace(average_fps=30)

        def get_frames_at(self, indices):
            raise ValueError("decode failed")

    monkeypatch.setattr(fake_decoders, "VideoDecoder", Decoder, raising=False)
    path = tmp_path / "video.mp4"
    path.write_bytes(b"video")
    cache = VideoDecoderCache(max_size=1)
    with pytest.raises(ValueError, match="decode failed"):
        decode_video_frames_torchcodec(path, [0.0], 0.01, decoder_cache=cache)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(cache.clear).result(timeout=5)
    assert cache.size() == 0

def test_distinct_videos_decode_concurrently(tmp_path, monkeypatch, fake_decoders):
    both_decoding = Barrier(2, timeout=5)

    class Decoder:
        def __init__(self, handle, **kwargs):
            self.handle = handle
            self.metadata = SimpleNamespace(average_fps=30)

        def get_frames_at(self, indices):
            # A cache-wide lock held for the whole decode keeps the second reader
            # out and trips the barrier timeout.
            both_decoding.wait()
            return SimpleNamespace(
                data=torch.zeros(len(indices), 3, 2, 2, dtype=torch.uint8),
                pts_seconds=torch.tensor(indices, dtype=torch.float64) / 30,
            )

    monkeypatch.setattr(fake_decoders, "VideoDecoder", Decoder, raising=False)
    paths = [tmp_path / name for name in ("a.mp4", "b.mp4")]
    for path in paths:
        path.write_bytes(b"video")
    cache = VideoDecoderCache(max_size=4)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(decode_video_frames_torchcodec, path, [0.0], 0.01, decoder_cache=cache)
            for path in paths
        ]
        for future in futures:
            future.result(timeout=10)

    assert cache.size() == 2
