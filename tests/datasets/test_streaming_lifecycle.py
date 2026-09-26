# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")

from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.streaming import episode_cache
from lerobot.streaming.range_fetch import ThreadLocalRangeFetcher


def bare_dataset() -> StreamingLeRobotDataset:
    dataset = object.__new__(StreamingLeRobotDataset)
    dataset.shuffle = True
    dataset.repeat = True
    dataset.seed = 47
    dataset.sampling_strategy = "remaining"
    dataset.max_num_shards = 1
    dataset.episode_pool_size = 2
    dataset.prefetch_episodes = 0
    dataset.byte_budget = 100
    dataset.decode_threads = 2
    dataset.decoded_queue_size = 3
    dataset._data_root = "/tmp"
    dataset._projected_columns = ["episode_index"]
    dataset._streaming_io_token = None
    dataset._rank_episodes = lambda: ([0, 1, 2, 3], 0, 1)
    dataset._episode_frame_count = lambda episode: 6
    dataset._make_video_cache = lambda *args: None
    dataset._load_episode_dataset = lambda *args: None
    dataset._make_episode_item = lambda data, ep, frame, **kwargs: {"index": ep * 6 + frame}
    dataset._apply_image_transforms = lambda item: None
    dataset.set_epoch(0)
    return dataset


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("strategy", ["remaining", "round_robin"])
def test_second_checkpoint_after_cross_epoch_resume(shuffle: bool, strategy: str) -> None:
    source = bare_dataset()
    source.shuffle = shuffle
    source.sampling_strategy = strategy
    source.load_state_dict({"epoch": 0, "offset": 32, "batch_size": 4})
    iterator = iter(source)
    next(iterator)
    state = source.state_dict()
    expected = [next(iterator)["index"] for _ in range(10)]
    iterator.close()
    restored = bare_dataset()
    restored.shuffle = shuffle
    restored.sampling_strategy = strategy
    restored.load_state_dict(state)
    resumed = iter(restored)
    actual = [next(resumed)["index"] for _ in range(10)]
    resumed.close()
    assert actual == expected, state


def test_setup_failure_closes_created_cache() -> None:
    dataset = bare_dataset()
    closed = []
    cache = SimpleNamespace(
        manifest=SimpleNamespace(episode_byte_size=lambda ep: 101),
        close=lambda: closed.append(True),
    )
    dataset._make_video_cache = lambda *args: cache
    with pytest.raises((ValueError, MemoryError)):
        next(iter(dataset))
    assert closed, "Created video cache never closed after oversized-episode rejection"


def test_repeating_iterator_closes_inner_iterator_after_first_sample() -> None:
    closed = []

    def once():
        try:
            yield {"index": 0}
        finally:
            closed.append(True)

    dataset = bare_dataset()
    inner = once()
    dataset._iter_once = lambda: inner
    iterator = iter(dataset)
    next(iterator)
    iterator.close()
    assert closed


def test_decoder_eviction_does_not_split_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        episode_cache, "make_range_fetcher", lambda *a, **k: SimpleNamespace(close=lambda: None)
    )
    manifest = SimpleNamespace(lookup=lambda *a: SimpleNamespace(source_start_pts=0))
    cache = episode_cache.EpisodeByteCache(manifest, "/tmp", max_open_decoders=1)
    cache._get_entry = lambda *a: {"bytes": b""}
    cache.retain_episode = lambda *a, **k: None
    cache.release_episode = lambda *a: None
    b_inside = threading.Event()
    a_has_decoder = threading.Event()
    continue_a = threading.Event()
    release_b = threading.Event()
    a_inside = threading.Event()
    active_lock = threading.Lock()
    active = 0
    high_water = 0

    class Decoder:
        metadata = SimpleNamespace(average_fps=30, num_frames=10)

        def get_frames_at(self, **kwargs):
            nonlocal active, high_water
            with active_lock:
                active += 1
                high_water = max(high_water, active)
            if threading.current_thread().name.startswith("reader-b"):
                b_inside.set()
                assert release_b.wait(5)
            if threading.current_thread().name.startswith("reader-a"):
                a_inside.set()
            with active_lock:
                active -= 1
            return SimpleNamespace(data=None)

    cache._open_decoder = lambda *a: Decoder()
    original = cache._decoder_for_frames

    def paused_lookup(*args):
        result = original(*args)
        if threading.current_thread().name.startswith("reader-a"):
            a_has_decoder.set()
            assert continue_a.wait(5)
        return result

    cache._decoder_for_frames = paused_lookup
    with (
        ThreadPoolExecutor(1, thread_name_prefix="reader-a") as a,
        ThreadPoolExecutor(1, thread_name_prefix="reader-b") as b,
    ):
        try:
            first = a.submit(cache._get_frames, 0, "camera", [0.0])
            assert a_has_decoder.wait(5)
            second = b.submit(cache._get_frames, 0, "camera", [0.0])
            assert b_inside.wait(5)
            cache.get_decoder(1, "camera")
            continue_a.set()
            a_inside.wait(0.1)
        finally:
            continue_a.set()
            release_b.set()
        first.result(timeout=5)
        second.result(timeout=5)
    cache.close()
    assert high_water == 1, f"Same decoder entered concurrently by {high_water} threads"


def test_source_handles_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handles = []
    reader = ThreadLocalRangeFetcher(tmp_path)
    original_open = reader.fs.open

    def open_source(*args, **kwargs):
        handle = original_open(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(reader.fs, "open", open_source)
    try:
        for index in range(256):
            path = tmp_path / f"{index}.mp4"
            path.write_bytes(b"abc")
            assert reader.read_range(path.name, 0, 1) == b"a"
        assert sum(not handle.closed for handle in handles) <= reader.max_open_files
    finally:
        reader.close()
    assert all(handle.closed for handle in handles)
