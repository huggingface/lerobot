"""Byte reservations cover prefetch and the final in-flight episode decode."""

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.streaming.episode_cache import EpisodeByteCache
from lerobot.streaming.episode_pool import ExactCoveragePool
from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import parse_mp4_index, synthesized_mp4_size
from tests.datasets.test_episode_video_streaming import _minimal_mp4


@pytest.fixture
def bounded_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[EpisodeByteCache]:
    manifest = EpisodeVideoManifest(video_keys=["left", "right"], files=[], spans={})
    monkeypatch.setattr(manifest, "episode_byte_size", lambda _episode: 10)
    with EpisodeByteCache(manifest, tmp_path, byte_budget=10, workers=2, open_decoders=False) as cache:
        monkeypatch.setattr(cache, "_fetch_and_synthesize", lambda ep, cam: {"bytes": bytes([ep]) * 5})
        yield cache


def test_inflight_prefetch_reserves_all_cameras(
    bounded_cache: EpisodeByteCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    finish = threading.Event()

    def fetch(episode: int, camera: str) -> dict[str, bytes]:
        started.set()
        assert finish.wait(5)
        return {"bytes": bytes([episode]) * 5}

    monkeypatch.setattr(bounded_cache, "_fetch_and_synthesize", fetch)
    try:
        assert bounded_cache.submit_prefetch(0)
        assert started.wait(5)
        assert bounded_cache.reserved_bytes == 10
        assert bounded_cache.resident_bytes == 0
        assert not bounded_cache.submit_prefetch(1)
    finally:
        finish.set()
    bounded_cache.ensure_ready(0)
    assert bounded_cache.resident_bytes == bounded_cache.reserved_bytes == 10


def test_waiting_admission_wakes_on_decode_release(bounded_cache: EpisodeByteCache) -> None:
    bounded_cache.retain_episode(0)
    bounded_cache.ensure_ready(0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(bounded_cache.retain_episode, 1, wait=True)
        try:
            assert not pending.done()
        finally:
            bounded_cache.release_episode(0)
        pending.result(timeout=5)
    bounded_cache.ensure_ready(1)
    assert bounded_cache.resident_bytes == bounded_cache.reserved_bytes == 10
    assert set(bounded_cache._cache) == {(1, "left"), (1, "right")}
    bounded_cache.release_episode(1)


def test_close_unblocks_waiting_admission(bounded_cache: EpisodeByteCache) -> None:
    bounded_cache.retain_episode(0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(bounded_cache.retain_episode, 1, wait=True)
        bounded_cache.close()
        with pytest.raises(RuntimeError, match="closed"):
            pending.result(timeout=5)
    assert bounded_cache.resident_bytes == bounded_cache.reserved_bytes == 0


def test_prefetch_error_propagates_and_reservation_can_be_reused(
    bounded_cache: EpisodeByteCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fetch(episode: int, camera: str) -> dict[str, bytes]:
        if episode == 0:
            raise OSError("injected range read failure")
        return {"bytes": bytes([episode]) * 5}

    monkeypatch.setattr(bounded_cache, "_fetch_and_synthesize", fetch)
    bounded_cache.retain_episode(0)
    with pytest.raises(OSError, match="injected"):
        bounded_cache.ensure_ready(0)
    bounded_cache.release_episode(0)
    bounded_cache.retain_episode(1, wait=True)
    bounded_cache.ensure_ready(1)
    assert bounded_cache.resident_bytes == bounded_cache.reserved_bytes == 10
    bounded_cache.release_episode(1)


def test_concurrent_camera_reads_share_one_reservation(bounded_cache: EpisodeByteCache) -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        requests = [(0, "left"), (0, "right")] * 8
        futures = [executor.submit(bounded_cache.get_bytes, *request) for request in requests]
        assert all(future.result(timeout=5) == bytes(5) for future in futures)
    assert bounded_cache.resident_bytes == bounded_cache.reserved_bytes == 10


def test_prefetch_reserves_bytes_before_fetch(tmp_path: Path) -> None:
    source = _minimal_mp4([10_000, 10_050, 10_025])
    (tmp_path / "video.mp4").write_bytes(source)
    index = parse_mp4_index("video.mp4", source)
    span = index.sample_slice(0, 2, keyframe_pad_s=0, keyframe_pad_fraction=0)
    columns = {
        "file_id": 0,
        "mdat_offset": span.byte_offset,
        "mdat_length": span.byte_length,
        "first_pts": 0.0,
        "last_pts": 2.0,
        "frame_count": 3,
        "sample_lo": span.sample_lo,
        "sample_hi": span.sample_hi,
        "source_start_pts": span.source_start_pts,
    }
    manifest = EpisodeVideoManifest(
        video_keys=["camera"],
        files=[VideoFileRecord("video.mp4", len(source), index)],
        spans={key: np.full((3, 1), value) for key, value in columns.items()},
    )
    budget = synthesized_mp4_size(index, span)
    with EpisodeByteCache(manifest, tmp_path, byte_budget=budget, workers=1) as cache:
        cache.retain_episode(0)
        for episode in range(3):
            cache.submit_prefetch(episode)
        for future in list(cache._futures.values()):
            future.result(timeout=5)
        # Inspect both containers independently of the reported counter. Old futures own
        # their payloads and bypass _cache entirely.
        payloads = {id(entry["bytes"]): entry["bytes"] for entry in cache._cache.values()}
        for future in cache._futures.values():
            result = future.result()
            if result is not None:
                payloads[id(result["bytes"])] = result["bytes"]
        assert sum(map(len, payloads.values())) <= budget
        assert cache.resident_bytes == budget


@pytest.mark.parametrize("prefetch", [0, 2])
@pytest.mark.parametrize("decode_threads,queue_size", [(1, 1), (2, 2), (4, 8)])
@pytest.mark.parametrize("sampling_strategy", ["remaining", "round_robin"])
def test_rotation_keeps_bytes_until_last_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefetch: int,
    decode_threads: int,
    queue_size: int,
    sampling_strategy: str,
) -> None:
    manifest = EpisodeVideoManifest(video_keys=["camera"], files=[], spans={})
    monkeypatch.setattr(manifest, "episode_byte_size", lambda episode: 60)
    cache = EpisodeByteCache(manifest, tmp_path, byte_budget=60, workers=1, open_decoders=False)
    monkeypatch.setattr(cache, "_fetch_and_synthesize", lambda ep, cam: {"bytes": bytes([ep]) * 60})
    ds = StreamingLeRobotDataset.__new__(StreamingLeRobotDataset)
    ds.repeat = False
    ds.shuffle = True
    ds._next_epoch = ds._resume_offset = ds._state_offset = 0
    ds.seed = 42
    ds.sampling_strategy = sampling_strategy
    ds.max_num_shards = ds.episode_pool_size = 1
    ds.prefetch_episodes = prefetch
    ds.byte_budget = 60
    ds.decode_threads = decode_threads
    ds.decoded_queue_size = queue_size
    ds._data_root = str(tmp_path)
    ds._streaming_io_token = None
    ds._projected_columns = ("episode_index",)
    monkeypatch.setattr(ds, "_rank_episodes", lambda: ([0, 1, 2], 0, 1))
    monkeypatch.setattr(ds, "_episode_frame_count", lambda ep: 1)
    monkeypatch.setattr(ds, "_make_video_cache", lambda *args: cache)
    monkeypatch.setattr(ds, "_load_episode_dataset", lambda *args: None)
    monkeypatch.setattr(ds, "_apply_image_transforms", lambda item: None)
    first_decoding = threading.Event()
    finish_first = threading.Event()
    admission_attempted = threading.Event()
    admission_finished = threading.Event()
    first_episode = ExactCoveragePool([(0, 1), (1, 1), (2, 1)], 1, seed=42).resident[0]
    retain = cache.retain_episode

    def retain_with_rotation_gate(episode: int, *, wait: bool = False) -> None:
        rotation = episode != first_episode and not finish_first.is_set()
        if rotation:
            assert first_decoding.wait(5)
            admission_attempted.set()
        retain(episode, wait=wait)
        if rotation:
            admission_finished.set()

    monkeypatch.setattr(cache, "retain_episode", retain_with_rotation_gate)
    attempted: list[int] = []

    def decode(_: object, episode: int, frame_index: int, **kwargs: Any) -> dict[str, torch.Tensor]:
        attempted.append(episode)
        data = cache.get_bytes(episode, "camera")
        if len(attempted) == 1:
            first_decoding.set()
            assert finish_first.wait(5)
        assert data == bytes([episode]) * 60
        return {"index": torch.tensor(episode)}

    monkeypatch.setattr(ds, "_make_episode_item", decode)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(lambda: [int(item["index"]) for item in ds])
        try:
            assert first_decoding.wait(5)
            assert admission_attempted.wait(5)
            assert not admission_finished.wait(0.05)
        finally:
            finish_first.set()
        expected = [episode for episode, _ in ExactCoveragePool([(0, 1), (1, 1), (2, 1)], 1, seed=42)]
        assert result.result(timeout=10) == expected
