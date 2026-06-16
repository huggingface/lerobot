"""Acceptance tests for manifest byte-index sidecars.

Run on a compute node (not login-node):

  srun --partition=hopper-dev --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:30:00 \\
    bash -lc 'cd /admin/home/pepijn/lerobot && conda run --no-capture-output -n lerobot \\
      env -u HF_HUB_ENABLE_HF_TRANSFER python -m pytest tests/datasets/test_byte_index.py -m integration -v'
"""

from __future__ import annotations

import json
import socket

import pytest

pytest.importorskip("torchcodec")

REPO = "allenai/MolmoAct2-BimanualYAM-Dataset"
REV = "e9f21ae15074330839f2ac25ed4b49d76dfa1f9c"
BUCKET = "hf://buckets/pepijn223/MolmoAct2-BimanualYAM-Dataset-bucket"
MAX_EPISODES = 64

COMPUTE_NODE = pytest.mark.skipif(
    "login" in socket.gethostname(),
    reason="run on compute node via srun (see module docstring), not login-node",
)


@pytest.fixture(scope="module")
def byte_index_dir(tmp_path_factory):
    from lerobot.datasets.byte_index_builder import build_byte_index_tables, write_byte_index
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    out = tmp_path_factory.mktemp("byte_index")
    meta = LeRobotDatasetMetadata(REPO, revision=REV)
    files, episodes, _ = build_byte_index_tables(
        meta, BUCKET, workers=4, max_episodes=MAX_EPISODES, include_keyframes=False
    )
    write_byte_index(out, files, episodes, None, merge_existing=False)
    return out, meta


@pytest.mark.integration
@COMPUTE_NODE
def test_index_load_fast_and_small(byte_index_dir):
    from lerobot.datasets.byte_index import EpisodeByteIndex

    out, meta = byte_index_dir
    index = EpisodeByteIndex(out, video_keys=meta.video_keys, num_episodes=MAX_EPISODES)
    assert index.load_time_s < 1.0
    assert index.resident_bytes < 1_000_000_000


@pytest.mark.integration
@COMPUTE_NODE
def test_tight_fetch_under_25mb(byte_index_dir):
    from lerobot.datasets.byte_index import EpisodeByteIndex
    from lerobot.datasets.byte_index_builder import build_byte_index_in_memory
    from lerobot.datasets.episode_byte_cache import EpisodeByteCache

    _, meta = byte_index_dir
    index = build_byte_index_in_memory(meta, BUCKET, workers=4, max_episodes=MAX_EPISODES)
    cache = EpisodeByteCache(index, max_bytes=80_000_000_000, data_root=BUCKET)
    for ep in [0, MAX_EPISODES // 2, MAX_EPISODES - 1]:
        cache.submit_prefetch(ep)
        cache.ensure_ready(ep)
    stats = cache.stats.stats_dict()
    assert stats["byte_cache_bytes_per_miss"] < 25 * 1024 * 1024


@pytest.mark.integration
@COMPUTE_NODE
def test_in_memory_build_matches_parquet(byte_index_dir):
    from lerobot.datasets.byte_index import EpisodeByteIndex
    from lerobot.datasets.byte_index_builder import build_byte_index_in_memory

    out, meta = byte_index_dir
    disk = EpisodeByteIndex(out, video_keys=meta.video_keys, num_episodes=MAX_EPISODES)
    mem = build_byte_index_in_memory(meta, BUCKET, workers=4, max_episodes=MAX_EPISODES)
    for ep in [0, MAX_EPISODES // 2, MAX_EPISODES - 1]:
        for cam in meta.video_keys:
            a = disk.lookup(ep, cam)
            b = mem.lookup(ep, cam)
            assert a.mdat_offset == b.mdat_offset
            assert a.mdat_length == b.mdat_length
            assert abs(a.first_pts - b.first_pts) < 1e-6


@pytest.mark.integration
@COMPUTE_NODE
def test_custom_frame_mappings_available(byte_index_dir):
    from lerobot.datasets.byte_index_builder import build_byte_index_in_memory

    _, meta = byte_index_dir
    index = build_byte_index_in_memory(meta, BUCKET, workers=4, max_episodes=MAX_EPISODES)
    cam = meta.video_keys[0]
    ep = MAX_EPISODES // 2
    payload = index.custom_frame_mappings(ep, cam)
    assert payload is not None
    data = json.loads(payload)
    assert len(data["frames"]) > 10
    assert any(f["key_frame"] for f in data["frames"])
    assert all("pts" in f and "duration" in f for f in data["frames"])


@pytest.mark.integration
@COMPUTE_NODE
def test_metadata_skip_decoder_init(byte_index_dir):
    from lerobot.datasets.byte_index_builder import build_byte_index_in_memory
    from lerobot.datasets.episode_byte_cache import EpisodeByteCache

    _, meta = byte_index_dir
    index = build_byte_index_in_memory(meta, BUCKET, workers=4, max_episodes=MAX_EPISODES)
    cache = EpisodeByteCache(index, max_bytes=8_000_000_000, data_root=BUCKET)
    cam = meta.video_keys[0]
    ep = 0
    cache.submit_prefetch(ep)
    cache.ensure_ready(ep)
    dec = cache.get_decoder(ep, cam)
    assert dec.metadata.num_frames is not None
    assert dec.metadata.num_frames > 0
    begin = float(dec.metadata.begin_stream_seconds)
    end = float(dec.metadata.end_stream_seconds)
    ts = begin + 0.5 * (end - begin)
    frame = dec.get_frames_played_at([ts]).data
    assert frame.ndim == 4


@pytest.mark.integration
@COMPUTE_NODE
def test_sparse_decode_produces_frames(byte_index_dir):
    from lerobot.datasets.byte_index_builder import build_byte_index_in_memory
    from lerobot.datasets.episode_byte_cache import EpisodeByteCache

    _, meta = byte_index_dir
    index = build_byte_index_in_memory(meta, BUCKET, workers=4, max_episodes=MAX_EPISODES)
    cache = EpisodeByteCache(index, max_bytes=80_000_000_000, data_root=BUCKET)
    cam = meta.video_keys[0]
    ep = 0
    cache.submit_prefetch(ep)
    cache.ensure_ready(ep)
    dec = cache.get_decoder(ep, cam)
    begin = float(dec.metadata.begin_stream_seconds)
    end = float(dec.metadata.end_stream_seconds)
    ts = begin + 0.5 * (end - begin)
    frame = dec.get_frames_played_at([ts]).data
    assert frame.ndim == 4
    assert frame.numel() > 0
    assert float(frame.float().std()) > 1.0
