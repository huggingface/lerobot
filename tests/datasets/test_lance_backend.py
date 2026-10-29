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
"""Lance storage backend: LeRobotDataset over a Lance root must return the same
items as over the default parquet/mp4 layout, through the same public class."""

import json
import pickle
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("lancedb")
pytest.importorskip("lerobot_lancedb", reason="lerobot-lancedb converts the test fixtures")

import lancedb
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot_lancedb.convert import convert

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets import lance_backend, lance_utils
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.factory import make_dataset, make_train_eval_datasets
from lerobot.datasets.lance_backend import LanceDatasetReader, lance_mp_context
from lerobot.datasets.language import (
    LANGUAGE_COLUMNS,
    language_events_arrow_type,
    language_feature_info,
    language_persistent_arrow_type,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.storage import localize_remote_root
from lerobot.datasets.video_utils import FrameTimestampError
from lerobot.policies.factory import make_policy_config
from lerobot.utils.constants import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT
from tests.fixtures.constants import (
    DUMMY_CAMERA_FEATURES_WITH_DEPTH,
    DUMMY_REPO_ID,
)

pytestmark = pytest.mark.dataset


def _storage_type(dtype: pa.DataType) -> pa.DataType:
    """Extension type -> its storage type (pa.array can't build extension values directly)."""
    if isinstance(dtype, pa.BaseExtensionType):
        return _storage_type(dtype.storage_type)
    if pa.types.is_list(dtype):
        return pa.list_(_storage_type(dtype.value_type))
    if pa.types.is_large_list(dtype):
        return pa.large_list(_storage_type(dtype.value_type))
    if pa.types.is_struct(dtype):
        return pa.struct([field.with_type(_storage_type(field.type)) for field in dtype])
    return dtype


@pytest.fixture
def dataset_roots(tmp_path, lerobot_dataset_factory) -> tuple[Path, Path]:
    src_root = tmp_path / "src"
    lerobot_dataset_factory(
        root=src_root, total_episodes=3, total_frames=90, use_videos=False, camera_features={}
    )
    lance_root = tmp_path / "lance"
    convert(lance_root, root=src_root)
    return src_root, lance_root


def assert_items_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for key, expected_val in expected.items():
        if isinstance(expected_val, str):
            assert actual[key] == expected_val, key
        else:
            assert actual[key].dtype == expected_val.dtype, key
            assert actual[key].shape == expected_val.shape, key
            torch.testing.assert_close(actual[key], expected_val, rtol=0, atol=0, msg=key)


class _FakeRGBDecoder:
    metadata = SimpleNamespace(average_fps=10.0)

    def __init__(self):
        self.calls = []

    def get_frames_at(self, indices):
        self.calls.append(indices)
        return SimpleNamespace(
            data=torch.stack([torch.full((3, 2, 2), index, dtype=torch.uint8) for index in indices]),
            pts_seconds=torch.tensor(indices, dtype=torch.float64) / self.metadata.average_fps,
        )


def _depth_decode_reader(
    file_key: tuple[str, int, int], shape: tuple[int, int, int] = (2, 2, 1)
) -> tuple[LanceDatasetReader, list[list[float]]]:
    reader = LanceDatasetReader.__new__(LanceDatasetReader)
    reader.meta = SimpleNamespace(fps=10, features={file_key[0]: {"shape": shape}})
    reader.tolerance_s = 1e-4
    calls: list[list[float]] = []

    def decode_depth_frames(source, timestamps: list[float], key: tuple) -> torch.Tensor:
        calls.append(timestamps)
        return torch.tensor(timestamps, dtype=torch.float32).reshape(-1, 1, 1, 1)

    reader._decode_depth_frames = decode_depth_frames
    return reader, calls


def test_tabular_parity(dataset_roots):
    src_root, lance_root = dataset_roots
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root)
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root)
    assert isinstance(lance_ds.reader, LanceDatasetReader)
    assert len(lance_ds) == len(upstream)
    for idx in [0, len(upstream) // 2, len(upstream) - 1]:
        assert_items_equal(lance_ds[idx], upstream[idx])

    # delta windows: first/last frames of an episode exercise clamping + pads
    fps = upstream.meta.fps
    delta_timestamps = {"state": [-2 / fps, -1 / fps, 0.0], "action": [0.0, 1 / fps, 2 / fps, 3 / fps]}
    upstream_d = LeRobotDataset(DUMMY_REPO_ID, root=src_root, delta_timestamps=delta_timestamps)
    lance_d = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, delta_timestamps=delta_timestamps)
    ep_start = int(upstream_d.meta.episodes[1]["dataset_from_index"])
    ep_end = int(upstream_d.meta.episodes[1]["dataset_to_index"])
    for idx in [ep_start, ep_start + 5, ep_end - 1]:
        expected = upstream_d[idx]
        actual = lance_d[idx]
        assert actual["state_is_pad"].any() == expected["state_is_pad"].any()
        assert_items_equal(actual, expected)

    # batched fetch must equal singles (duplicates included)
    indices = [3, 3, 17, 55]
    for idx, item in zip(indices, lance_ds.__getitems__(indices), strict=True):
        assert_items_equal(item, lance_ds[idx])

    # negative indexing mirrors upstream; out-of-range raises
    assert_items_equal(lance_ds[-1], upstream[-1])
    with pytest.raises(IndexError):
        lance_ds[len(lance_ds)]

    # out-of-range episode indices are dropped (same resolve_episode_indices helper);
    # an all-invalid selection errors on both readers rather than wrapping around
    with pytest.raises(ValueError):
        LeRobotDataset(DUMMY_REPO_ID, root=src_root, episodes=[-1])[0]
    with pytest.raises(ValueError, match="None of the requested episodes"):
        LeRobotDataset(DUMMY_REPO_ID, root=lance_root, episodes=[-1])
    upstream_m = LeRobotDataset(DUMMY_REPO_ID, root=src_root, episodes=[-1, 1])
    lance_m = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, episodes=[-1, 1])
    assert len(lance_m) == len(upstream_m)
    assert lance_m.num_episodes == upstream_m.num_episodes
    assert lance_m.episodes == [1]
    assert_items_equal(lance_m[0], upstream_m[0])

    # episode subset: relative/absolute mapping mirrors upstream
    upstream_s = LeRobotDataset(DUMMY_REPO_ID, root=src_root, episodes=[1])
    lance_s = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, episodes=[1])
    assert len(lance_s) == len(upstream_s)
    assert lance_s.num_episodes == upstream_s.num_episodes
    assert lance_s.absolute_to_relative_idx == upstream_s.reader._absolute_to_relative_idx
    for idx in [0, len(upstream_s) - 1]:
        assert_items_equal(lance_s[idx], upstream_s[idx])

    # unordered episodes list: user order kept on .episodes, row order mirrors upstream
    upstream_u = LeRobotDataset(DUMMY_REPO_ID, root=src_root, episodes=[1, 0])
    lance_u = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, episodes=[1, 0])
    assert lance_u.episodes == upstream_u.episodes == [1, 0]
    for idx in [0, len(upstream_u) // 2, len(upstream_u) - 1]:
        assert_items_equal(lance_u[idx], upstream_u[idx])

    # close() releases pools/handles; the next read reopens
    lance_ds.reader.close()
    assert_items_equal(lance_ds[0], upstream[0])


@pytest.fixture
def video_dataset_roots(tmp_path, lerobot_dataset_factory) -> tuple[Path, Path]:
    src_root = tmp_path / "src_video"
    lerobot_dataset_factory(
        root=src_root,
        total_episodes=2,
        total_frames=40,
        use_videos=True,
        camera_features=DUMMY_CAMERA_FEATURES_WITH_DEPTH,
    )
    lance_root = tmp_path / "lance_video"
    convert(lance_root, root=src_root)
    return src_root, lance_root


def test_video_parity(video_dataset_roots):
    src_root, lance_root = video_dataset_roots
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root, video_backend="torchcodec")
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root)
    for idx in [0, len(upstream) // 2, len(upstream) - 1]:
        assert_items_equal(lance_ds[idx], upstream[idx])

    # depth maps ride the same fixture (static items covered by the compare above)
    assert upstream.meta.depth_keys

    fps = upstream.meta.fps
    video_key = upstream.meta.video_keys[0]
    depth_key = upstream.meta.depth_keys[0]
    # delta windows over both an rgb and a depth camera in one shot
    delta_timestamps = {
        video_key: [-1 / fps, 0.0, 1 / fps],
        depth_key: [-1 / fps, 0.0, 1 / fps],
        "action": [0.0, 1 / fps],
    }
    upstream_d = LeRobotDataset(
        DUMMY_REPO_ID, root=src_root, delta_timestamps=delta_timestamps, video_backend="torchcodec"
    )
    lance_d = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, delta_timestamps=delta_timestamps)
    ep_start = int(upstream_d.meta.episodes[1]["dataset_from_index"])
    for idx in [0, ep_start - 1, ep_start]:
        expected = upstream_d[idx]
        actual = lance_d[idx]
        assert actual[f"{video_key}_is_pad"].tolist() == expected[f"{video_key}_is_pad"].tolist()
        assert_items_equal(actual, expected)

    # Batched depth windows share a PyAV open/seek but must remain identical to
    # individual reads, including duplicate and adjacent sample indices.
    indices = [3, 4, 4, 5]
    for idx, item in zip(indices, lance_d.__getitems__(indices), strict=True):
        assert_items_equal(item, lance_d[idx])

    # one-element window: squeeze semantics must match upstream's shape
    single = {video_key: [0.0]}
    upstream_1 = LeRobotDataset(
        DUMMY_REPO_ID, root=src_root, delta_timestamps=single, video_backend="torchcodec"
    )
    lance_1 = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, delta_timestamps=single)
    for idx in [0, len(upstream_1) - 1]:
        assert_items_equal(lance_1[idx], upstream_1[idx])

    # return_uint8: raw frames, no normalization
    lance_u8 = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, return_uint8=True)
    item = lance_u8[0]
    assert item[video_key].dtype == torch.uint8

    # A real TorchCodec batch must remain byte-identical to individual reads,
    # including duplicate and adjacent sample indices.
    indices = [3, 3, 17, 18]
    for idx, item in zip(indices, lance_u8.__getitems__(indices), strict=True):
        assert_items_equal(item, lance_u8[idx])


@pytest.mark.parametrize(
    ("shape", "bytes_per_value", "expected_frames"),
    [
        pytest.param((1024, 1024, 3), 1, 21, id="rgb-uint8"),
        pytest.param((1024, 1024, 3), 5, 4, id="rgb-float32"),
        pytest.param((720, 1280, 1), 8, 9, id="depth"),
        pytest.param((2, 2, 1), 8, 64, id="frame-cap"),
        pytest.param((4096, 4096, 3), 5, 1, id="oversized-frame"),
        pytest.param(None, 8, 64, id="unknown-shape"),
    ],
)
def test_decode_frame_limit(
    shape: tuple[int, ...] | None, bytes_per_value: int, expected_frames: int
) -> None:
    assert (
        lance_backend._decode_frame_limit(
            shape, max_frames=64, max_bytes=64 << 20, bytes_per_value=bytes_per_value
        )
        == expected_frames
    )


def test_rgb_batch_decode_deduplicates_overlapping_frames():
    file_key = ("observation.images.camera", 0, 0)
    file_requests = [
        (0, [0.1, 0.2, 0.2]),
        (1, [0.2, 0.3]),
        (2, [0.3]),
    ]

    decoder = _FakeRGBDecoder()
    reader = LanceDatasetReader.__new__(LanceDatasetReader)
    reader.meta = SimpleNamespace(depth_keys=[], features={file_key[0]: {"shape": (2, 2, 3)}})
    reader.return_uint8 = True
    reader.tolerance_s = 1e-4
    reader._build_video_requests = lambda *args: {file_key: file_requests}

    with ThreadPoolExecutor(max_workers=1) as reader._decode_pool:
        decoded = reader._decode_videos([{}, {}, {}], {"timestamp": []}, {}, {file_key: (decoder, None)})

    assert decoder.calls == [[1, 2, 3]]
    assert decoded[0][file_key[0]][:, 0, 0, 0].tolist() == [1, 2, 2]
    assert decoded[1][file_key[0]][:, 0, 0, 0].tolist() == [2, 3]
    assert decoded[2][file_key[0]][0, 0, 0].item() == 3


def test_rgb_batch_decode_limits_and_releases_chunks():
    file_key = ("observation.images.camera", 0, 0)
    max_frames = lance_backend._RGB_DECODE_MAX_FRAMES
    file_requests = [
        (0, [index / 10 for index in range(max_frames)]),
        (1, [index / 10 for index in range(max_frames, 2 * max_frames + 5)]),
    ]

    class Decoder(_FakeRGBDecoder):
        def __init__(self):
            super().__init__()
            self.previous_data = None

        def get_frames_at(self, indices):
            if self.previous_data is not None:
                assert self.previous_data() is None
            batch = super().get_frames_at(indices)
            self.previous_data = weakref.ref(batch.data)
            return batch

    decoder = Decoder()
    reader = LanceDatasetReader.__new__(LanceDatasetReader)
    reader.meta = SimpleNamespace(depth_keys=[], features={file_key[0]: {"shape": (2, 2, 3)}})
    reader.return_uint8 = True
    reader.tolerance_s = 1e-4
    reader._build_video_requests = lambda *args: {file_key: file_requests}

    with ThreadPoolExecutor(max_workers=1) as reader._decode_pool:
        decoded = reader._decode_videos([{}, {}], {"timestamp": []}, {}, {file_key: (decoder, None)})

    assert decoder.calls == [
        list(range(max_frames)),
        list(range(max_frames, 2 * max_frames)),
        list(range(2 * max_frames, 2 * max_frames + 5)),
    ]
    assert decoded[0][file_key[0]][:, 0, 0, 0].tolist() == list(range(max_frames))
    assert decoded[1][file_key[0]][:, 0, 0, 0].tolist() == list(range(max_frames, 2 * max_frames + 5))


def test_rgb_batch_decode_rejects_bad_pts_after_chunk_scatter():
    file_key = ("observation.images.camera", 0, 0)
    max_frames = lance_backend._RGB_DECODE_MAX_FRAMES
    file_requests = [
        (0, [index / 10 for index in range(max_frames)]),
        (1, [index / 10 for index in range(max_frames, max_frames + 5)]),
    ]

    class Decoder(_FakeRGBDecoder):
        def get_frames_at(self, indices):
            batch = super().get_frames_at(indices)
            if indices[0] >= max_frames:
                batch.pts_seconds[-1] += 2e-4
            return batch

    decoder = Decoder()
    reader = LanceDatasetReader.__new__(LanceDatasetReader)
    reader.meta = SimpleNamespace(depth_keys=[], features={file_key[0]: {"shape": (2, 2, 3)}})
    reader.return_uint8 = True
    reader.tolerance_s = 1e-4
    reader._build_video_requests = lambda *args: {file_key: file_requests}

    with (
        ThreadPoolExecutor(max_workers=1) as reader._decode_pool,
        pytest.raises(FrameTimestampError, match="chunk 0, file 0"),
    ):
        reader._decode_videos([{}, {}], {"timestamp": []}, {}, {file_key: (decoder, None)})

    assert decoder.calls == [list(range(max_frames)), list(range(max_frames, max_frames + 5))]


def test_rgb_batch_decode_respects_byte_limit_for_float_output():
    file_key = ("observation.images.camera", 0, 0)
    file_requests = [
        (0, [index / 10 for index in range(4)]),
        (1, [index / 10 for index in range(3, 7)]),
    ]

    decoder = _FakeRGBDecoder()
    reader = LanceDatasetReader.__new__(LanceDatasetReader)
    reader.meta = SimpleNamespace(depth_keys=[], features={file_key[0]: {"shape": (1024, 1024, 3)}})
    reader.return_uint8 = False
    reader.tolerance_s = 1e-4
    reader._build_video_requests = lambda *args: {file_key: file_requests}

    with ThreadPoolExecutor(max_workers=1) as reader._decode_pool:
        decoded = reader._decode_videos([{}, {}], {"timestamp": []}, {}, {file_key: (decoder, None)})

    # At 1024x1024, each normalized frame temporarily occupies 3 MiB of decoder
    # uint8 storage plus 12 MiB of float32 output, so the 64 MiB budget permits four.
    assert decoder.calls == [list(range(4)), list(range(4, 7))]
    torch.testing.assert_close(
        decoded[0][file_key[0]][:, 0, 0, 0], torch.arange(4, dtype=torch.float32) / 255
    )
    torch.testing.assert_close(
        decoded[1][file_key[0]][:, 0, 0, 0], torch.arange(3, 7, dtype=torch.float32) / 255
    )


def test_rgb_single_request_keeps_direct_decode_result():
    file_key = ("observation.images.camera", 0, 0)
    batch_data = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 2, 2)

    class Decoder:
        metadata = SimpleNamespace(average_fps=10.0)

        def get_frames_at(self, indices):
            assert indices == [1, 2]
            return SimpleNamespace(data=batch_data, pts_seconds=torch.tensor([0.1, 0.2]))

    reader = LanceDatasetReader.__new__(LanceDatasetReader)
    reader.meta = SimpleNamespace(depth_keys=[], features={file_key[0]: {"shape": (2, 2, 3)}})
    reader.return_uint8 = True
    reader.tolerance_s = 1e-4
    reader._build_video_requests = lambda *args: {file_key: [(0, [0.1, 0.2])]}

    with ThreadPoolExecutor(max_workers=1) as reader._decode_pool:
        decoded = reader._decode_videos([{}], {"timestamp": []}, {}, {file_key: (Decoder(), None)})

    assert decoded[0][file_key[0]].data_ptr() == batch_data.data_ptr()


def test_depth_batch_decode_merges_nearby_windows() -> None:
    file_key = ("observation.images.depth", 0, 0)
    file_requests = [
        (0, [0.2, 0.1, 0.2, 0.0]),
        (1, [0.1, 0.2, 0.3]),
        (2, [5.0]),
    ]
    reader, calls = _depth_decode_reader(file_key)
    decoded = dict(reader._decode_depth_file_requests(None, file_requests, file_key))

    assert calls == [[0.0, 0.1, 0.2, 0.3], [5.0]]
    # The shared decode must preserve each window's original order and duplicates.
    torch.testing.assert_close(decoded[0][:, 0, 0, 0], torch.tensor([0.2, 0.1, 0.2, 0.0]))
    torch.testing.assert_close(decoded[1][:, 0, 0, 0], torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(decoded[2], torch.full((1, 1, 1), 5.0))


@pytest.mark.parametrize(
    ("shape", "max_frames"),
    [
        pytest.param((720, 1280, 1), 9, id="byte-cap"),
        pytest.param((2, 2, 1), 64, id="frame-cap"),
    ],
)
def test_depth_batch_decode_limits_merged_span(shape: tuple[int, int, int], max_frames: int) -> None:
    file_key = ("observation.images.depth", 0, 0)
    file_requests = [(index, [index / 10]) for index in range(max_frames + 6)]
    reader, calls = _depth_decode_reader(file_key, shape)
    decoded = dict(reader._decode_depth_file_requests(None, file_requests, file_key))

    assert len(calls) == 2  # Nearby requests still merge while respecting the limit.
    # Requests lie on the 10 FPS grid, so rounding measures span without testing
    # the implementation's floating-point ceiling or exact partition boundary.
    assert all(round((max(timestamps) - min(timestamps)) * 10) + 1 <= max_frames for timestamps in calls)
    assert set(decoded) == set(range(max_frames + 6))
    for sample_idx, timestamps in file_requests:
        torch.testing.assert_close(decoded[sample_idx], torch.tensor(timestamps).reshape(1, 1, 1))


def test_reader_reopens_after_failed_open(video_dataset_roots, monkeypatch):
    _, lance_root = video_dataset_roots
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root)
    reader = lance_ds.reader
    # the video row-id map is built at construction and survives pickling
    assert reader._video_row_ids
    assert pickle.loads(pickle.dumps(reader))._video_row_ids == reader._video_row_ids

    def fail_open(*args, **kwargs):
        raise OSError("simulated 429")

    monkeypatch.setattr(lance_backend.Permutation, "identity", fail_open)
    with pytest.raises(OSError):
        lance_ds[0]
    # nothing was published, so the next read starts over
    assert reader._frames_perm is None
    assert reader._prefetch_pool is None
    monkeypatch.undo()
    lance_ds[0]


def test_video_decoder_cache_size_reaches_reader(video_dataset_roots):
    src_root, lance_root = video_dataset_roots
    assert LeRobotDataset(DUMMY_REPO_ID, root=lance_root).reader._decoder_cache.capacity == 256
    small = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, video_decoder_cache_size=4)
    assert small.reader._decoder_cache.capacity == 4
    assert small[0][small.meta.camera_keys[0]].ndim == 3
    with pytest.raises(ValueError, match="non-default storage formats"):
        LeRobotDataset(DUMMY_REPO_ID, root=src_root, video_decoder_cache_size=4)


@pytest.mark.parametrize("cached_decoder", [False, True])
def test_batch_metadata_survives_cache_eviction(video_dataset_roots, cached_decoder):
    _, lance_root = video_dataset_roots
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root)
    expected = lance_ds[0]
    reader = lance_ds.reader

    # Keep one required file as the oldest entry in a full cache. Loading the
    # other cameras' metadata will evict it before this batch uses it.
    file_key, meta = next(iter(reader._file_meta.items()))
    reader._file_meta = OrderedDict([(file_key, meta)])
    for i in range(2047):
        reader._file_meta[("unused", 0, i)] = meta
    if not cached_decoder:
        reader._decoder_cache = lance_utils._VideoDecoderLRU(
            reader._decoder_cache.capacity, byte_budget=reader._decoder_cache.byte_budget
        )

    assert_items_equal(lance_ds[0], expected)
    assert len(reader._file_meta) <= 2048


def test_batch_metadata_larger_than_cache(tmp_path):
    num_files = 2049
    table = lancedb.connect(tmp_path).create_table(
        "videos",
        data=[
            {
                "video_key": "camera",
                "chunk_index": 0,
                "file_index": i,
                "file_size": 1024 + i,
                "moov_offset": 0,
                "moov_size": 100,
                "kf_indices": [0],
                "kf_positions": [100],
            }
            for i in range(num_files)
        ],
    )
    reader = LanceDatasetReader.__new__(LanceDatasetReader)
    reader._videos_table = table
    reader._file_meta = OrderedDict()
    reader._video_row_ids = {
        ("camera", 0, row["file_index"]): row["_rowid"]
        for row in table.search().select(["file_index"]).with_row_id(True).to_arrow().to_pylist()
    }

    file_meta = reader._load_file_meta(list(reader._video_row_ids))

    assert set(file_meta) == set(reader._video_row_ids)
    assert len(reader._file_meta) <= 2048
    for i in range(num_files):
        assert file_meta[("camera", 0, i)]["file_size"] == 1024 + i


def test_sparse_source_fetches_handle_once():
    class Handle:
        opened = 0

        def read_range(self, pos, n):
            return b"x" * n

    def open_handle():
        Handle.opened += 1
        return Handle()

    source = lance_utils._SparseBlobSource(8, open_handle)
    source.add(0, b"abcd")
    assert source.read(4) == b"abcd"
    assert Handle.opened == 0  # buffered reads never touch the handle
    assert source.read(2) == b"xx"
    assert source.read(2) == b"xx"
    assert Handle.opened == 1  # one handle for every miss


def test_storage_format_routing(video_dataset_roots):
    src_root, lance_root = video_dataset_roots

    def make(root):
        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=str(root)),
            policy=make_policy_config("act"),
        )
        return make_dataset(cfg)

    # storage_format in meta/info.json routes to the backend; both are the same public class
    lance_ds = make(lance_root)
    assert isinstance(lance_ds, LeRobotDataset)
    assert lance_ds.meta.storage_format == "lance"
    assert isinstance(lance_ds.reader, LanceDatasetReader)
    # parquet-only surface reports absent instead of raising, so probes like
    # lerobot_train's hasattr(ds, "hf_dataset") skip it gracefully
    assert not hasattr(lance_ds, "hf_dataset")
    with pytest.raises(NotImplementedError, match="push_to_hub"):
        lance_ds.push_to_hub()

    parquet_ds = make(src_root)
    assert isinstance(parquet_ds, LeRobotDataset)
    assert parquet_ds.meta.storage_format == "lerobot"
    assert isinstance(parquet_ds.reader, DatasetReader)

    # A file:// URI root must route to the same local lance dataset
    uri_ds = make(f"file://{lance_root}")
    assert isinstance(uri_ds.reader, LanceDatasetReader)

    # An unknown format fails with a clear error instead of the parquet path
    info_path = src_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["storage_format"] = "warehouse13"
    info_path.write_text(json.dumps(info, indent=4))
    with pytest.raises(ValueError, match="storage_format 'warehouse13'"):
        LeRobotDataset(DUMMY_REPO_ID, root=src_root)


def test_repo_type_bucket_matrix(dataset_roots, monkeypatch):
    src_root, _ = dataset_roots

    # repo_type="bucket" without a root derives hf://buckets/{repo_id}
    import lerobot.datasets.lerobot_dataset as lerobot_dataset_module

    seen = {}

    def capture_root(repo_id, root, *args, **kwargs):
        seen["root"] = str(root)
        raise FileNotFoundError("stop before network")

    monkeypatch.setattr(lerobot_dataset_module, "localize_remote_root", capture_root)
    with pytest.raises(FileNotFoundError):
        LeRobotDataset(DUMMY_REPO_ID, repo_type="bucket")
    assert seen["root"] == f"hf://buckets/{DUMMY_REPO_ID}"
    monkeypatch.undo()

    # a default-format dataset at an object-store root points at streaming
    with pytest.raises(FileNotFoundError, match="streaming"):
        localize_remote_root(DUMMY_REPO_ID, f"file://{src_root}")

    # factory: bucket + default format without streaming keeps the old error
    import lerobot.datasets.factory as factory_module

    monkeypatch.setattr(
        factory_module,
        "load_dataset_metadata",
        lambda *args, **kwargs: LeRobotDatasetMetadata(DUMMY_REPO_ID, root=src_root),
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=str(src_root), repo_type="bucket"),
        policy=make_policy_config("act"),
    )
    with pytest.raises(ValueError, match="streaming-only"):
        make_dataset(cfg)


def test_force_cache_sync_refreshes_remote_meta(video_dataset_roots):
    _, lance_root = video_dataset_roots
    uri = f"file://{lance_root}"
    ds = LeRobotDataset(DUMMY_REPO_ID, root=uri)

    # simulate a stale materialized cache from an earlier version of the dataset
    cached_info = Path(ds.root) / "meta" / "info.json"
    stale = json.loads(cached_info.read_text())
    stale["fps"] = 999
    cached_info.write_text(json.dumps(stale, indent=4))

    assert LeRobotDataset(DUMMY_REPO_ID, root=uri).meta.fps == 999  # cache reused as-is
    refreshed = LeRobotDataset(DUMMY_REPO_ID, root=uri, force_cache_sync=True)
    assert refreshed.meta.fps != 999  # re-materialized from the meta table


def test_metadata_cache_identity_preserves_default_cache_path() -> None:
    root = "hf://datasets/user/repo"
    expected = "hf_datasets_user_repo_main"
    for options in (None, {}, {"token": "rotated-token", "timeout": "60s"}):
        assert lance_backend.metadata_cache_identity(root, "main", options) == expected
        assert lance_backend.metadata_cache_identity(f"{root}/", "main", options) == expected


def test_metadata_cache_identity_separates_sources_and_revisions() -> None:
    root = "s3://shared-bucket/dataset"
    options = {"aws_endpoint": "https://endpoint.example.com", "aws_region": "us-east-1"}
    identity = lance_backend.metadata_cache_identity(root, "main", options)
    original_options = options.copy()
    rotated_options = {
        **{key.upper(): value for key, value in reversed(options.items())},
        "aws_access_key_id": "rotated-id",
        "aws_secret_access_key": "rotated-secret",
        "aws_session_token": "rotated-session",
        "timeout": "60s",
    }
    assert lance_backend.metadata_cache_identity(root, "main", rotated_options) == identity
    assert options == original_options
    assert lance_backend.metadata_cache_identity(root, "other-revision", options) != identity
    assert lance_backend.metadata_cache_identity(f"{root}-other", "main", options) != identity
    for key, value in (
        ("aws_endpoint", "https://other-endpoint.example.com"),
        ("aws_region", "us-west-2"),
        ("aws_virtual_hosted_style_request", "true"),
        ("azure_storage_account_name", "other-account"),
    ):
        assert lance_backend.metadata_cache_identity(root, "main", {**options, key: value}) != identity


@pytest.mark.parametrize("endpoint_key", ["endpoint", "aws_endpoint", "aws_endpoint_url"])
def test_remote_metadata_cache_separates_endpoints(
    tmp_path, monkeypatch, lerobot_dataset_factory, tasks_factory, endpoint_key: str
) -> None:
    remote_root = "s3://shared-bucket/dataset"
    roots = {}
    for label in ("endpoint-a", "endpoint-b"):
        tasks = tasks_factory(total_tasks=1)
        tasks.index = [label]
        src_root = tmp_path / label / "src"
        lerobot_dataset_factory(
            root=src_root,
            total_episodes=3,
            total_frames=90,
            use_videos=False,
            camera_features={},
            tasks=tasks,
        )
        lance_root = tmp_path / label / "lance"
        convert(lance_root, root=src_root)
        roots[f"https://{label}.example.com"] = lance_root

    connect = lancedb.connect

    def connect_endpoint(uri, *, storage_options):
        if uri != remote_root:
            assert Path(uri) in roots.values()
            return connect(uri, storage_options=storage_options)
        assert uri == remote_root
        return connect(str(roots[storage_options[endpoint_key]]))

    monkeypatch.setattr(lancedb, "connect", connect_endpoint)
    monkeypatch.setattr(lance_utils, "HF_LEROBOT_HOME", tmp_path / "cache")
    materialize_meta = Mock(wraps=lance_utils._materialize_meta)
    monkeypatch.setattr(lance_utils, "_materialize_meta", materialize_meta)
    options_a = {endpoint_key: "https://endpoint-a.example.com", "aws_region": "us-east-1"}
    rotated_options_a = {
        **dict(reversed(options_a.items())),
        "aws_access_key_id": "rotated-id",
        "aws_secret_access_key": "rotated-secret",
        "aws_session_token": "rotated-session",
        "token": "rotated-hub-token",
        "timeout": "60s",
    }
    datasets = []
    try:
        for label, options in (
            ("endpoint-a", options_a),
            ("endpoint-b", {**options_a, endpoint_key: "https://endpoint-b.example.com"}),
            ("endpoint-a", rotated_options_a),
        ):
            dataset = LeRobotDataset(
                DUMMY_REPO_ID,
                root=remote_root,
                storage_options=options,
            )
            datasets.append(dataset)
            # Equal row counts must not hide metadata from the wrong endpoint.
            assert len(dataset) == 90
            assert dataset.meta.tasks.index.tolist() == [label]
            assert dataset[0]["task"] == label
        assert datasets[0].root != datasets[1].root
        assert datasets[0].root == datasets[2].root
        # Credentials, request tuning and dict order must not trigger another metadata read.
        assert materialize_meta.call_count == 2

        # Refreshing B must not remove or replace A's cached metadata.
        refreshed = LeRobotDataset(
            DUMMY_REPO_ID,
            root=remote_root,
            storage_options={**options_a, endpoint_key: "https://endpoint-b.example.com"},
            force_cache_sync=True,
        )
        datasets.append(refreshed)
        assert refreshed[0]["task"] == "endpoint-b"
        assert LeRobotDatasetMetadata(DUMMY_REPO_ID, root=datasets[0].root).tasks.index.tolist() == [
            "endpoint-a"
        ]
    finally:
        for dataset in datasets:
            dataset.reader.close()


@pytest.mark.parametrize("revision_key", ["revision", "REVISION"])
def test_storage_options_rejects_revision(revision_key):
    with pytest.raises(ValueError, match="storage_options.*revision"):
        lance_utils._storage_options(
            "hf://datasets/user/repo", {revision_key: "branch-in-options"}, revision="main"
        )

    assert lance_utils._storage_options("hf://datasets/user/repo", {}, revision="main")["revision"] == "main"


@pytest.mark.parametrize("entry_point", ["dataset", "train_eval"])
def test_storage_options_remote_reads(dataset_roots, tmp_path, monkeypatch, entry_point: str) -> None:
    src_root, lance_root = dataset_roots
    remote_root = "s3://test-bucket/dataset"
    storage_options = {"aws_endpoint": "https://s3.example.com", "aws_region": "us-east-1"}
    expected_options = storage_options.copy()
    connections = []
    connect = lancedb.connect

    def connect_with_options(uri, **kwargs):
        # Emulate a store that requires these options, while reading real local
        # Lance tables. Missing options must fail even on the first metadata read.
        assert uri in (remote_root, str(lance_root))
        assert kwargs.get("storage_options") == expected_options
        if uri == remote_root:
            connections.append(kwargs["storage_options"].copy())
        return connect(str(lance_root), **kwargs)

    monkeypatch.setattr(lance_utils, "HF_LEROBOT_HOME", tmp_path / "cache")
    monkeypatch.setattr(lancedb, "connect", connect_with_options)
    if entry_point == "dataset":
        datasets = [LeRobotDataset(DUMMY_REPO_ID, root=remote_root, storage_options=storage_options)]
    else:
        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(
                repo_id=DUMMY_REPO_ID,
                root=remote_root,
                storage_options=storage_options,
                eval_split=0.34,
            ),
            policy=make_policy_config("act"),
        )
        # This also exercises make_dataset before rebuilding the train and eval readers.
        datasets = list(make_train_eval_datasets(cfg))

    # Cold metadata loading has already connected, before any frame was read.
    assert len(connections) == 1
    assert storage_options == expected_options
    storage_options["aws_region"] = "changed-after-construction"
    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root)
    if entry_point == "train_eval":
        assert len(datasets[0]) + len(datasets[1]) == len(upstream)
        assert set(datasets[0].episodes).isdisjoint(datasets[1].episodes)
    for dataset in datasets:
        sample = dataset[0]
        reference = LeRobotDataset(DUMMY_REPO_ID, root=src_root, delta_timestamps=dataset.delta_timestamps)
        assert_items_equal(sample, reference[int(sample["index"])])

        if entry_point == "dataset":
            # Serialization and reopening share the same reader across all entry points.
            previous_connections = len(connections)
            restored = pickle.loads(pickle.dumps(dataset))
            assert_items_equal(restored[0], sample)
            assert len(connections) == previous_connections + 1
            restored.reader.close()
            dataset.reader.close()
            assert_items_equal(dataset[0], sample)
            assert len(connections) == previous_connections + 2
        dataset.reader.close()


def test_default_storage_rejects_storage_options(dataset_roots):
    src_root, _ = dataset_roots
    with pytest.raises(ValueError, match="storage_options.*default"):
        LeRobotDataset(DUMMY_REPO_ID, root=src_root, storage_options={"aws_region": "us-east-1"})

    assert len(LeRobotDataset(DUMMY_REPO_ID, root=src_root, storage_options={})) == 90


def test_pickle_and_dataloader(dataset_roots):
    _, lance_root = dataset_roots
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root, storage_options={"aws_region": "us-east-1"})
    restored = pickle.loads(pickle.dumps(lance_ds))
    assert_items_equal(restored[7], lance_ds[7])

    loader = torch.utils.data.DataLoader(
        lance_ds,
        batch_size=8,
        num_workers=2,
        multiprocessing_context=lance_mp_context(),  # forkserver: lance's worker context
        shuffle=True,
    )
    batch = next(iter(loader))
    assert batch["state"].shape[0] == 8
    assert len(batch["task"]) == 8


def _language_msg(role, content, style, tool_calls=None):
    return {
        "role": role,
        "content": content,
        "style": style,
        "timestamp": 0.0,
        "camera": None,
        "tool_calls": tool_calls,
    }


def add_language_columns(src_root: Path) -> None:
    """Add persistent + event language columns, hitting every shape the loader must
    round-trip: populated, tool_calls (JSON), empty list, and null."""
    for f in sorted((src_root / "data").rglob("*.parquet")):
        table = pq.read_table(f)
        persistent, events = [], []
        for idx in table.column("index").to_pylist():
            if idx % 7 == 5:
                persistent.append(None)
            elif idx % 3 == 0:
                persistent.append([_language_msg("user", f"subtask {idx}", "subtask")])
            else:
                persistent.append([])
            events.append(
                [_language_msg("assistant", "on it", "interjection", tool_calls=['{"text": "on it"}'])]
                if idx % 4 == 0
                else []
            )
        # pa.array can't build the nested JSON extension values; use storage types.
        table = table.append_column(
            LANGUAGE_PERSISTENT, pa.array(persistent, type=_storage_type(language_persistent_arrow_type()))
        )
        table = table.append_column(
            LANGUAGE_EVENTS, pa.array(events, type=_storage_type(language_events_arrow_type()))
        )
        pq.write_table(table, f)

    info_path = src_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"].update(language_feature_info())
    info_path.write_text(json.dumps(info, indent=4))


def test_language_columns_parity(tmp_path, lerobot_dataset_factory):
    src_root = tmp_path / "src"
    lerobot_dataset_factory(
        root=src_root, total_episodes=3, total_frames=90, use_videos=False, camera_features={}
    )
    add_language_columns(src_root)
    lance_root = tmp_path / "lance"
    convert(lance_root, root=src_root)

    upstream = LeRobotDataset(DUMMY_REPO_ID, root=src_root)
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root)
    assert lance_ds.meta.has_language_columns

    # indices hit every shape: both populated (0,12), persistent-only (33),
    # events-only (4), persistent-null (5), neither (1)
    for idx in [0, 1, 4, 5, 12, 33, len(upstream) - 1]:
        up_item, lance_item = upstream[idx], lance_ds[idx]
        for col in LANGUAGE_COLUMNS:
            assert lance_item[col] == up_item[col], (idx, col)
        assert_items_equal(
            {k: v for k, v in lance_item.items() if k not in LANGUAGE_COLUMNS},
            {k: v for k, v in up_item.items() if k not in LANGUAGE_COLUMNS},
        )


def test_materialize_meta_rejects_escaping_paths(tmp_path):
    """A meta table entry with an absolute or traversing path must not escape the cache."""
    for i, bad in enumerate(["/etc/passwd", "../../etc/passwd"]):
        db = lancedb.connect(str(tmp_path / f"db_{i}"))
        db.create_table(
            lance_utils.META_TABLE,
            pa.table(
                {"path": [bad], "data": [b"x"]},
                schema=pa.schema([("path", pa.string()), ("data", pa.large_binary())]),
            ),
        )
        with pytest.raises(ValueError, match="escapes the cache directory"):
            lance_utils._materialize_meta(db, tmp_path / f"root_{i}")
