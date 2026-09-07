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
from pathlib import Path

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
from lerobot.datasets import lance_utils
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lance_backend import LanceDatasetReader, lance_mp_context
from lerobot.datasets.language import (
    LANGUAGE_COLUMNS,
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
    language_events_arrow_type,
    language_feature_info,
    language_persistent_arrow_type,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.storage import localize_remote_root
from lerobot.policies.factory import make_policy_config
from tests.fixtures.constants import (
    DUMMY_CAMERA_FEATURES_WITH_DEPTH,
    DUMMY_REPO_ID,
)


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


def test_pickle_and_dataloader(dataset_roots):
    _, lance_root = dataset_roots
    lance_ds = LeRobotDataset(DUMMY_REPO_ID, root=lance_root)
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
