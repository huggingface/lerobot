# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Video addressing through normally constructed readers and on-disk datasets."""

import json
import warnings
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from tests.fixtures.constants import DUMMY_CAMERA_FEATURES, DUMMY_REPO_ID


@pytest.fixture(params=["map", "streaming", "lance"])
def video_reader(request, tmp_path, lerobot_dataset_factory, info_factory, episodes_factory, tasks_factory):
    kind = request.param
    if kind == "lance":
        pytest.importorskip("lancedb")
        pytest.importorskip("lerobot_lancedb")
        pytest.importorskip("torchcodec")
    info = info_factory(
        total_episodes=2,
        total_frames=8,
        total_tasks=1,
        camera_features={"camera": DUMMY_CAMERA_FEATURES["laptop"]},
    )
    tasks = tasks_factory(total_tasks=1)
    episodes = episodes_factory(
        info.features, info.fps, total_episodes=2, total_frames=8, video_keys=["camera"], tasks=tasks
    ).map(
        lambda _row, index: {
            "length": 4,
            "dataset_from_index": index * 4,
            "dataset_to_index": (index + 1) * 4,
            "videos/camera/from_timestamp": index * 4 / info.fps,
            "videos/camera/to_timestamp": (index + 1) * 4 / info.fps,
        },
        with_indices=True,
    )
    source = lerobot_dataset_factory(
        root=tmp_path / "source",
        info=info,
        tasks=tasks,
        episodes_metadata=episodes,
        video_backend="pyav",
    )

    def read(*, frame_offset=0, timestamp_offset=0.0, delta=False, tabular_windows=False):
        # Extend the selected episode with real, contiguous Parquet rows.
        # Precision tests replace only the decoder; ordinary-data tests use real video.
        for path in (source.root / "data").glob("*/*.parquet"):
            table = pq.read_table(path)
            if frame_offset:
                selected = table.filter(pc.equal(table["episode_index"], 1))
                prefix = selected.take(pa.array(np.zeros(frame_offset, dtype=np.int64)))
                selected = pa.concat_tables([prefix, selected])
                for key, values in (
                    ("frame_index", np.arange(len(selected), dtype=np.int64)),
                    ("index", np.arange(len(selected), dtype=np.int64) + 4),
                ):
                    column_index = selected.schema.get_field_index(key)
                    selected = selected.set_column(column_index, selected.schema.field(key), pa.array(values))
                table = pa.concat_tables([table.filter(pc.equal(table["episode_index"], 0)), selected])
            frames = table["frame_index"].to_numpy()
            times = (frames / source.fps + timestamp_offset).astype(np.float32)
            for key, values in (("frame_index", frames), ("timestamp", times)):
                column_index = table.schema.get_field_index(key)
                table = table.set_column(column_index, table.schema.field(key), pa.array(values))
            pq.write_table(table, path)
        if frame_offset:
            for path in (source.root / "meta" / "episodes").glob("*/*.parquet"):
                table = pq.read_table(path)
                rows = table.to_pylist()
                for row in rows:
                    if row["episode_index"] == 1:
                        row["length"] += frame_offset
                        row["dataset_to_index"] += frame_offset
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
            info_path = source.root / "meta" / "info.json"
            info = json.loads(info_path.read_text())
            info["total_frames"] += frame_offset
            info_path.write_text(json.dumps(info))
        root = source.root
        if kind == "lance":
            from lerobot_lancedb.convert import convert

            root = tmp_path / "lance"
            convert(root, root=source.root)
        kwargs = {
            "episodes": [1],
            "return_uint8": True,
            "delta_timestamps": {"camera": [-1 / source.fps, 0.0, 1 / source.fps]} if delta else None,
        }
        if tabular_windows:
            kwargs["delta_timestamps"] = {
                "frame_index": [-1 / source.fps, 0.0],
                "timestamp": [-1 / source.fps, 0.0],
            }
        if kind == "streaming":
            dataset = StreamingLeRobotDataset(
                DUMMY_REPO_ID, root=root, shuffle=False, video_backend="pyav", decode_threads=1, **kwargs
            )
            with closing(iter(dataset)) as iterator:
                if not frame_offset:
                    return list(iterator)
                items = []
                for item in iterator:
                    items.append(item)
                    if abs(float(item["timestamp"]) - int(item["frame_index"]) / source.fps) > 1e-4:
                        break
                return items
        dataset = LeRobotDataset(DUMMY_REPO_ID, root=root, video_backend="pyav", **kwargs)
        try:
            return dataset.__getitems__(list(range(frame_offset, len(dataset))))
        finally:
            if kind == "lance":
                dataset.reader.close()

    return kind, source, read


def _capture_decode_queries(monkeypatch, kind, fps):
    queries = []

    def capture(timestamps):
        queries.append(list(timestamps))
        return torch.zeros((len(timestamps), 3, 64, 96), dtype=torch.uint8)

    if kind == "map":
        monkeypatch.setattr(
            "lerobot.datasets.dataset_reader.decode_video_frames",
            lambda _path, times, *_args, **_kwargs: capture(times),
        )
    elif kind == "streaming":
        monkeypatch.setattr(
            "lerobot.streaming.episode_cache.EpisodeByteCache.get_frames",
            lambda _self, _episode, _key, times: capture(times),
        )
    else:
        # Keep the real Lance row/blob fetch and query tolerance check. A decoder
        # oracle supplies exact grid PTS: rounded float32 queries fail that check.
        class Decoder:
            metadata = SimpleNamespace(average_fps=fps)

            def __init__(self, *_args, **_kwargs):
                pass

            def get_frames_at(self, *, indices):
                times = [index / fps for index in indices]
                return SimpleNamespace(
                    data=capture(times), pts_seconds=torch.tensor(times, dtype=torch.float64)
                )

        monkeypatch.setattr("torchcodec.decoders.VideoDecoder", Decoder)
    return queries


@pytest.mark.parametrize("delta", [False, True])
def test_video_queries_survive_float32_rounding(video_reader, monkeypatch, delta):
    kind, source, read = video_reader
    offset = 100004
    assert abs(float(np.float32(offset / source.fps)) - offset / source.fps) > 1e-4
    queries = _capture_decode_queries(monkeypatch, kind, source.fps)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        items = read(frame_offset=offset, delta=delta)

    episode = source.meta.episodes[1]
    length = int(episode["length"])
    assert items
    assert any(abs(float(item["timestamp"]) - int(item["frame_index"]) / source.fps) > 1e-4 for item in items)
    base = float(episode["videos/camera/from_timestamp"])
    expected = []
    for item in items:
        anchor = int(item["frame_index"])
        window = [max(0, anchor - 1), anchor, min(offset + length - 1, anchor + 1)] if delta else [anchor]
        expected.append([base + value / source.fps for value in window])
    if kind == "lance":
        # Lance may deduplicate and batch overlapping windows before decoding.
        np.testing.assert_allclose(
            sorted({value for query in queries for value in query}),
            sorted({value for query in expected for value in query}),
            rtol=0,
            atol=1e-12,
        )
        return
    # Closing a streaming iterator may finish its already scheduled decode queue.
    assert len(queries) >= len(expected)
    queries = queries[: len(expected)]
    for actual, target in zip(queries, expected, strict=True):
        np.testing.assert_allclose(actual, target, rtol=0, atol=1e-12)


def test_inconsistent_stored_timestamps_warn(video_reader, monkeypatch):
    kind, source, read = video_reader
    queries = _capture_decode_queries(monkeypatch, kind, source.fps)
    with pytest.warns(RuntimeWarning, match="Stored timestamp differs from frame_index / fps"):
        items = read(timestamp_offset=0.05, delta=True)
    assert len(items) == int(source.meta.episodes[1]["length"])
    if kind == "lance":
        return  # Its real query-to-PTS check runs against the decoder oracle.
    base = float(source.meta.episodes[1]["videos/camera/from_timestamp"])
    anchor = int(items[0]["frame_index"])
    window = [max(0, anchor - 1), anchor, min(len(items) - 1, anchor + 1)]
    np.testing.assert_allclose(
        queries[0], [base + value / source.fps for value in window], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("delta", [False, True])
def test_ordinary_video_frames_are_unchanged(video_reader, delta):
    """Use real decoders, episode filtering, shared-video offsets, and padded windows."""
    _kind, source, read = video_reader
    oracle = LeRobotDataset(DUMMY_REPO_ID, root=source.root, return_uint8=True, video_backend="pyav")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        items = read(delta=delta)
    start = int(source.meta.episodes[1]["dataset_from_index"])
    for item in items:
        index = int(item["frame_index"])
        window = [max(0, index - 1), index, min(len(items) - 1, index + 1)] if delta else [index]
        expected = torch.stack([oracle[value + start]["camera"] for value in window])
        if not delta:
            expected = expected.squeeze(0)
        torch.testing.assert_close(item["camera"], expected, rtol=0, atol=0)


@pytest.mark.parametrize("fps", [10, 30, 50])
def test_float32_rounding_allowance(fps):
    from lerobot.datasets.utils import frame_indices_to_timestamps

    frames = [0, 1, 61441, 100004, 2**24]
    expected = [frame / fps for frame in frames]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = frame_indices_to_timestamps(
            frames, fps, stored_timestamps=np.asarray(expected, dtype=np.float32).tolist(), tolerance_s=1e-4
        )
    assert actual == expected


def test_tabular_time_windows_do_not_replace_video_anchor(video_reader):
    _kind, source, read = video_reader
    items = read(tabular_windows=True)
    oracle = LeRobotDataset(DUMMY_REPO_ID, root=source.root, return_uint8=True, video_backend="pyav")
    start = int(source.meta.episodes[1]["dataset_from_index"])
    for item in items:
        anchor = int(item["frame_index"][-1])
        assert item["timestamp"].shape == (2,)
        torch.testing.assert_close(item["camera"], oracle[start + anchor]["camera"], rtol=0, atol=0)


@pytest.mark.parametrize("timestamp", [0.05, float("nan"), float("inf"), -float("inf")])
def test_invalid_stored_timestamp_warns(timestamp):
    from lerobot.datasets.utils import frame_indices_to_timestamps

    with pytest.warns(RuntimeWarning, match="Stored timestamp differs"):
        assert frame_indices_to_timestamps([0], 30, stored_timestamps=[timestamp], tolerance_s=1e-4) == [0.0]
