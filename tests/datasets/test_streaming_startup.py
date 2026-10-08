# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from datasets import Dataset

from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.streaming.manifest import EpisodeVideoManifest, VideoFileRecord
from lerobot.streaming.mp4 import (
    Mp4Index,
    Mp4SampleSlice,
    _make_moov,
    parse_mp4_index,
    synthesize_mp4,
    synthesized_mp4_size,
)
from tests.datasets.test_episode_video_streaming import _minimal_mp4


def test_manifest_finishes_each_file_before_touching_the_next(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rank ordering must not cause repeated random walks over the mapped sample tables."""
    paths = ["0.mp4", "1.mp4"]
    records = {
        path: VideoFileRecord(path, 20_000, parse_mp4_index(path, _minimal_mp4([10_000, 10_050, 10_025])))
        for path in paths
    }
    meta = SimpleNamespace(
        total_episodes=4,
        video_keys=["camera"],
        ensure_readable=lambda: None,
        episodes=Dataset.from_dict(
            {
                "videos/camera/chunk_index": [0] * 4,
                "videos/camera/file_index": [0, 1, 0, 1],
                "videos/camera/from_timestamp": [0.0] * 4,
                "videos/camera/to_timestamp": [2.0] * 4,
            }
        ),
        video_path="{file_index}.mp4",
        get_video_file_path=lambda ep, key: paths[ep % 2],
    )
    monkeypatch.setattr(EpisodeVideoManifest, "load_file_sidecar", lambda *args, **kwargs: records)
    touched: list[str] = []
    original = Mp4Index.sample_slice

    def trace(index: Mp4Index, *args, **kwargs):
        touched.append(index.file_path)
        return original(index, *args, **kwargs)

    monkeypatch.setattr(Mp4Index, "sample_slice", trace)
    manifest = EpisodeVideoManifest.build(meta, "unused", sidecar_path="unused")
    assert touched == ["0.mp4", "0.mp4", "1.mp4", "1.mp4"]
    expected = len(synthesize_mp4(records["0.mp4"].mp4, original(records["0.mp4"].mp4, 0.0, 2.0), b"x" * 60))

    def unexpected(*args, **kwargs):
        pytest.fail("Admission sizes must reuse the file-local pass, not reread mapped arrays")

    monkeypatch.setattr("lerobot.streaming.manifest.synthesized_mp4_size", unexpected)
    assert [manifest.episode_byte_size(ep) for ep in range(4)] == [expected] * 4
    assert [manifest.lookup(ep, "camera").file_id for ep in range(4)] == [0, 1, 0, 1]


def test_sidecar_spec_does_not_materialize_full_episode_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.datasets.streaming_sidecar import make_sidecar_spec

    def unexpected(*args, **kwargs):
        pytest.fail("Path discovery must project columns instead of materializing every episode row")

    meta = SimpleNamespace(
        repo_id="owner/data",
        revision="a" * 40,
        total_episodes=3,
        video_keys=["camera"],
        ensure_readable=lambda: None,
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        episodes=Dataset.from_dict(
            {
                "videos/camera/chunk_index": [0, 0, 1],
                "videos/camera/file_index": [2, 2, 4],
                "unrelated": [[1, 2, 3]] * 3,
            }
        ),
        get_video_file_path=unexpected,
    )
    spec = make_sidecar_spec(meta, f"hf://datasets/owner/data@{'a' * 40}")
    assert spec.source_files == (
        ("videos/camera/chunk-000/file-002.mp4", None),
        ("videos/camera/chunk-001/file-004.mp4", None),
    )


def test_frame_counts_project_metadata_once(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = object.__new__(StreamingLeRobotDataset)
    stream.meta = SimpleNamespace(
        episodes=Dataset.from_dict(
            {
                "dataset_from_index": [0, 12, 40],
                "dataset_to_index": [12, 40, 41],
                "unrelated": [[1, 2, 3]] * 3,
            }
        )
    )
    original = Dataset.__getitem__

    def reject_rows(table, key):
        if isinstance(key, int):
            pytest.fail("Rank planning must not materialize a full metadata row per episode")
        return original(table, key)

    monkeypatch.setattr(Dataset, "__getitem__", reject_rows)
    assert [stream._episode_frame_count(ep) for ep in [2, 0, 1, 2]] == [1, 12, 28, 1]


@pytest.mark.parametrize("composition", [[0, 0, 0], [1000, 2000, 0], [-1000, 1000, 0]])
@pytest.mark.parametrize("sync", [[], [0], [0, 1, 2]])
def test_size_count_does_not_serialize_sample_tables(
    monkeypatch: pytest.MonkeyPatch, composition: list[int], sync: list[int]
) -> None:
    index = replace(
        parse_mp4_index("test.mp4", _minimal_mp4([10_000, 10_050, 10_025])),
        sample_composition_offsets=np.array(composition, dtype=np.int64),
        sync_samples=np.array(sync, dtype=np.int64),
        sample_durations=np.array([1000, 2000, 1000], dtype=np.int64),
    )
    sample_slice = index.sample_slice(0.0, 2.0, keyframe_pad_s=0, keyframe_pad_fraction=0)
    expected = len(synthesize_mp4(index, sample_slice, b"x" * sample_slice.byte_length))

    def unexpected(*args, **kwargs):
        pytest.fail("Counting bytes must not serialize movie/sample-table boxes")

    monkeypatch.setattr("lerobot.streaming.mp4._make_moov", unexpected)
    assert synthesized_mp4_size(index, sample_slice) == expected


@pytest.mark.parametrize("last_offset", [100, 2**32 - 1024, 2**32 - 100, 2**32, 2**32 + 100])
def test_size_count_handles_co64_and_extended_mdat(last_offset: int) -> None:
    """Compare to actual serialized headers without allocating a four-GiB payload."""
    index = replace(
        parse_mp4_index("test.mp4", _minimal_mp4([10_000, 10_050, 10_025])),
        sample_offsets=np.array([0, 50, last_offset], dtype=np.int64),
    )
    span = Mp4SampleSlice(0, 2, 0, last_offset + 10, 0.0)
    args = (
        index,
        index.sample_durations,
        index.sample_sizes,
        index.sample_offsets,
        index.sync_samples + 1,
        index.sample_composition_offsets,
    )
    first_header = _make_moov(*args, mdat_data_offset=0)
    mdat_header = 8 if span.byte_length + 8 <= 0xFFFFFFFF else 16
    final_header = _make_moov(*args, mdat_data_offset=len(index.ftyp) + len(first_header) + mdat_header)
    assert (
        synthesized_mp4_size(index, span)
        == len(index.ftyp) + len(final_header) + mdat_header + span.byte_length
    )
