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

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import datasets

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.annotations import (
    AnnotationWriter,
    add_annotations,
    list_annotations,
    load_annotations,
    load_annotations_info,
    merge_annotations,
    push_annotations_to_hub,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

pytestmark = pytest.mark.dataset

COLUMN = "progress"
INFO = {"source": "test/model"}


class FakeDataset(LeRobotDataset):
    _storage_root = None
    reader = SimpleNamespace()
    writer = None

    def __init__(
        self,
        root,
        lengths: list[int],
        *,
        episodes: list[int] | None = None,
        storage_format: str = "lerobot",
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        bounds = []
        start = 0
        for episode_index, length in enumerate(lengths):
            bounds.append(
                {
                    "episode_index": episode_index,
                    "dataset_from_index": start,
                    "dataset_to_index": start + length,
                }
            )
            start += length
        self.root = root
        self.repo_id = "test/dataset"
        self.revision = "dataset-revision"
        self.episodes = episodes
        self.meta = SimpleNamespace(
            episodes=bounds,
            total_episodes=len(bounds),
            total_frames=start,
            storage_format=storage_format,
        )


class RecordingFunction:
    """Wrap an annotation function and record the episodes it is called with."""

    def __init__(self, function: Callable[[int], tuple[Any, Any]]) -> None:
        self._function = function
        self.calls: list[int] = []

    def __call__(self, episode_index: int) -> tuple[Any, Any]:
        self.calls.append(episode_index)
        return self._function(episode_index)


def dense(dataset: FakeDataset) -> RecordingFunction:
    def function(episode_index: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        episode = dataset.meta.episodes[episode_index]
        length = episode["dataset_to_index"] - episode["dataset_from_index"]
        return np.arange(length, dtype=np.int64), {COLUMN: np.linspace(0.0, 1.0, length, dtype=np.float32)}

    return RecordingFunction(function)


def test_annotations_write_load_and_list_named_tables(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [3, 4])

    def function(episode_index: int):
        if episode_index == 0:
            return dense(dataset)(episode_index)
        return np.asarray([0, 2], dtype=np.int64), {COLUMN: np.asarray([0.25, 0.75], dtype=np.float32)}

    add_annotations(dataset, "experiment", function, info=INFO)

    table = load_annotations(dataset, "experiment")
    assert isinstance(table, datasets.Dataset)
    assert list(table["index"]) == [0, 1, 2, 3, 5]
    assert list(table["episode_index"]) == [0, 0, 0, 1, 1]
    assert list(table["frame_index"]) == [0, 1, 2, 0, 2]
    assert table[0][COLUMN] == 0.0
    assert load_annotations_info(dataset, "experiment") == INFO
    output_path = dataset.root / "annotations" / "experiment.parquet"
    assert pq.ParquetFile(output_path).metadata.num_row_groups == 2
    assert not output_path.with_name(".experiment.parquet.parts").exists()

    add_annotations(dataset, "other", dense(dataset), info=INFO)
    assert list_annotations(dataset) == ["experiment", "other"]


def test_annotations_load_selected_columns(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2])
    add_annotations(dataset, "test", dense(dataset), info=INFO)

    table = load_annotations(dataset, "test", columns=["episode_index", COLUMN])

    assert table.column_names == ["episode_index", COLUMN]


def test_annotations_accept_repeated_frames_and_string_columns(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [3])

    def function(episode_index: int):
        return np.asarray([0, 0, 2], dtype=np.int64), {"text": ["a", "b", "c"]}

    add_annotations(dataset, "language", function, info=INFO)

    table = load_annotations(dataset, "language")
    assert list(table["frame_index"]) == [0, 0, 2]
    assert list(table["text"]) == ["a", "b", "c"]


def test_annotations_upload_uses_owned_hub_location(monkeypatch, tmp_path):
    import lerobot.datasets.annotations as annotations

    dataset = FakeDataset(tmp_path / "dataset", [1])
    add_annotations(dataset, "test", dense(dataset), info=INFO)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        annotations,
        "HfApi",
        lambda: SimpleNamespace(upload_file=lambda **kwargs: captured.update(kwargs)),
    )

    push_annotations_to_hub(dataset, "test")

    assert captured["path_or_fileobj"] == dataset.root / "annotations" / "test.parquet"
    assert captured["path_in_repo"] == "annotations/test.parquet"
    assert captured["repo_id"] == dataset.repo_id
    assert captured["repo_type"] == "dataset"


def test_annotations_resume_skips_atomic_completed_parts(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])

    def fail_second(episode_index: int):
        if episode_index == 1:
            raise RuntimeError("model failed")
        return dense(dataset)(episode_index)

    with pytest.raises(RuntimeError, match="model failed"):
        add_annotations(dataset, "test", fail_second, info=INFO)

    parts_dir = dataset.root / "annotations" / ".test.parquet.parts"
    assert (parts_dir / "episode-000000.parquet").is_file()
    assert not (parts_dir / "episode-000001.parquet").exists()

    function = dense(dataset)
    add_annotations(dataset, "test", function, info=INFO)

    assert function.calls == [1]
    assert len(load_annotations(dataset, "test")) == 4
    assert not parts_dir.exists()


def test_annotation_writer_reports_pending_episodes(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2, 2])
    writer = AnnotationWriter(dataset, "test", info=INFO, episodes=[2, 0])
    assert writer.pending_episodes == (0, 2)

    writer.write_episode(2, np.arange(2), {COLUMN: np.zeros(2, dtype=np.float32)})
    assert writer.pending_episodes == (0,)

    with pytest.raises(RuntimeError, match="missing episodes: \\[0\\]"):
        writer.finalize()

    resumed = AnnotationWriter(dataset, "test", info=INFO, episodes=[0, 2])
    assert resumed.pending_episodes == (0,)

    with pytest.raises(ValueError, match="not in the selected episodes"):
        resumed.write_episode(1, np.arange(2), {COLUMN: np.zeros(2, dtype=np.float32)})


def test_annotations_reject_published_extension_without_overwrite(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    add_annotations(dataset, "test", dense(dataset), info=INFO, episodes=[0])

    function = dense(dataset)
    with pytest.raises(FileExistsError, match="Published annotations"):
        add_annotations(dataset, "test", function, info=INFO, episodes=[0, 1])

    assert function.calls == []


def test_annotations_overwrite_keeps_published_output_until_success(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2])
    add_annotations(dataset, "test", dense(dataset), info=INFO)
    original = load_annotations(dataset, "test")

    def fail(episode_index: int):
        raise RuntimeError("replacement failed")

    with pytest.raises(RuntimeError, match="replacement failed"):
        add_annotations(dataset, "test", fail, info=INFO, overwrite=True)

    current = load_annotations(dataset, "test")
    assert current[:] == original[:]


def test_annotations_resume_rejects_changed_selection(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])

    def fail_second(episode_index: int):
        if episode_index == 1:
            raise RuntimeError("model failed")
        return dense(dataset)(episode_index)

    with pytest.raises(RuntimeError, match="model failed"):
        add_annotations(dataset, "test", fail_second, info=INFO, episodes=[0, 1])

    with pytest.raises(ValueError, match="info or episode selection changed"):
        add_annotations(dataset, "test", dense(dataset), info=INFO, episodes=[0])


def test_annotations_resume_rejects_changed_info(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])

    def fail_second(episode_index: int):
        if episode_index == 1:
            raise RuntimeError("model failed")
        return dense(dataset)(episode_index)

    with pytest.raises(RuntimeError, match="model failed"):
        add_annotations(dataset, "test", fail_second, info=INFO)

    with pytest.raises(ValueError, match="info or episode selection changed"):
        add_annotations(dataset, "test", dense(dataset), info={"source": "other/model"})


def test_annotations_explicit_episodes_must_belong_to_view(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2, 2], episodes=[0, 2])
    function = dense(dataset)

    with pytest.raises(ValueError, match="visible in the current dataset view"):
        add_annotations(dataset, "test", function, info=INFO, episodes=[1])

    assert function.calls == []


@pytest.mark.parametrize("name", ["", ".", "..", "nested/name", r"nested\name"])
def test_annotations_reject_invalid_names(tmp_path, name):
    dataset = FakeDataset(tmp_path / "dataset", [1])

    with pytest.raises(ValueError, match="Annotation name"):
        add_annotations(dataset, name, dense(dataset), info=INFO)


def test_annotations_reject_unsupported_storage_before_work(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [1], storage_format="lance")
    function = dense(dataset)

    with pytest.raises(NotImplementedError, match="storage_format"):
        add_annotations(dataset, "test", function, info=INFO)

    assert function.calls == []


def test_annotations_reject_snapshot_cache_before_work(monkeypatch, tmp_path):
    import lerobot.datasets.annotations as annotations

    cache_root = tmp_path / "hub"
    dataset = FakeDataset(cache_root / "snapshots" / "revision", [1])
    function = dense(dataset)
    monkeypatch.setattr(annotations, "HF_LEROBOT_HUB_CACHE", cache_root)

    with pytest.raises(NotImplementedError, match="snapshot cache"):
        add_annotations(dataset, "test", function, info=INFO)

    assert function.calls == []


@pytest.mark.parametrize(
    "frame_indices,columns,error",
    [
        (np.asarray([0]), {"index": np.asarray([0.5], dtype=np.float32)}, "reserved columns"),
        (np.asarray([0]), {COLUMN: np.asarray([0.5, 0.6], dtype=np.float32)}, "must have 1 values"),
        (np.asarray([0]), {}, "At least one column"),
        (np.asarray([0.0]), {COLUMN: np.asarray([0.5], dtype=np.float32)}, "integer array"),
        (np.asarray([1, 0]), {COLUMN: np.asarray([0.5, 0.6], dtype=np.float32)}, "non-decreasing"),
        (
            np.asarray([1, 0], dtype=np.uint64),
            {COLUMN: np.asarray([0.5, 0.6], dtype=np.float32)},
            "non-decreasing",
        ),
        (np.asarray([2]), {COLUMN: np.asarray([0.5], dtype=np.float32)}, "non-decreasing and in \\[0, 2\\)"),
    ],
)
def test_annotations_reject_structurally_invalid_rows(tmp_path, frame_indices, columns, error):
    dataset = FakeDataset(tmp_path / "dataset", [2])

    with pytest.raises(ValueError, match=error):
        add_annotations(dataset, "test", lambda episode_index: (frame_indices, columns), info=INFO)


def test_annotations_reject_columns_that_change_between_episodes(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [1, 1])

    def function(episode_index: int):
        dtype = np.float32 if episode_index == 0 else np.float64
        return np.asarray([0]), {COLUMN: np.asarray([0.5], dtype=dtype)}

    with pytest.raises(ValueError, match="differ from earlier episodes"):
        add_annotations(dataset, "test", function, info=INFO)


def test_annotations_store_dataset_identity_and_info_under_separate_keys(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [1])
    add_annotations(dataset, "test", dense(dataset), info={"dataset": "cannot-clobber"})

    metadata = pq.read_schema(dataset.root / "annotations" / "test.parquet").metadata
    assert set(metadata) == {
        b"lerobot.annotations.format",
        b"lerobot.annotations.schema_version",
        b"lerobot.annotations.dataset",
        b"lerobot.annotations.info",
    }
    dataset_block = json.loads(metadata[b"lerobot.annotations.dataset"])
    assert dataset_block["repo_id"] == "test/dataset"
    assert dataset_block["total_frames"] == 1
    assert load_annotations_info(dataset, "test") == {"dataset": "cannot-clobber"}


def test_load_annotations_rejects_parquet_without_annotation_metadata(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [1])
    path = dataset.root / "annotations" / "other.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"value": [1.0]}), path)

    with pytest.raises(ValueError, match="Not a LeRobot annotation table"):
        load_annotations(dataset, "other")


def test_merge_annotations_combines_disjoint_episodes(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 3, 1])
    add_annotations(dataset, "part-1", dense(dataset), info=INFO, episodes=[2, 0])
    add_annotations(dataset, "part-0", dense(dataset), info=INFO, episodes=[1])

    merge_annotations(dataset, ["part-1", "part-0"], name="test")

    table = load_annotations(dataset, "test")
    assert list(table["episode_index"]) == [0, 0, 1, 1, 1, 2]
    assert list(table["frame_index"]) == [0, 1, 0, 1, 2, 0]
    assert table[1][COLUMN] == 1.0
    assert load_annotations_info(dataset, "test") == INFO
    assert pq.read_schema(dataset.root / "annotations" / "test.parquet").equals(
        pq.read_schema(dataset.root / "annotations" / "part-0.parquet"), check_metadata=True
    )
    assert pq.ParquetFile(dataset.root / "annotations" / "test.parquet").metadata.num_row_groups == 3
    assert list_annotations(dataset) == ["part-0", "part-1", "test"]


def test_merge_annotations_keeps_row_order_within_an_episode(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [3, 3])

    def function(episode_index: int):
        return np.asarray([0, 0, 1, 1, 2]), {"text": [f"{episode_index}-{row}" for row in range(5)]}

    add_annotations(dataset, "a", function, info=INFO, episodes=[1])
    add_annotations(dataset, "b", function, info=INFO, episodes=[0])

    merge_annotations(dataset, ["a", "b"], name="test")

    table = load_annotations(dataset, "test")
    assert list(table["text"]) == [f"{episode}-{row}" for episode in (0, 1) for row in range(5)]


def test_merge_annotations_rejects_shared_episodes(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    add_annotations(dataset, "a", dense(dataset), info=INFO, episodes=[0, 1])
    add_annotations(dataset, "b", dense(dataset), info=INFO, episodes=[1])

    with pytest.raises(ValueError, match="repeat episodes"):
        merge_annotations(dataset, ["a", "b"], name="test")


def test_merge_annotations_rejects_different_info(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    add_annotations(dataset, "a", dense(dataset), info=INFO, episodes=[0])
    add_annotations(dataset, "b", dense(dataset), info={"source": "other/model"}, episodes=[1])

    with pytest.raises(ValueError, match="different columns or info"):
        merge_annotations(dataset, ["a", "b"], name="test")


def test_merge_annotations_replaces_a_table_only_with_overwrite(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    add_annotations(dataset, "a", dense(dataset), info=INFO, episodes=[0])
    add_annotations(dataset, "b", dense(dataset), info=INFO, episodes=[1])
    merge_annotations(dataset, ["a", "b"], name="test")

    with pytest.raises(FileExistsError, match="Published annotations"):
        merge_annotations(dataset, ["a", "b"], name="test")
    merge_annotations(dataset, ["a", "b"], name="test", overwrite=True)
