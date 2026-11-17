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
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import datasets
from datasets.table import MemoryMappedTable
from httpx import Request, Response
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
from lerobot.datasets.scores import (
    ScoreWriter,
    add_scores,
    list_scores,
    load_scores,
    load_scores_info,
    merge_scores,
    push_scores_to_hub,
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
    """Wrap a score function and record the episodes it is called with."""

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


def test_scores_write_load_and_list_by_name(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [3, 4])

    def function(episode_index: int):
        if episode_index == 0:
            return dense(dataset)(episode_index)
        return np.asarray([0, 2], dtype=np.int64), {COLUMN: np.asarray([0.25, 0.75], dtype=np.float32)}

    add_scores(dataset, "experiment", function, info=INFO)

    table = load_scores(dataset, "experiment")
    assert isinstance(table, datasets.Dataset)
    assert list(table["index"]) == [0, 1, 2, 3, 5]
    assert list(table["episode_index"]) == [0, 0, 0, 1, 1]
    assert list(table["frame_index"]) == [0, 1, 2, 0, 2]
    assert table[0][COLUMN] == 0.0
    assert load_scores_info(dataset, "experiment") == INFO
    output_path = dataset.root / "scores" / "experiment.parquet"
    assert pq.ParquetFile(output_path).metadata.num_row_groups == 2
    assert not output_path.with_name(".experiment.parquet.parts").exists()

    add_scores(dataset, "other", dense(dataset), info=INFO)
    assert list_scores(dataset) == ["experiment", "other"]


def test_scores_load_selected_columns(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2])
    add_scores(dataset, "test", dense(dataset), info=INFO)

    table = load_scores(dataset, "test", columns=["episode_index", COLUMN])

    assert table.column_names == ["episode_index", COLUMN]


def test_score_writer_rejects_dataset_being_recorded(tmp_path):
    dataset = LeRobotDataset.create(
        repo_id="test/dataset",
        fps=10,
        features={"state": {"dtype": "float32", "shape": (2,), "names": None}},
        root=tmp_path / "dataset",
    )

    with pytest.raises(RuntimeError, match="being recorded"):
        ScoreWriter(dataset, "test", info=INFO)

    assert not (dataset.root / "scores").exists()


def test_scores_accept_repeated_frames_and_string_columns(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [3])

    def function(episode_index: int):
        return np.asarray([0, 0, 2], dtype=np.int64), {"text": ["a", "b", "c"]}

    add_scores(dataset, "language", function, info=INFO)

    table = load_scores(dataset, "language")
    assert list(table["frame_index"]) == [0, 0, 2]
    assert list(table["text"]) == ["a", "b", "c"]


@pytest.mark.parametrize("tag_exists", [True, False])
def test_scores_upload_uses_owned_hub_location(monkeypatch, tmp_path, tag_exists):
    import lerobot.datasets.scores as scores

    dataset = FakeDataset(tmp_path / "dataset", [1])
    add_scores(dataset, "test", dense(dataset), info=INFO)
    api = Mock(spec=scores.HfApi)
    api.upload_file.return_value = SimpleNamespace(oid="uploaded-commit")
    if not tag_exists:
        api.delete_tag.side_effect = RevisionNotFoundError(
            "No version tag yet", response=Response(404, request=Request("DELETE", "https://huggingface.co"))
        )
    monkeypatch.setattr(scores, "HfApi", lambda: api)

    push_scores_to_hub(dataset, "test")

    api.upload_file.assert_called_once()
    captured = api.upload_file.call_args.kwargs
    assert captured["path_or_fileobj"] == dataset.root / "scores" / "test.parquet"
    assert captured["path_in_repo"] == "scores/test.parquet"
    assert captured["repo_id"] == dataset.repo_id
    assert captured["repo_type"] == "dataset"
    assert [call[0] for call in api.method_calls] == ["upload_file", "delete_tag", "create_tag"]
    api.delete_tag.assert_called_once_with(dataset.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    api.create_tag.assert_called_once_with(
        dataset.repo_id, tag=CODEBASE_VERSION, revision="uploaded-commit", repo_type="dataset"
    )


def test_scores_upload_failure_leaves_version_tag_unchanged(monkeypatch, tmp_path):
    import lerobot.datasets.scores as scores

    dataset = FakeDataset(tmp_path / "dataset", [1])
    add_scores(dataset, "test", dense(dataset), info=INFO)
    api = Mock(spec=scores.HfApi)
    api.upload_file.side_effect = OSError("Upload failed")
    monkeypatch.setattr(scores, "HfApi", lambda: api)

    with pytest.raises(OSError, match="Upload failed"):
        push_scores_to_hub(dataset, "test")

    api.delete_tag.assert_not_called()
    api.create_tag.assert_not_called()


def test_scores_resume_skips_atomic_completed_parts(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])

    def fail_second(episode_index: int):
        if episode_index == 1:
            raise RuntimeError("model failed")
        return dense(dataset)(episode_index)

    with pytest.raises(RuntimeError, match="model failed"):
        add_scores(dataset, "test", fail_second, info=INFO)

    parts_dir = dataset.root / "scores" / ".test.parquet.parts"
    assert (parts_dir / "episode-000000.parquet").is_file()
    assert not (parts_dir / "episode-000001.parquet").exists()

    function = dense(dataset)
    add_scores(dataset, "test", function, info=INFO)

    assert function.calls == [1]
    assert len(load_scores(dataset, "test")) == 4
    assert not parts_dir.exists()


def test_score_writer_reports_pending_episodes(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2, 2])
    writer = ScoreWriter(dataset, "test", info=INFO, episodes=[2, 0])
    assert writer.pending_episodes == (0, 2)

    writer.write_episode(2, np.arange(2), {COLUMN: np.zeros(2, dtype=np.float32)})
    assert writer.pending_episodes == (0,)

    with pytest.raises(RuntimeError, match="missing episodes: \\[0\\]"):
        writer.finalize()

    resumed = ScoreWriter(dataset, "test", info=INFO, episodes=[0, 2])
    assert resumed.pending_episodes == (0,)

    with pytest.raises(ValueError, match="not in the selected episodes"):
        resumed.write_episode(1, np.arange(2), {COLUMN: np.zeros(2, dtype=np.float32)})


def test_scores_reject_published_extension_without_overwrite(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    add_scores(dataset, "test", dense(dataset), info=INFO, episodes=[0])

    function = dense(dataset)
    with pytest.raises(FileExistsError, match="Published scores"):
        add_scores(dataset, "test", function, info=INFO, episodes=[0, 1])

    assert function.calls == []


def test_score_writer_finalize_is_idempotent(tmp_path: Path) -> None:
    dataset = FakeDataset(tmp_path / "dataset", [2])
    writer = ScoreWriter(dataset, "test", info=INFO)
    writer.write_episode(0, *dense(dataset)(0))

    writer.finalize()
    published = writer.output_path.read_bytes()
    writer.finalize()

    assert writer.output_path.read_bytes() == published
    assert not writer.parts_dir.exists()


def test_score_writer_rejects_writes_after_finalization(tmp_path: Path) -> None:
    dataset = FakeDataset(tmp_path / "dataset", [2])
    writer = ScoreWriter(dataset, "test", info=INFO)
    writer.write_episode(0, *dense(dataset)(0))
    writer.finalize()

    with pytest.raises(RuntimeError, match="after finalization"):
        writer.write_episode(0, *dense(dataset)(0))

    assert not writer.parts_dir.exists()


def test_score_writer_finalize_retries_after_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lerobot.datasets.scores as scores

    dataset = FakeDataset(tmp_path / "dataset", [2])
    add_scores(dataset, "test", dense(dataset), info=INFO)
    writer = ScoreWriter(dataset, "test", info=INFO, overwrite=True)
    writer.write_episode(0, np.arange(2), {COLUMN: np.ones(2, dtype=np.float32)})
    published = writer.output_path.read_bytes()

    with monkeypatch.context() as context:
        context.setattr(scores, "_write_episode_tables", Mock(side_effect=OSError("publication failed")))
        with pytest.raises(OSError, match="publication failed"):
            writer.finalize()

    assert writer.output_path.read_bytes() == published
    assert writer.parts_dir.exists()
    writer.finalize()

    assert list(load_scores(dataset, "test")[COLUMN]) == [1.0, 1.0]
    assert not writer.parts_dir.exists()


def test_score_writer_finalize_logs_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import lerobot.datasets.scores as scores

    dataset = FakeDataset(tmp_path / "dataset", [2])
    writer = ScoreWriter(dataset, "test", info=INFO)
    writer.write_episode(0, *dense(dataset)(0))
    cleanup = Mock(side_effect=PermissionError("cleanup denied"))

    with monkeypatch.context() as context:
        context.setattr(scores.shutil, "rmtree", cleanup)
        with caplog.at_level("WARNING", logger="lerobot.datasets.scores"):
            writer.finalize()
            writer.finalize()

    cleanup.assert_called_once_with(writer.parts_dir)
    assert len(load_scores(dataset, "test")) == 2
    assert writer.parts_dir.exists()
    assert "Could not remove score staging directory" in caplog.text
    assert "cleanup denied" in caplog.text


def test_scores_overwrite_keeps_published_output_until_success(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2])
    add_scores(dataset, "test", dense(dataset), info=INFO)
    original = load_scores(dataset, "test")

    def fail(episode_index: int):
        raise RuntimeError("replacement failed")

    with pytest.raises(RuntimeError, match="replacement failed"):
        add_scores(dataset, "test", fail, info=INFO, overwrite=True)

    current = load_scores(dataset, "test")
    assert current[:] == original[:]


@pytest.mark.parametrize("resume", [True, False])
def test_scores_retry_interrupted_overwrite(tmp_path, resume):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    add_scores(dataset, "test", dense(dataset), info=INFO)
    original = load_scores(dataset, "test")[:]
    replacement_info = {"source": "replacement/model"}

    def replacement(episode_index: int):
        assert load_scores(dataset, "test")[:] == original
        assert load_scores_info(dataset, "test") == INFO
        return np.arange(2), {COLUMN: np.full(2, 0.5)}

    def fail_second(episode_index: int):
        if episode_index == 1:
            raise RuntimeError("replacement interrupted")
        return replacement(episode_index)

    with pytest.raises(RuntimeError, match="replacement interrupted"):
        add_scores(dataset, "test", fail_second, info=replacement_info, overwrite=True)

    assert load_scores(dataset, "test")[:] == original
    assert load_scores_info(dataset, "test") == INFO
    function = RecordingFunction(replacement)
    add_scores(dataset, "test", function, info=replacement_info, overwrite=True, resume=resume)

    assert function.calls == ([1] if resume else [0, 1])
    assert list(load_scores(dataset, "test")[COLUMN]) == [0.5] * 4
    assert load_scores_info(dataset, "test") == replacement_info
    assert not (dataset.root / "scores" / ".test.parquet.parts").exists()


def test_scores_overwrite_requires_explicit_restart_for_changed_manifest(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    add_scores(dataset, "test", dense(dataset), info=INFO)
    original = load_scores(dataset, "test")[:]
    writer = ScoreWriter(dataset, "test", info=INFO, overwrite=True)
    writer.write_episode(0, *dense(dataset)(0))
    changed_info = {"source": "another/model"}
    function = dense(dataset)

    with pytest.raises(ValueError, match="info or episode selection changed"):
        add_scores(dataset, "test", function, info=changed_info, overwrite=True)

    assert function.calls == []
    assert load_scores(dataset, "test")[:] == original
    assert ScoreWriter(dataset, "test", info=INFO, overwrite=True).pending_episodes == (1,)
    add_scores(dataset, "test", function, info=changed_info, overwrite=True, resume=False)
    assert function.calls == [0, 1]
    assert load_scores_info(dataset, "test") == changed_info


def test_scores_resume_rejects_changed_selection(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])

    def fail_second(episode_index: int):
        if episode_index == 1:
            raise RuntimeError("model failed")
        return dense(dataset)(episode_index)

    with pytest.raises(RuntimeError, match="model failed"):
        add_scores(dataset, "test", fail_second, info=INFO, episodes=[0, 1])

    with pytest.raises(ValueError, match="info or episode selection changed"):
        add_scores(dataset, "test", dense(dataset), info=INFO, episodes=[0])


def test_scores_resume_rejects_changed_info(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])

    def fail_second(episode_index: int):
        if episode_index == 1:
            raise RuntimeError("model failed")
        return dense(dataset)(episode_index)

    with pytest.raises(RuntimeError, match="model failed"):
        add_scores(dataset, "test", fail_second, info=INFO)

    with pytest.raises(ValueError, match="info or episode selection changed"):
        add_scores(dataset, "test", dense(dataset), info={"source": "other/model"})


@pytest.mark.parametrize(
    "lengths,changed_lengths",
    [([2, 2], [1, 3]), ([2, 2, 2], [2, 1, 3])],
    ids=["completed-episode", "pending-episodes"],
)
def test_scores_resume_rejects_changed_episode_ranges(tmp_path, lengths, changed_lengths):
    dataset = FakeDataset(tmp_path / "dataset", lengths)
    writer = ScoreWriter(dataset, "test", info=INFO)
    writer.write_episode(0, *dense(dataset)(0))
    changed_dataset = FakeDataset(dataset.root, changed_lengths)
    function = dense(changed_dataset)

    with pytest.raises(ValueError, match="dataset, info or episode selection changed"):
        add_scores(changed_dataset, "test", function, info=INFO)

    assert function.calls == []
    assert not writer.output_path.exists()
    assert ScoreWriter(dataset, "test", info=INFO).pending_episodes == tuple(range(1, len(lengths)))


def test_scores_explicit_episodes_must_belong_to_view(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2, 2], episodes=[0, 2])
    function = dense(dataset)

    with pytest.raises(ValueError, match="visible in the current dataset view"):
        add_scores(dataset, "test", function, info=INFO, episodes=[1])

    assert function.calls == []


@pytest.mark.parametrize("name", ["", ".", "..", ".sarm", "nested/name", r"nested\name"])
def test_scores_reject_invalid_names(tmp_path, name):
    dataset = FakeDataset(tmp_path / "dataset", [1])

    with pytest.raises(ValueError, match="Score name"):
        add_scores(dataset, name, dense(dataset), info=INFO)


def test_scores_reject_unsupported_storage_before_work(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [1], storage_format="lance")
    function = dense(dataset)

    with pytest.raises(NotImplementedError, match="storage_format"):
        add_scores(dataset, "test", function, info=INFO)

    assert function.calls == []


def test_scores_reject_snapshot_cache_before_work(monkeypatch, tmp_path):
    import lerobot.datasets.scores as scores

    cache_root = tmp_path / "hub"
    dataset = FakeDataset(cache_root / "snapshots" / "revision", [1])
    function = dense(dataset)
    monkeypatch.setattr(scores, "HF_LEROBOT_HUB_CACHE", cache_root)

    with pytest.raises(NotImplementedError, match="snapshot cache"):
        add_scores(dataset, "test", function, info=INFO)

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
def test_scores_reject_structurally_invalid_rows(tmp_path, frame_indices, columns, error):
    dataset = FakeDataset(tmp_path / "dataset", [2])

    with pytest.raises(ValueError, match=error):
        add_scores(dataset, "test", lambda episode_index: (frame_indices, columns), info=INFO)


def test_scores_reject_columns_that_change_between_episodes(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [1, 1])

    def function(episode_index: int):
        dtype = np.float32 if episode_index == 0 else np.float64
        return np.asarray([0]), {COLUMN: np.asarray([0.5], dtype=dtype)}

    with pytest.raises(ValueError, match="differ from earlier episodes"):
        add_scores(dataset, "test", function, info=INFO)


def test_scores_store_dataset_identity_and_info_under_separate_keys(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [1])
    add_scores(dataset, "test", dense(dataset), info={"dataset": "cannot-clobber"})

    metadata = pq.read_schema(dataset.root / "scores" / "test.parquet").metadata
    assert set(metadata) == {
        b"lerobot.scores.format",
        b"lerobot.scores.schema_version",
        b"lerobot.scores.dataset",
        b"lerobot.scores.info",
    }
    dataset_metadata = json.loads(metadata[b"lerobot.scores.dataset"])
    assert dataset_metadata["repo_id"] == "test/dataset"
    assert dataset_metadata["total_frames"] == 1
    assert load_scores_info(dataset, "test") == {"dataset": "cannot-clobber"}


def test_load_scores_rejects_parquet_without_score_metadata(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [1])
    path = dataset.root / "scores" / "other.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"value": [1.0]}), path)

    with pytest.raises(ValueError, match="Not a LeRobot score file"):
        load_scores(dataset, "other")


def test_merge_scores_combines_disjoint_episodes(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 3, 1])
    add_scores(dataset, "part-1", dense(dataset), info=INFO, episodes=[2, 0])
    add_scores(dataset, "part-0", dense(dataset), info=INFO, episodes=[1])

    merge_scores(dataset, ["part-1", "part-0"], name="test")

    table = load_scores(dataset, "test")
    assert list(table["episode_index"]) == [0, 0, 1, 1, 1, 2]
    assert list(table["frame_index"]) == [0, 1, 0, 1, 2, 0]
    assert table[1][COLUMN] == 1.0
    assert load_scores_info(dataset, "test") == INFO
    assert pq.read_schema(dataset.root / "scores" / "test.parquet").equals(
        pq.read_schema(dataset.root / "scores" / "part-0.parquet"), check_metadata=True
    )
    assert pq.ParquetFile(dataset.root / "scores" / "test.parquet").metadata.num_row_groups == 3
    assert list_scores(dataset) == ["part-0", "part-1", "test"]


def test_merge_scores_keeps_row_order_within_an_episode(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [3, 3])

    def function(episode_index: int):
        return np.asarray([0, 0, 1, 1, 2]), {"text": [f"{episode_index}-{row}" for row in range(5)]}

    add_scores(dataset, "a", function, info=INFO, episodes=[1])
    add_scores(dataset, "b", function, info=INFO, episodes=[0])

    merge_scores(dataset, ["a", "b"], name="test")

    table = load_scores(dataset, "test")
    assert list(table["text"]) == [f"{episode}-{row}" for episode in (0, 1) for row in range(5)]


def test_merge_scores_rejects_shared_episodes(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    add_scores(dataset, "a", dense(dataset), info=INFO, episodes=[0, 1])
    add_scores(dataset, "b", dense(dataset), info=INFO, episodes=[1])

    with pytest.raises(ValueError, match="repeat episodes"):
        merge_scores(dataset, ["a", "b"], name="test")


def test_merge_scores_rejects_different_info(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    add_scores(dataset, "a", dense(dataset), info=INFO, episodes=[0])
    add_scores(dataset, "b", dense(dataset), info={"source": "other/model"}, episodes=[1])

    with pytest.raises(ValueError, match="different columns or info"):
        merge_scores(dataset, ["a", "b"], name="test")


def test_merge_scores_replaces_existing_scores_only_with_overwrite(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    add_scores(dataset, "a", dense(dataset), info=INFO, episodes=[0])
    add_scores(dataset, "b", dense(dataset), info=INFO, episodes=[1])
    merge_scores(dataset, ["a", "b"], name="test")

    with pytest.raises(FileExistsError, match="Published scores"):
        merge_scores(dataset, ["a", "b"], name="test")
    merge_scores(dataset, ["a", "b"], name="test", overwrite=True)


def test_scores_use_global_ids_for_noncontiguous_episode_views(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2] * 10, episodes=[2, 5, 9])
    function = dense(dataset)
    add_scores(dataset, "test", function, info=INFO)

    assert function.calls == [2, 5, 9]
    assert list(load_scores(dataset, "test")["index"]) == [4, 5, 10, 11, 18, 19]
    assert list(load_scores(dataset, "test", episodes=[5, 9])["episode_index"]) == [5, 5, 9, 9]
    with pytest.raises(ValueError, match="not visible"):
        load_scores(dataset, "test", episodes=[0])


@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("info_first", [False, True])
def test_load_scores_downloads_only_named_file_at_dataset_revision(
    tmp_path, monkeypatch, snapshot, info_first
):
    import lerobot.datasets.scores as scores

    original = FakeDataset(tmp_path / "original", [2])
    add_scores(original, "test", dense(original), info=INFO)
    cache = tmp_path / "cache"
    root = cache / "datasets--test--dataset/snapshots/commit-id" if snapshot else tmp_path / "local"
    dataset = FakeDataset(root, [2])

    def download(*args, **kwargs):
        target = root / "scores/test.parquet"
        target.parent.mkdir(parents=True)
        shutil.copyfile(original.root / "scores/test.parquet", target)
        return str(root)

    download_snapshot = Mock(side_effect=download)
    monkeypatch.setattr(scores, "HF_LEROBOT_HUB_CACHE", cache)
    monkeypatch.setattr(scores, "snapshot_download", download_snapshot)

    with pytest.raises(FileNotFoundError):
        load_scores(dataset, "test", download=False)
    download_snapshot.assert_not_called()

    reader = load_scores_info if info_first else load_scores
    reader(dataset, "test", token=False)
    download_snapshot.assert_called_once_with(
        dataset.repo_id,
        repo_type="dataset",
        revision="commit-id" if snapshot else dataset.revision,
        allow_patterns=["scores/test.parquet"],
        token=False,
        **({"cache_dir": cache} if snapshot else {"local_dir": root}),
    )

    assert len(load_scores(dataset, "test", download=False)) == 2
    assert load_scores_info(dataset, "test") == INFO
    assert list_scores(dataset) == ["test"]
    assert len(load_scores(dataset, "test")) == 2
    assert download_snapshot.call_count == 1


def test_load_scores_memory_maps_selected_episodes_and_columns(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2] * 10)
    add_scores(dataset, "test", dense(dataset), info=INFO)

    table = load_scores(dataset, "test", episodes=[2, 5, 9], columns=[COLUMN, "episode_index", "index"])

    assert isinstance(table.data, MemoryMappedTable)
    assert table.cache_files
    assert len(table) == 6
    assert table.column_names == [COLUMN, "episode_index", "index"]
    assert list(table["episode_index"]) == [2, 2, 5, 5, 9, 9]
    assert list(table["index"]) == [4, 5, 10, 11, 18, 19]


@pytest.mark.parametrize("empty_file", [False, True])
def test_load_scores_preserves_types_when_no_rows_match(tmp_path, empty_file):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    frames = np.asarray([] if empty_file else [0, 1], dtype=np.int64)
    add_scores(
        dataset,
        "test",
        lambda _: (frames, {COLUMN: np.ones(len(frames), dtype=np.float32)}),
        info=INFO,
        episodes=[0],
    )

    table = load_scores(dataset, "test", episodes=[1], columns=["index", COLUMN])

    assert len(table) == 0
    assert table.column_names == ["index", COLUMN]
    assert table.features["index"].dtype == "int64"
    assert table.features[COLUMN].dtype == "float32"
