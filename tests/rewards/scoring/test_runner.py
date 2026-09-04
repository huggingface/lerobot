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

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.rewards.scoring import (
    FrameSignals,
    SignalDescriptor,
    get_scoring_provenance,
    get_signal_descriptors,
    read_frame_signals,
    score_dataset,
)

pq = pytest.importorskip("pyarrow.parquet")

PROGRESS_NAME = "reward.test.progress"
PROGRESS_DESCRIPTOR = SignalDescriptor(
    description="Normalized task progress.",
    direction="higher",
    bounds=(0.0, 1.0),
)


class FakeDataset:
    def __init__(self, lengths: list[int]) -> None:
        start = 0
        episodes = []
        for episode_index, length in enumerate(lengths):
            episodes.append(
                {
                    "episode_index": episode_index,
                    "dataset_from_index": start,
                    "dataset_to_index": start + length,
                }
            )
            start += length
        self.meta = SimpleNamespace(total_episodes=len(episodes), episodes=episodes)
        self.episodes = None


def dense_scorer(dataset: FakeDataset, episode_index: int) -> FrameSignals:
    episode = dataset.meta.episodes[episode_index]
    length = episode["dataset_to_index"] - episode["dataset_from_index"]
    return FrameSignals(
        frame_indices=np.arange(length, dtype=np.int64),
        signals={PROGRESS_NAME: np.linspace(0.0, 1.0, length, dtype=np.float32)},
        descriptors={PROGRESS_NAME: PROGRESS_DESCRIPTOR},
    )


def test_score_dataset_writes_dense_and_sparse_signals_with_roundtripped_metadata(tmp_path):
    dataset = FakeDataset([3, 4])
    output_path = tmp_path / "signals.parquet"

    def scorer(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        if episode_index == 0:
            return dense_scorer(dataset, episode_index)
        return FrameSignals(
            frame_indices=np.asarray([0, 2], dtype=np.int64),
            signals={PROGRESS_NAME: np.asarray([0.25, 0.75], dtype=np.float32)},
            descriptors={PROGRESS_NAME: PROGRESS_DESCRIPTOR},
        )

    provenance = {"model": {"id": "test/model", "revision": "abc"}, "schema": "test"}
    summary = score_dataset(
        dataset,
        scorer,
        output_path=output_path,
        provenance=provenance,
        resume=True,
    )

    assert summary.output_path == output_path
    assert summary.episode_count == 2
    assert summary.new_episode_count == 2
    assert summary.resumed_episode_count == 0
    assert summary.frame_count == 5

    table = read_frame_signals(output_path)
    assert table.column_names == ["index", "episode_index", "frame_index", PROGRESS_NAME]
    assert table["index"].to_pylist() == [0, 1, 2, 3, 5]
    assert table["episode_index"].to_pylist() == [0, 0, 0, 1, 1]
    assert table["frame_index"].to_pylist() == [0, 1, 2, 0, 2]
    assert get_signal_descriptors(table) == {PROGRESS_NAME: PROGRESS_DESCRIPTOR}
    assert get_scoring_provenance(table) == provenance
    assert pq.ParquetFile(output_path).metadata.num_row_groups == 2
    assert not (tmp_path / ".signals.parquet.parts").exists()


def test_score_dataset_resumes_only_atomic_completed_episode_parts(tmp_path):
    dataset = FakeDataset([2, 2])
    output_path = tmp_path / "signals.parquet"

    def failing_scorer(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        if episode_index == 1:
            raise RuntimeError("model failed")
        return dense_scorer(dataset, episode_index)

    with pytest.raises(RuntimeError, match="model failed"):
        score_dataset(
            dataset,
            failing_scorer,
            output_path=output_path,
            provenance={"model_revision": "abc"},
        )

    parts_dir = tmp_path / ".signals.parquet.parts"
    assert (parts_dir / "episode-000000.parquet").is_file()
    assert not (parts_dir / "episode-000001.parquet").exists()
    assert not output_path.exists()
    assert not list(parts_dir.glob("*.tmp"))

    calls: list[int] = []

    def resumed_scorer(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        calls.append(episode_index)
        return dense_scorer(dataset, episode_index)

    summary = score_dataset(
        dataset,
        resumed_scorer,
        output_path=output_path,
        provenance={"model_revision": "abc"},
    )

    assert calls == [1]
    assert summary.new_episode_count == 1
    assert summary.resumed_episode_count == 1
    assert summary.episode_count == 2
    assert pq.read_table(output_path).num_rows == 4
    assert not parts_dir.exists()


def test_score_dataset_reuses_a_valid_completed_output(tmp_path):
    dataset = FakeDataset([2, 3])
    output_path = tmp_path / "signals.parquet"
    provenance = {"model_revision": "abc"}
    score_dataset(dataset, dense_scorer, output_path=output_path, provenance=provenance)

    calls: list[int] = []

    def scorer(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        calls.append(episode_index)
        return dense_scorer(dataset, episode_index)

    summary = score_dataset(dataset, scorer, output_path=output_path, provenance=provenance)

    assert calls == []
    assert summary.new_episode_count == 0
    assert summary.resumed_episode_count == 2
    assert summary.frame_count == 5


def test_score_dataset_writes_an_empty_sparse_episode(tmp_path):
    dataset = FakeDataset([2])
    output_path = tmp_path / "signals.parquet"

    def scorer(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        return FrameSignals(
            frame_indices=np.asarray([], dtype=np.int64),
            signals={PROGRESS_NAME: np.asarray([], dtype=np.float32)},
            descriptors={PROGRESS_NAME: PROGRESS_DESCRIPTOR},
        )

    summary = score_dataset(
        dataset,
        scorer,
        output_path=output_path,
        provenance={"model_revision": "abc"},
    )

    assert summary.frame_count == 0
    table = read_frame_signals(output_path)
    assert table.num_rows == 0
    assert table.column_names == ["index", "episode_index", "frame_index", PROGRESS_NAME]
    assert not (tmp_path / ".signals.parquet.parts").exists()


@pytest.mark.parametrize(
    "changed_provenance",
    [
        {"model_revision": "changed", "dataset_revision": "dataset-a", "options": {"batch": 8}},
        {"model_revision": "model-a", "dataset_revision": "changed", "options": {"batch": 8}},
        {"model_revision": "model-a", "dataset_revision": "dataset-a", "options": {"batch": 16}},
    ],
)
def test_resume_rejects_changed_provenance(tmp_path, changed_provenance):
    dataset = FakeDataset([2])
    output_path = tmp_path / "signals.parquet"
    provenance = {
        "model_revision": "model-a",
        "dataset_revision": "dataset-a",
        "options": {"batch": 8},
    }
    score_dataset(dataset, dense_scorer, output_path=output_path, provenance=provenance)

    with pytest.raises(ValueError, match="Cannot resume scoring"):
        score_dataset(
            dataset,
            dense_scorer,
            output_path=output_path,
            provenance=changed_provenance,
        )


def test_resume_rejects_changed_episode_selection(tmp_path):
    dataset = FakeDataset([2, 2])
    output_path = tmp_path / "signals.parquet"
    score_dataset(
        dataset,
        dense_scorer,
        output_path=output_path,
        provenance={"model": "test"},
        episode_indices=[0],
    )

    with pytest.raises(ValueError, match="Cannot resume scoring"):
        score_dataset(
            dataset,
            dense_scorer,
            output_path=output_path,
            provenance={"model": "test"},
            episode_indices=[0, 1],
        )


def test_resume_rejects_changed_schema_version(tmp_path):
    dataset = FakeDataset([2])
    output_path = tmp_path / "signals.parquet"
    score_dataset(dataset, dense_scorer, output_path=output_path, provenance={"model": "test"})

    table = pq.read_table(output_path)
    metadata = dict(table.schema.metadata or {})
    metadata[b"lerobot.reward_scoring.schema_version"] = b"999"
    pq.write_table(table.replace_schema_metadata(metadata), output_path)

    with pytest.raises(ValueError, match="Cannot resume scoring"):
        score_dataset(dataset, dense_scorer, output_path=output_path, provenance={"model": "test"})


def test_final_merge_failure_keeps_completed_parts_for_retry(monkeypatch, tmp_path):
    import lerobot.rewards.scoring.writer as scoring_writer

    dataset = FakeDataset([2, 2])
    output_path = tmp_path / "signals.parquet"
    provenance = {"model_revision": "abc"}
    parquet_writer = scoring_writer.pq.ParquetWriter

    def fail_merge(*args, **kwargs):
        raise RuntimeError("merge failed")

    monkeypatch.setattr(scoring_writer.pq, "ParquetWriter", fail_merge)
    with pytest.raises(RuntimeError, match="merge failed"):
        score_dataset(dataset, dense_scorer, output_path=output_path, provenance=provenance)

    parts_dir = tmp_path / ".signals.parquet.parts"
    assert (parts_dir / "episode-000000.parquet").is_file()
    assert (parts_dir / "episode-000001.parquet").is_file()
    assert not output_path.exists()
    assert not list(tmp_path.glob(".signals.parquet.*.tmp"))

    monkeypatch.setattr(scoring_writer.pq, "ParquetWriter", parquet_writer)
    calls: list[int] = []

    def scorer(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        calls.append(episode_index)
        return dense_scorer(dataset, episode_index)

    summary = score_dataset(dataset, scorer, output_path=output_path, provenance=provenance)

    assert calls == []
    assert summary.new_episode_count == 0
    assert summary.resumed_episode_count == 2
    assert summary.frame_count == 4
    assert output_path.is_file()
    assert not parts_dir.exists()


@pytest.mark.parametrize(
    ("frame_indices", "values", "error"),
    [
        (np.asarray([0, 0]), np.asarray([0.1, 0.2]), "strictly increasing"),
        (np.asarray([0, 2]), np.asarray([0.1, 0.2]), "must be in"),
        (np.asarray([0]), np.asarray([np.nan]), "forbidden NaN"),
        (np.asarray([0]), np.asarray([1.5]), "outside its semantic bounds"),
    ],
)
def test_score_dataset_rejects_invalid_frame_signals(tmp_path, frame_indices, values, error):
    dataset = FakeDataset([2])

    def invalid_scorer(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        return FrameSignals(
            frame_indices=frame_indices,
            signals={PROGRESS_NAME: values},
            descriptors={PROGRESS_NAME: PROGRESS_DESCRIPTOR},
        )

    with pytest.raises(ValueError, match=error):
        score_dataset(
            dataset,
            invalid_scorer,
            output_path=tmp_path / "signals.parquet",
            provenance={"model": "test"},
        )


@pytest.mark.parametrize(
    ("descriptor", "error"),
    [
        (SignalDescriptor(description="", direction="higher"), "non-empty description"),
        (
            SignalDescriptor(description="Progress.", direction="sideways"),  # type: ignore[arg-type]
            "Invalid direction",
        ),
        (
            SignalDescriptor(
                description="Progress.",
                direction="higher",
                missing_values="optional",  # type: ignore[arg-type]
            ),
            "Invalid missing_values",
        ),
        (
            SignalDescriptor(description="Progress.", direction="higher", bounds=(1.0, 0.0)),
            "finite and ordered",
        ),
    ],
)
def test_score_dataset_rejects_invalid_signal_descriptors(tmp_path, descriptor, error):
    dataset = FakeDataset([1])

    def scorer(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        return FrameSignals(
            frame_indices=np.asarray([0], dtype=np.int64),
            signals={PROGRESS_NAME: np.asarray([0.5], dtype=np.float32)},
            descriptors={PROGRESS_NAME: descriptor},
        )

    with pytest.raises(ValueError, match=error):
        score_dataset(
            dataset,
            scorer,
            output_path=tmp_path / "signals.parquet",
            provenance={"model": "test"},
        )


def test_score_dataset_preserves_explicitly_allowed_nan_values(tmp_path):
    dataset = FakeDataset([3])
    descriptor = SignalDescriptor(
        description="Sparse handled-frame signal.",
        direction="higher",
        missing_values="nan",
        bounds=(0.0, 1.0),
    )

    def scorer(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        return FrameSignals(
            frame_indices=np.arange(3, dtype=np.int64),
            signals={PROGRESS_NAME: np.asarray([0.2, np.nan, 0.8], dtype=np.float32)},
            descriptors={PROGRESS_NAME: descriptor},
        )

    summary = score_dataset(
        dataset,
        scorer,
        output_path=tmp_path / "signals.parquet",
        provenance={"model": "test"},
    )

    assert summary.frame_count == 3
    table = read_frame_signals(summary.output_path)
    values = table[PROGRESS_NAME].to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(values[[0, 2]], [0.2, 0.8])
    assert np.isnan(values[1])
    assert get_signal_descriptors(table)[PROGRESS_NAME] == descriptor
