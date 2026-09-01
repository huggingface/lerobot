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
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from lerobot.rewards.scoring import (
    FrameSignals,
    SignalDescriptor,
    get_scoring_provenance,
    get_signal_descriptors,
    read_frame_signals,
    score_dataset,
    score_dataset_with_reward_model,
)

PROGRESS_NAME = "reward.test.progress"
PROGRESS_DESCRIPTOR = SignalDescriptor(
    description="Normalized task progress.",
    unit=None,
    direction="higher",
    comparison_scope="task",
    missing_values="forbidden",
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

    assert summary.artifact_path == output_path
    assert summary.episode_count == 2
    assert summary.new_episode_count == 2
    assert summary.resumed_episode_count == 0
    assert summary.frame_count == 5
    assert summary.signal_nan_counts == {PROGRESS_NAME: 0}
    assert summary.observed_ranges == {PROGRESS_NAME: (0.0, 1.0)}

    table = read_frame_signals(output_path)
    assert table.column_names == ["index", "episode_index", "frame_index", PROGRESS_NAME]
    assert table["index"].to_pylist() == [0, 1, 2, 3, 5]
    assert table["episode_index"].to_pylist() == [0, 0, 0, 1, 1]
    assert table["frame_index"].to_pylist() == [0, 1, 2, 0, 2]
    assert get_signal_descriptors(table) == {PROGRESS_NAME: PROGRESS_DESCRIPTOR}
    assert get_scoring_provenance(table) == provenance


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

    manifest_path = tmp_path / ".signals.parquet.parts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Cannot resume scoring"):
        score_dataset(dataset, dense_scorer, output_path=output_path, provenance={"model": "test"})


@pytest.mark.parametrize(
    ("frame_indices", "values", "error"),
    [
        (np.asarray([0, 0]), np.asarray([0.1, 0.2]), "strictly increasing"),
        (np.asarray([0, 2]), np.asarray([0.1, 0.2]), "must be in"),
        (np.asarray([0]), np.asarray([np.nan]), "forbids missing"),
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


def test_score_dataset_reports_allowed_nan_without_inventing_values(tmp_path):
    dataset = FakeDataset([3])
    descriptor = SignalDescriptor(
        description="Sparse handled-frame signal.",
        unit=None,
        direction="higher",
        comparison_scope="episode",
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

    assert summary.signal_nan_counts == {PROGRESS_NAME: 1}
    assert summary.observed_ranges[PROGRESS_NAME] == pytest.approx((0.2, 0.8))


def test_score_dataset_with_reward_model_builds_reproducible_robometer_provenance(monkeypatch, tmp_path):
    import lerobot.rewards.robometer.modeling_robometer as modeling_robometer
    import lerobot.rewards.robometer.scoring_robometer as scoring_robometer
    import lerobot.rewards.scoring.runner as runner

    class FakeRobometerRewardModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                type="robometer",
                pretrained_path="model/default",
                pretrained_revision="config-revision",
            )

    fake_scorer = SimpleNamespace(options={"batch_size": 4, "sampling": "test"})
    captured: dict[str, object] = {}
    expected_summary = SimpleNamespace(artifact_path=tmp_path / "signals.parquet")

    def fake_make_scorer(model, config, **kwargs):
        captured["make_scorer"] = (model, config, kwargs)
        return fake_scorer

    def fake_score_dataset(dataset, scorer, **kwargs):
        captured["score_dataset"] = (dataset, scorer, kwargs)
        return expected_summary

    monkeypatch.setattr(modeling_robometer, "RobometerRewardModel", FakeRobometerRewardModel)
    monkeypatch.setattr(scoring_robometer, "make_robometer_frame_scorer", fake_make_scorer)
    monkeypatch.setattr(runner, "score_dataset", fake_score_dataset)

    dataset = SimpleNamespace(repo_id="user/dataset", revision="dataset-commit")
    model = FakeRobometerRewardModel()
    output_path = tmp_path / "signals.parquet"
    summary = score_dataset_with_reward_model(
        dataset,
        model,
        output_path=output_path,
        model_id="user/robometer",
        model_revision="model-commit",
        episode_indices=[2, 1],
        batch_size=4,
        num_subsampled_frames=8,
    )

    assert summary is expected_summary
    _, _, score_kwargs = captured["score_dataset"]
    assert score_kwargs["output_path"] == output_path
    assert score_kwargs["episode_indices"] == [2, 1]
    assert score_kwargs["provenance"] == {
        "schema_version": 1,
        "lerobot_version": __import__("lerobot").__version__,
        "dataset": {"repo_id": "user/dataset", "revision": "dataset-commit"},
        "model": {"type": "robometer", "id": "user/robometer", "revision": "model-commit"},
        "adapter": {
            "id": "lerobot.robometer.frame_prefix",
            "version": 1,
            "options": fake_scorer.options,
        },
    }
