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

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.annotations import (
    add_annotations,
    list_annotations,
    load_annotations,
    load_annotations_info,
)
from lerobot.rewards import FrameSignals, SignalDescriptor, load_signal_descriptors, score_dataset

pytestmark = pytest.mark.dataset


class FakeDataset(LeRobotDataset):
    _storage_root = None
    reader = SimpleNamespace()
    writer = None

    def __init__(self, root: Path, lengths: tuple[int, ...] = (3, 2)) -> None:
        root.mkdir(parents=True, exist_ok=True)
        episodes = []
        start = 0
        for episode_index, length in enumerate(lengths):
            episodes.append(
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
        self.episodes = None
        self.meta = SimpleNamespace(
            episodes=episodes,
            total_episodes=len(episodes),
            total_frames=start,
            storage_format="lerobot",
        )


class FakeScorer:
    name = "fake"

    def __init__(
        self,
        *,
        descriptors: Mapping[str, SignalDescriptor] | None = None,
        result: FrameSignals | None = None,
        fail_on_episode: int | None = None,
    ) -> None:
        self._descriptors = (
            {
                "progress": SignalDescriptor(
                    description="Task progress", greater_is_better=True, bounds=(0.0, 1.0), unit="fraction"
                )
            }
            if descriptors is None
            else dict(descriptors)
        )
        self.result = result
        self.fail_on_episode = fail_on_episode
        self.calls: list[int] = []

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {"model_id": "test/model", "checkpoint": "model-revision", "options": {"stride": 1}}

    @property
    def descriptors(self) -> Mapping[str, SignalDescriptor]:
        return self._descriptors

    def score_episode(self, dataset: LeRobotDataset, episode_index: int) -> FrameSignals:
        self.calls.append(episode_index)
        if episode_index == self.fail_on_episode:
            raise RuntimeError("Scoring interrupted")
        if self.result is not None:
            return self.result
        episode = dataset.meta.episodes[episode_index]
        length = episode["dataset_to_index"] - episode["dataset_from_index"]
        return FrameSignals(
            frame_indices=np.arange(length, dtype=np.int64),
            signals={"progress": np.linspace(0.0, 1.0, length, dtype=np.float32)},
        )


def test_score_dataset_writes_signals_descriptors_and_provenance(tmp_path: Path) -> None:
    dataset = FakeDataset(tmp_path / "dataset")
    scorer = FakeScorer()

    score_dataset(dataset, scorer)

    table = load_annotations(dataset, scorer.name)
    assert list(table["index"]) == [0, 1, 2, 3, 4]
    assert list(table["episode_index"]) == [0, 0, 0, 1, 1]
    assert list(table["frame_index"]) == [0, 1, 2, 0, 1]
    np.testing.assert_allclose(table["progress"], [0.0, 0.5, 1.0, 0.0, 1.0])
    descriptors = load_signal_descriptors(dataset, scorer.name)
    assert descriptors == dict(scorer.descriptors)
    assert isinstance(descriptors["progress"].bounds, tuple)
    assert load_annotations_info(dataset, scorer.name)["provenance"] == dict(scorer.provenance)
    assert scorer.calls == [0, 1]


@pytest.mark.parametrize("lengths", [(2, 3), (3, 1), (5,)])
def test_rabc_rejects_annotations_after_episode_ranges_change(
    tmp_path: Path, lengths: tuple[int, ...]
) -> None:
    from lerobot.rewards.sarm.rabc import RABCWeights

    dataset = FakeDataset(tmp_path / "dataset")
    score_dataset(dataset, FakeScorer())
    changed_dataset = FakeDataset(dataset.root, lengths=lengths)

    with pytest.raises(ValueError, match="no longer align"):
        RABCWeights(changed_dataset, "fake", "progress", chunk_size=1)


def test_rabc_rejects_sparse_annotations_after_episode_start_moves(tmp_path: Path) -> None:
    from lerobot.rewards.sarm.rabc import RABCWeights

    dataset = FakeDataset(tmp_path / "dataset")
    scorer = FakeScorer(
        result=FrameSignals(
            frame_indices=np.asarray([1], dtype=np.int64),
            signals={"progress": np.asarray([0.5], dtype=np.float32)},
        )
    )
    score_dataset(dataset, scorer)
    original = RABCWeights(dataset, "fake", "progress", chunk_size=1)
    assert sorted(original.progress_lookup) == [1, 4]

    # Both global indices still fit their episodes, but episode 1 now starts at 2 instead of 3.
    changed_dataset = FakeDataset(dataset.root, lengths=(2, 3))

    with pytest.raises(ValueError, match="no longer align"):
        RABCWeights(changed_dataset, "fake", "progress", chunk_size=1)


def test_rabc_accepts_annotations_after_episodes_are_appended(tmp_path: Path) -> None:
    from lerobot.rewards.sarm.rabc import RABCWeights

    dataset = FakeDataset(tmp_path / "dataset")
    score_dataset(dataset, FakeScorer())
    original = RABCWeights(dataset, "fake", "progress", chunk_size=1)
    appended_dataset = FakeDataset(dataset.root, lengths=(3, 2, 4))

    weighter = RABCWeights(appended_dataset, "fake", "progress", chunk_size=1)

    assert weighter.progress_lookup == original.progress_lookup
    assert weighter.episode_boundaries == original.episode_boundaries
    assert weighter.delta_mean == original.delta_mean
    assert weighter.delta_std == original.delta_std


def test_factory_rabc_reads_local_annotations_for_a_streaming_dataset(tmp_path: Path) -> None:
    import torch

    from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
    from lerobot.rewards.sarm.rabc import RABCWeights
    from lerobot.utils.sample_weighting import SampleWeightingConfig, make_sample_weighter

    dataset = FakeDataset(tmp_path / "dataset")
    score_dataset(dataset, FakeScorer())
    streaming = StreamingLeRobotDataset.__new__(StreamingLeRobotDataset)
    streaming.root = dataset.root
    streaming.meta = dataset.meta

    weighter = make_sample_weighter(
        SampleWeightingConfig(annotation_name="fake", signal_name="progress"),
        SimpleNamespace(config=SimpleNamespace(chunk_size=1)),
        torch.device("cpu"),
        dataset=streaming,
    )

    assert isinstance(weighter, RABCWeights)
    assert sorted(weighter.progress_lookup) == [0, 1, 2, 3, 4]


def test_score_dataset_default_and_explicit_names_are_separate(tmp_path: Path) -> None:
    dataset = FakeDataset(tmp_path / "dataset")
    scorer = FakeScorer()

    score_dataset(dataset, scorer)
    score_dataset(dataset, scorer, name="another-run")

    assert list_annotations(dataset) == ["another-run", scorer.name]
    assert load_annotations(dataset, "another-run")[:] == load_annotations(dataset, scorer.name)[:]


@pytest.mark.parametrize("greater_is_better", [True, False, None])
def test_score_dataset_round_trips_descriptor_directions(
    tmp_path: Path, greater_is_better: bool | None
) -> None:
    dataset = FakeDataset(tmp_path / "dataset")
    descriptor = SignalDescriptor(description="A signal", greater_is_better=greater_is_better)
    scorer = FakeScorer(descriptors={"progress": descriptor})

    score_dataset(dataset, scorer)

    assert load_signal_descriptors(dataset, scorer.name) == {"progress": descriptor}


@pytest.mark.parametrize("invalid_direction", [0, 1, "higher", np.bool_(True)])
def test_score_dataset_rejects_invalid_directions_before_scoring(
    tmp_path: Path, invalid_direction: Any
) -> None:
    dataset = FakeDataset(tmp_path / "dataset")
    descriptor = SignalDescriptor(description="A signal", greater_is_better=invalid_direction)
    scorer = FakeScorer(descriptors={"progress": descriptor})

    with pytest.raises(ValueError, match="greater_is_better"):
        score_dataset(dataset, scorer)

    assert scorer.calls == []
    assert not (dataset.root / "annotations").exists()


def test_score_dataset_rejects_signal_descriptor_name_mismatch(tmp_path: Path) -> None:
    dataset = FakeDataset(tmp_path / "dataset", lengths=(1,))
    scorer = FakeScorer(result=FrameSignals(np.asarray([0]), {"another-signal": np.asarray([0.5])}))

    with pytest.raises(ValueError, match="must match.*descriptors"):
        score_dataset(dataset, scorer)


def test_score_dataset_rejects_repeated_frame_indices(tmp_path: Path) -> None:
    dataset = FakeDataset(tmp_path / "dataset", lengths=(2,))
    scorer = FakeScorer(result=FrameSignals(np.asarray([0, 0]), {"progress": np.asarray([0.0, 0.5])}))

    with pytest.raises(ValueError, match="repeated"):
        score_dataset(dataset, scorer)


@pytest.mark.parametrize("values", [np.asarray(["text"]), np.asarray([1.0 + 2.0j])])
def test_score_dataset_rejects_non_numeric_signals(tmp_path: Path, values: np.ndarray) -> None:
    dataset = FakeDataset(tmp_path / "dataset", lengths=(1,))
    scorer = FakeScorer(result=FrameSignals(np.asarray([0]), {"progress": values}))

    with pytest.raises(ValueError, match="numeric dtype"):
        score_dataset(dataset, scorer)


@pytest.mark.parametrize(
    "values",
    [np.asarray([True]), np.asarray([1], dtype=np.int64), np.asarray([1], dtype=np.uint8)],
)
def test_score_dataset_accepts_boolean_and_integer_signals(tmp_path: Path, values: np.ndarray) -> None:
    dataset = FakeDataset(tmp_path / "dataset", lengths=(1,))
    scorer = FakeScorer(result=FrameSignals(np.asarray([0]), {"progress": values}))

    score_dataset(dataset, scorer)

    assert list(load_annotations(dataset, scorer.name)["progress"]) == values.tolist()


def test_score_dataset_resumes_only_missing_episodes(tmp_path: Path) -> None:
    dataset = FakeDataset(tmp_path / "dataset", lengths=(2, 3, 1))
    scorer = FakeScorer(fail_on_episode=1)

    with pytest.raises(RuntimeError, match="Scoring interrupted"):
        score_dataset(dataset, scorer)
    assert scorer.calls == [0, 1]

    scorer.calls.clear()
    scorer.fail_on_episode = None
    score_dataset(dataset, scorer)

    assert scorer.calls == [1, 2]
    assert len(load_annotations(dataset, scorer.name)) == 6


def test_score_dataset_forwards_episode_selection_and_overwrite(tmp_path: Path) -> None:
    dataset = FakeDataset(tmp_path / "dataset", lengths=(2, 3, 1))
    scorer = FakeScorer()

    score_dataset(dataset, scorer, episodes=[2, 0])
    assert scorer.calls == [0, 2]
    assert list(load_annotations(dataset, scorer.name)["episode_index"]) == [0, 0, 2]

    scorer.calls.clear()
    score_dataset(dataset, scorer, episodes=[1], overwrite=True)
    assert scorer.calls == [1]
    assert list(load_annotations(dataset, scorer.name)["episode_index"]) == [1, 1, 1]


def test_load_signal_descriptors_rejects_tables_without_signals_metadata(tmp_path: Path) -> None:
    dataset = FakeDataset(tmp_path / "dataset", lengths=(1,))
    add_annotations(
        dataset,
        "other-results",
        lambda episode_index: (np.asarray([0]), {"value": np.asarray([0.5])}),
        info={"source": "another-process"},
    )

    with pytest.raises(ValueError, match="no signal descriptors.*score_dataset"):
        load_signal_descriptors(dataset, "other-results")
