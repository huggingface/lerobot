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

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

import datasets
import numpy as np
import pytest

from lerobot.datasets import FrameSignals, LeRobotDataset, SignalDescriptor

pq = pytest.importorskip("pyarrow.parquet")

PROGRESS_NAME = "reward.test.progress"
PROGRESS_DESCRIPTOR = SignalDescriptor(
    description="Normalized task progress.",
    direction="higher",
    bounds=(0.0, 1.0),
)


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


class FakeScorer:
    name = "test"

    def __init__(
        self,
        score: Callable[[FakeDataset, int], FrameSignals],
        *,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self._score = score
        self._provenance = dict(provenance or {"model": {"id": "test/model"}})
        self.calls: list[int] = []

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    def score_episode(self, dataset: FakeDataset, episode_index: int) -> FrameSignals:
        self.calls.append(episode_index)
        return self._score(dataset, episode_index)


def dense_signals(dataset: FakeDataset, episode_index: int) -> FrameSignals:
    episode = dataset.meta.episodes[episode_index]
    length = episode["dataset_to_index"] - episode["dataset_from_index"]
    return FrameSignals(
        frame_indices=np.arange(length, dtype=np.int64),
        signals={PROGRESS_NAME: np.linspace(0.0, 1.0, length, dtype=np.float32)},
        descriptors={PROGRESS_NAME: PROGRESS_DESCRIPTOR},
    )


def test_dataset_score_api_writes_reads_and_lists_named_sidecars(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [3, 4])

    def score(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        if episode_index == 0:
            return dense_signals(dataset, episode_index)
        return FrameSignals(
            frame_indices=np.asarray([0, 2], dtype=np.int64),
            signals={PROGRESS_NAME: np.asarray([0.25, 0.75], dtype=np.float32)},
            descriptors={PROGRESS_NAME: PROGRESS_DESCRIPTOR},
        )

    scorer = FakeScorer(score, provenance={"dataset": "cannot-clobber", "model": {"id": "test/model"}})
    dataset.add_score(scorer, name="experiment")

    scores = dataset.read_score("experiment")
    assert isinstance(scores, datasets.Dataset)
    assert list(scores["index"]) == [0, 1, 2, 3, 5]
    assert list(scores["episode_index"]) == [0, 0, 0, 1, 1]
    assert list(scores["frame_index"]) == [0, 1, 2, 0, 2]
    assert scores[0][PROGRESS_NAME] == 0.0
    assert dataset.get_score_descriptors("experiment") == {PROGRESS_NAME: PROGRESS_DESCRIPTOR}
    provenance = dataset.get_score_provenance("experiment")
    assert provenance["dataset"]["repo_id"] == "test/dataset"
    assert provenance["dataset"]["total_frames"] == 7
    assert provenance["scoring"]["dataset"] == "cannot-clobber"
    output_path = dataset.root / "reward_signals" / "experiment.parquet"
    assert pq.ParquetFile(output_path).metadata.num_row_groups == 2
    assert not output_path.with_name(".experiment.parquet.parts").exists()

    dataset.add_score(FakeScorer(score), name="other")
    assert dataset.list_scores() == ["experiment", "other"]


def test_dataset_score_upload_uses_owned_hub_location(monkeypatch, tmp_path):
    import lerobot.datasets.scores.storage as score_storage

    dataset = FakeDataset(tmp_path / "dataset", [1])
    dataset.add_score(FakeScorer(dense_signals))
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        score_storage,
        "HfApi",
        lambda: SimpleNamespace(upload_file=lambda **kwargs: captured.update(kwargs)),
    )

    dataset.push_score_to_hub("test")

    assert captured["path_or_fileobj"] == dataset.root / "reward_signals" / "test.parquet"
    assert captured["path_in_repo"] == "reward_signals/test.parquet"
    assert captured["repo_id"] == dataset.repo_id
    assert captured["repo_type"] == "dataset"


def test_dataset_score_resume_skips_atomic_completed_parts(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])

    def fail_second(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        if episode_index == 1:
            raise RuntimeError("model failed")
        return dense_signals(dataset, episode_index)

    with pytest.raises(RuntimeError, match="model failed"):
        dataset.add_score(FakeScorer(fail_second))

    parts_dir = dataset.root / "reward_signals" / ".test.parquet.parts"
    assert (parts_dir / "episode-000000.parquet").is_file()
    assert not (parts_dir / "episode-000001.parquet").exists()

    scorer = FakeScorer(dense_signals)
    dataset.add_score(scorer)

    assert scorer.calls == [1]
    assert len(dataset.read_score("test")) == 4
    assert not parts_dir.exists()


def test_dataset_score_rejects_published_extension_without_overwrite(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])
    dataset.add_score(FakeScorer(dense_signals), episodes=[0])

    scorer = FakeScorer(dense_signals)
    with pytest.raises(FileExistsError, match="Published dataset score"):
        dataset.add_score(scorer, episodes=[0, 1])

    assert scorer.calls == []


def test_dataset_score_overwrite_keeps_published_output_until_success(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2])
    dataset.add_score(FakeScorer(dense_signals))
    original = dataset.read_score("test")

    def fail(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        raise RuntimeError("replacement failed")

    with pytest.raises(RuntimeError, match="replacement failed"):
        dataset.add_score(FakeScorer(fail), overwrite=True)

    current = dataset.read_score("test")
    assert current[:] == original[:]


def test_dataset_score_resume_rejects_changed_selection(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2])

    def fail_second(dataset: FakeDataset, episode_index: int) -> FrameSignals:
        if episode_index == 1:
            raise RuntimeError("model failed")
        return dense_signals(dataset, episode_index)

    with pytest.raises(RuntimeError, match="model failed"):
        dataset.add_score(FakeScorer(fail_second), episodes=[0, 1])

    with pytest.raises(ValueError, match="selection or provenance changed"):
        dataset.add_score(FakeScorer(dense_signals), episodes=[0])


def test_dataset_score_explicit_episodes_must_belong_to_view(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [2, 2, 2], episodes=[0, 2])
    scorer = FakeScorer(dense_signals)

    with pytest.raises(ValueError, match="visible in the current dataset view"):
        dataset.add_score(scorer, episodes=[1])

    assert scorer.calls == []


@pytest.mark.parametrize("name", ["", ".", "..", "nested/name", r"nested\name"])
def test_dataset_score_rejects_invalid_names(tmp_path, name):
    dataset = FakeDataset(tmp_path / "dataset", [1])

    with pytest.raises(ValueError, match="Score name"):
        dataset.add_score(FakeScorer(dense_signals), name=name)


def test_dataset_score_rejects_unsupported_storage_before_inference(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [1], storage_format="lance")
    scorer = FakeScorer(dense_signals)

    with pytest.raises(NotImplementedError, match="storage_format"):
        dataset.add_score(scorer)

    assert scorer.calls == []


def test_dataset_score_rejects_snapshot_cache_before_inference(monkeypatch, tmp_path):
    import lerobot.datasets.scores.storage as score_storage

    cache_root = tmp_path / "hub"
    dataset = FakeDataset(cache_root / "snapshots" / "revision", [1])
    scorer = FakeScorer(dense_signals)
    monkeypatch.setattr(score_storage, "HF_LEROBOT_HUB_CACHE", cache_root)

    with pytest.raises(NotImplementedError, match="snapshot cache"):
        dataset.add_score(scorer)

    assert scorer.calls == []


@pytest.mark.parametrize(
    "result,error",
    [
        (
            FrameSignals(
                frame_indices=np.asarray([0], dtype=np.int64),
                signals={"index": np.asarray([0.5], dtype=np.float32)},
                descriptors={"index": PROGRESS_DESCRIPTOR},
            ),
            "reserved columns",
        ),
        (
            FrameSignals(
                frame_indices=np.asarray([0], dtype=np.int64),
                signals={PROGRESS_NAME: np.asarray([0.5, 0.6], dtype=np.float32)},
                descriptors={PROGRESS_NAME: PROGRESS_DESCRIPTOR},
            ),
            "must have shape",
        ),
        (
            FrameSignals(
                frame_indices=np.asarray([0], dtype=np.int64),
                signals={PROGRESS_NAME: np.asarray([0.5], dtype=np.float32)},
                descriptors={},
            ),
            "identical names",
        ),
    ],
)
def test_dataset_score_rejects_structurally_invalid_signals(tmp_path, result, error):
    dataset = FakeDataset(tmp_path / "dataset", [1])
    scorer = FakeScorer(lambda dataset, episode_index: result)

    with pytest.raises(ValueError, match=error):
        dataset.add_score(scorer)


def test_dataset_score_metadata_provenance_is_nested(tmp_path):
    dataset = FakeDataset(tmp_path / "dataset", [1])
    dataset.add_score(FakeScorer(dense_signals))

    provenance = dataset.get_score_provenance("test")
    assert set(provenance) == {"lerobot_version", "dataset", "scoring"}
