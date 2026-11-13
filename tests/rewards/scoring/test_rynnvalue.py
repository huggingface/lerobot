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

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import pandas as pd

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.annotations import load_annotations, load_annotations_info
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import PolicyProcessorPipeline, ProcessorStep, batch_to_transition
from lerobot.rewards import load_signal_descriptors, make_frame_scorer, score_dataset
from lerobot.rewards.prediction import RemainingTimeOutput
from lerobot.rewards.rynnvalue.configuration_rynnvalue import RynnValueConfig
from lerobot.rewards.rynnvalue.scoring_rynnvalue import (
    REMAINING_TIME_SIGNAL,
    RynnValueFrameScorer,
    select_anchor_indices,
    select_prefix_indices,
)

pytestmark = pytest.mark.dataset

IMAGE_KEY = "observation.images.top"


@pytest.mark.parametrize(
    ("num_frames", "dataset_fps", "inference_fps", "expected"),
    [
        (61, 30.0, 1.0, [0, 30, 60]),
        (70, 30.0, 1.0, [0, 30, 60, 69]),
        (1, 30.0, 1.0, [0]),
        (4, 2.0, 10.0, [0, 1, 2, 3]),
    ],
)
def test_select_anchor_indices(num_frames, dataset_fps, inference_fps, expected):
    assert select_anchor_indices(num_frames, dataset_fps, inference_fps).tolist() == expected


@pytest.mark.parametrize(
    ("anchor_index", "max_frames", "expected"),
    [(2, 8, [0, 1, 2]), (3, None, [0, 1, 2, 3]), (20, 5, [0, 5, 10, 15, 20]), (7, 1, [7])],
)
def test_select_prefix_indices(anchor_index, max_frames, expected):
    assert select_prefix_indices(anchor_index, max_frames).tolist() == expected


def test_make_rynnvalue_frame_scorer_builds_processor_without_frame_cap(monkeypatch):
    import lerobot.rewards.rynnvalue.scoring_rynnvalue as scoring_rynnvalue

    received = {}
    preprocessor = object()

    def fake_make_processors(config):
        received["config"] = config
        return preprocessor, object()

    monkeypatch.setattr(scoring_rynnvalue, "make_rynnvalue_pre_post_processors", fake_make_processors)
    model = SimpleNamespace(config=RynnValueConfig(device="cpu"))

    scorer = make_frame_scorer(model, inference_fps=0.5)

    assert scorer.preprocessor is preprocessor
    assert scorer.inference_fps == 0.5
    assert received["config"].max_frames is None
    assert model.config.max_frames == 8


class _FakeDataset:
    fps = 2.0

    def __init__(self, episodes, *, view_start: int = 0):
        self.meta = SimpleNamespace(
            episodes=episodes,
            tasks=pd.DataFrame({"task_index": [0, 1]}, index=["another task", "pick up the cube"]),
            features={"instruction": {"dtype": "string"}},
        )
        self.view_start = view_start
        self.absolute_to_relative_idx = (
            {index: index - view_start for index in range(view_start, episodes[-1]["dataset_to_index"])}
            if view_start
            else None
        )
        self.frame_requests: list[tuple[int, list[str], list[int]]] = []
        self.row_requests: list[int] = []

    def get_frames(
        self, episode_index: int, keys: Sequence[str], frame_indices: Sequence[int]
    ) -> dict[str, torch.Tensor]:
        self.frame_requests.append((episode_index, list(keys), list(frame_indices)))
        start = self.meta.episodes[episode_index]["dataset_from_index"] - self.view_start
        return {
            key: torch.stack(
                [
                    torch.tensor(1) if key == "task_index" else torch.full((3, 2, 2), float(start + index))
                    for index in frame_indices
                ]
            )
            for key in keys
        }

    def __getitem__(self, position: int):
        self.row_requests.append(position)
        return {"instruction": f"custom task at frame {position}"}


@dataclass
class _PrefixEncoderStep(ProcessorStep):
    """Stand in for RynnValue's encoder: record each prefix's frames and write its last one."""

    prefixes: list[list[float]] = field(default_factory=list)
    tasks: list[str | None] = field(default_factory=list)
    task_key: str = "task"

    def __call__(self, transition):
        observation = dict(transition[TransitionKey.OBSERVATION])
        self.tasks.append(transition[TransitionKey.COMPLEMENTARY_DATA].get(self.task_key))
        frames = observation[IMAGE_KEY]
        self.prefixes.append(frames[0, :, 0, 0, 0].tolist())
        observation["observation.last_frame"] = frames[:, -1, 0, 0, 0]
        return {**transition, TransitionKey.OBSERVATION: observation}

    def transform_features(self, features):
        return features


class _FakeEncoder(PolicyProcessorPipeline):
    """A real preprocessor pipeline around the stand-in encoder step."""

    def __init__(self, task_key: str = "task"):
        super().__init__(
            steps=[_PrefixEncoderStep(task_key=task_key)],
            to_transition=partial(batch_to_transition, complementary_keys=(task_key,)),
        )

    @property
    def prefixes(self) -> list[list[float]]:
        return self.steps[0].prefixes


class _FakeModel:
    config = SimpleNamespace(
        type="rynnvalue",
        pretrained_path="test/rynnvalue",
        pretrained_revision="model-revision",
        model_id="fake/rynnvalue",
        model_revision=None,
        image_key=IMAGE_KEY,
        task_key="task",
        max_frames=3,
        robot_description="a single-arm robot",
        camera_description=None,
        use_meta=None,
    )

    def eval(self):
        return self

    def predict_remaining_time(self, batch):
        assert torch.is_inference_mode_enabled()
        # Four seconds remain at frame 0, then one second less per frame.
        return RemainingTimeOutput(remaining_time_s=4.0 - batch["observation.last_frame"])


@pytest.mark.parametrize("task_key", ["task", "missing_task", "instruction"])
def test_rynnvalue_frame_scorer_scores_prefixes_and_interpolates_remaining_time(monkeypatch, task_key: str):
    monkeypatch.setattr(_FakeModel.config, "task_key", task_key)
    encoder = _FakeEncoder(task_key=task_key)
    scorer = RynnValueFrameScorer(_FakeModel(), encoder, inference_fps=1.0)
    dataset = _FakeDataset([{"dataset_from_index": 0, "dataset_to_index": 5}])

    result = scorer.score_episode(dataset, 0)

    # At 2 fps, RynnValue runs once a second, on frames 0, 2 and 4, with at most three frames per prefix.
    assert encoder.prefixes == [[0.0], [0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]
    np.testing.assert_array_equal(result.frame_indices, np.arange(5))
    np.testing.assert_allclose(result.signals[REMAINING_TIME_SIGNAL], [4.0, 3.0, 2.0, 1.0, 0.0])
    assert result.signals[REMAINING_TIME_SIGNAL].dtype == np.float32
    assert scorer.descriptors[REMAINING_TIME_SIGNAL].unit == "s"
    assert scorer.descriptors[REMAINING_TIME_SIGNAL].greater_is_better is False
    expected_task_reads = [(0, ["task_index"], [0])] if task_key == "task" else []
    assert dataset.frame_requests == [
        *expected_task_reads,
        (0, [IMAGE_KEY], [0]),
        (0, [IMAGE_KEY], [0, 1, 2]),
        (0, [IMAGE_KEY], [0, 2, 4]),
    ]
    assert dataset.row_requests == ([0] if task_key == "instruction" else [])
    task = {"task": "pick up the cube", "instruction": "custom task at frame 0"}.get(task_key)
    assert encoder.steps[0].tasks == [task] * 3
    assert scorer.name == "rynnvalue"
    options = scorer.provenance["scorer"]["options"]
    assert options["inference_fps"] == 1.0
    assert options["model_id"] == "fake/rynnvalue"
    assert options["robot_description"] == "a single-arm robot"


@pytest.mark.parametrize("task_key", ["task", "instruction"])
def test_rynnvalue_frame_scorer_supports_episode_filtered_dataset_views(monkeypatch, task_key: str):
    monkeypatch.setattr(_FakeModel.config, "task_key", task_key)
    dataset = _FakeDataset(
        [{"dataset_from_index": 0, "dataset_to_index": 5}, {"dataset_from_index": 5, "dataset_to_index": 8}],
        view_start=5,
    )
    encoder = _FakeEncoder(task_key=task_key)

    result = RynnValueFrameScorer(_FakeModel(), encoder, inference_fps=1.0).score_episode(dataset, 1)

    assert encoder.prefixes == [[0.0], [0.0, 1.0, 2.0]]
    np.testing.assert_allclose(result.signals[REMAINING_TIME_SIGNAL], [4.0, 3.0, 2.0])
    expected_task_reads = [(1, ["task_index"], [0])] if task_key == "task" else []
    assert dataset.frame_requests == [
        *expected_task_reads,
        (1, [IMAGE_KEY], [0]),
        (1, [IMAGE_KEY], [0, 1, 2]),
    ]
    assert dataset.row_requests == ([0] if task_key == "instruction" else [])


def test_rynnvalue_frame_scorer_rejects_non_positive_inference_fps():
    with pytest.raises(ValueError, match="inference_fps must be > 0"):
        RynnValueFrameScorer(_FakeModel(), _FakeEncoder(), inference_fps=0)


def test_rynnvalue_scores_publish_as_annotations(tmp_path):
    class FakeDataset(LeRobotDataset):
        absolute_to_relative_idx = None
        episodes = None
        fps = 2.0
        reader = SimpleNamespace()
        writer = None
        _storage_root = None

        def __init__(self) -> None:
            self.root = tmp_path
            self.repo_id = "test/dataset"
            self.revision = "dataset-revision"
            self.meta = SimpleNamespace(
                total_episodes=1,
                total_frames=5,
                storage_format="lerobot",
                episodes=[{"dataset_from_index": 0, "dataset_to_index": 5}],
                tasks=pd.DataFrame({"task_index": [0, 1]}, index=["another task", "pick up the cube"]),
            )

        def get_frames(
            self, episode_index: int, keys: Sequence[str], frame_indices: Sequence[int]
        ) -> dict[str, torch.Tensor]:
            return {
                key: torch.stack(
                    [
                        torch.tensor(1) if key == "task_index" else torch.full((3, 2, 2), float(index))
                        for index in frame_indices
                    ]
                )
                for key in keys
            }

    dataset = FakeDataset()
    score_dataset(dataset, RynnValueFrameScorer(_FakeModel(), _FakeEncoder(), inference_fps=1.0))

    scores = load_annotations(dataset, "rynnvalue")
    assert list(scores[REMAINING_TIME_SIGNAL]) == [4.0, 3.0, 2.0, 1.0, 0.0]
    assert load_signal_descriptors(dataset, "rynnvalue")[REMAINING_TIME_SIGNAL].greater_is_better is False
    assert load_annotations_info(dataset, "rynnvalue")["provenance"]["model"]["id"] == "test/rynnvalue"
