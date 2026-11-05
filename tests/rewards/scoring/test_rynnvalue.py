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

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import LeRobotDataset
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import PolicyProcessorPipeline, ProcessorStep
from lerobot.rewards.prediction import RemainingTimePrediction
from lerobot.rewards.rynnvalue.configuration_rynnvalue import RynnValueConfig
from lerobot.rewards.rynnvalue.scoring_rynnvalue import (
    REMAINING_TIME_SIGNAL,
    RynnValueFrameScorer,
    make_rynnvalue_frame_scorer,
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

    scorer = make_rynnvalue_frame_scorer(model, inference_fps=0.5)

    assert scorer.preprocessor is preprocessor
    assert scorer.inference_fps == 0.5
    assert received["config"].max_frames is None
    assert model.config.max_frames == 8


class _FakeDataset:
    absolute_to_relative_idx: dict[int, int] | None = None
    fps = 2.0

    def __init__(self, episodes):
        self.meta = SimpleNamespace(episodes=episodes)

    def __getitem__(self, position: int):
        return {IMAGE_KEY: torch.full((3, 2, 2), float(position)), "task": "pick up the cube"}


@dataclass
class _PrefixEncoderStep(ProcessorStep):
    """Stand in for RynnValue's encoder: record each prefix's frames and write its last one."""

    prefixes: list[list[float]] = field(default_factory=list)

    def __call__(self, transition):
        observation = dict(transition[TransitionKey.OBSERVATION])
        frames = observation[IMAGE_KEY]
        self.prefixes.append(frames[0, :, 0, 0, 0].tolist())
        observation["observation.last_frame"] = frames[:, -1, 0, 0, 0]
        return {**transition, TransitionKey.OBSERVATION: observation}

    def transform_features(self, features):
        return features


class _FakeEncoder(PolicyProcessorPipeline):
    """A real preprocessor pipeline around the stand-in encoder step."""

    def __init__(self):
        super().__init__(steps=[_PrefixEncoderStep()])

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
        return RemainingTimePrediction(remaining_time_s=4.0 - batch["observation.last_frame"])


def test_rynnvalue_frame_scorer_scores_prefixes_and_interpolates_remaining_time():
    encoder = _FakeEncoder()
    scorer = RynnValueFrameScorer(_FakeModel(), encoder, inference_fps=1.0)

    result = scorer.score_episode(_FakeDataset([{"dataset_from_index": 0, "dataset_to_index": 5}]), 0)

    # At 2 fps, RynnValue runs once a second, on frames 0, 2 and 4, with at most three frames per prefix.
    assert encoder.prefixes == [[0.0], [0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]
    np.testing.assert_array_equal(result.frame_indices, np.arange(5))
    np.testing.assert_allclose(result.signals[REMAINING_TIME_SIGNAL], [4.0, 3.0, 2.0, 1.0, 0.0])
    assert result.signals[REMAINING_TIME_SIGNAL].dtype == np.float32
    assert result.descriptors[REMAINING_TIME_SIGNAL].unit == "s"
    assert result.descriptors[REMAINING_TIME_SIGNAL].direction == "lower"
    assert scorer.name == "rynnvalue"
    options = scorer.provenance["scorer"]["options"]
    assert options["inference_fps"] == 1.0
    assert options["model_id"] == "fake/rynnvalue"
    assert options["robot_description"] == "a single-arm robot"


def test_rynnvalue_frame_scorer_supports_episode_filtered_dataset_views():
    dataset = _FakeDataset(
        [{"dataset_from_index": 0, "dataset_to_index": 5}, {"dataset_from_index": 5, "dataset_to_index": 8}]
    )
    dataset.absolute_to_relative_idx = {5: 0, 6: 1, 7: 2}
    encoder = _FakeEncoder()

    result = RynnValueFrameScorer(_FakeModel(), encoder, inference_fps=1.0).score_episode(dataset, 1)

    assert encoder.prefixes == [[0.0], [0.0, 1.0, 2.0]]
    np.testing.assert_allclose(result.signals[REMAINING_TIME_SIGNAL], [4.0, 3.0, 2.0])


def test_rynnvalue_frame_scorer_rejects_non_positive_inference_fps():
    with pytest.raises(ValueError, match="inference_fps must be > 0"):
        RynnValueFrameScorer(_FakeModel(), _FakeEncoder(), inference_fps=0)


def test_rynnvalue_scores_publish_through_dataset_score_api(tmp_path):
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
            )

        def __getitem__(self, index: int):
            return {IMAGE_KEY: torch.full((3, 2, 2), float(index)), "task": "pick up the cube"}

    dataset = FakeDataset()
    dataset.add_score(RynnValueFrameScorer(_FakeModel(), _FakeEncoder(), inference_fps=1.0))

    scores = dataset.read_score("rynnvalue")
    assert list(scores[REMAINING_TIME_SIGNAL]) == [4.0, 3.0, 2.0, 1.0, 0.0]
    assert dataset.get_score_descriptors("rynnvalue")[REMAINING_TIME_SIGNAL].direction == "lower"
    assert dataset.get_score_provenance("rynnvalue")["scoring"]["model"]["id"] == "test/rynnvalue"
