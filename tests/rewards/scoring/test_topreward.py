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
from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
from lerobot.rewards.topreward.scoring_topreward import (
    PROGRESS_SIGNAL,
    TOPRewardFrameScorer,
    build_prefix_lengths,
    make_topreward_frame_scorer,
)

pytestmark = pytest.mark.dataset

IMAGE_KEY = "observation.images.top"


@pytest.mark.parametrize(
    ("num_frames", "num_samples", "expected"),
    [(5, None, [1, 2, 3, 4, 5]), (5, 15, [1, 2, 3, 4, 5]), (20, 4, [1, 7, 14, 20])],
)
def test_build_prefix_lengths(num_frames, num_samples, expected):
    assert build_prefix_lengths(num_frames, num_samples).tolist() == expected


def test_make_topreward_frame_scorer_builds_processor_without_frame_cap(monkeypatch):
    import lerobot.rewards.topreward.scoring_topreward as scoring_topreward

    received = {}
    preprocessor = object()

    def fake_make_processors(config):
        received["config"] = config
        return preprocessor, object()

    monkeypatch.setattr(scoring_topreward, "make_topreward_pre_post_processors", fake_make_processors)
    model = SimpleNamespace(config=TOPRewardConfig(device="cpu"))

    scorer = make_topreward_frame_scorer(model, num_samples=3)

    assert scorer.preprocessor is preprocessor
    assert scorer.num_samples == 3
    assert received["config"].max_frames is None
    assert model.config.max_frames == 16


class _FakeDataset:
    absolute_to_relative_idx: dict[int, int] | None = None

    def __init__(self, episodes):
        self.meta = SimpleNamespace(episodes=episodes)

    def __getitem__(self, position: int):
        return {IMAGE_KEY: torch.full((3, 2, 2), float(position)), "task": "pick up the cube"}


@dataclass
class _PrefixLengthEncoderStep(ProcessorStep):
    """Stand in for TOPReward's encoder: record each prefix and write its length."""

    prefix_shapes: list[tuple[int, ...]] = field(default_factory=list)

    def __call__(self, transition):
        observation = dict(transition[TransitionKey.OBSERVATION])
        frames = observation[IMAGE_KEY]
        self.prefix_shapes.append(tuple(frames.shape))
        observation["observation.prefix_length"] = frames.shape[1]
        return {**transition, TransitionKey.OBSERVATION: observation}

    def transform_features(self, features):
        return features


class _FakeEncoder(PolicyProcessorPipeline):
    """A real preprocessor pipeline around the stand-in encoder step."""

    def __init__(self):
        super().__init__(steps=[_PrefixLengthEncoderStep()])

    @property
    def prefix_shapes(self) -> list[tuple[int, ...]]:
        return self.steps[0].prefix_shapes


class _FakeModel:
    config = SimpleNamespace(
        type="topreward",
        pretrained_path=None,
        pretrained_revision=None,
        vlm_name="fake/vlm",
        image_key=IMAGE_KEY,
        task_key="task",
        fps=2.0,
        prompt_prefix="prefix",
        prompt_suffix_template="{instruction}",
        add_chat_template=False,
    )

    def eval(self):
        return self

    def compute_log_probability(self, batch):
        # Longer prefixes get higher log-probabilities: -4, -2 and 0 for lengths 1, 3 and 5.
        return torch.tensor([float(batch["observation.prefix_length"]) - 5.0])


def test_topreward_frame_scorer_scores_whole_prefixes_and_interpolates_progress():
    encoder = _FakeEncoder()
    scorer = TOPRewardFrameScorer(_FakeModel(), encoder, num_samples=3)

    result = scorer.score_episode(_FakeDataset([{"dataset_from_index": 0, "dataset_to_index": 5}]), 0)

    assert encoder.prefix_shapes == [(1, 1, 3, 2, 2), (1, 3, 3, 2, 2), (1, 5, 3, 2, 2)]
    np.testing.assert_array_equal(result.frame_indices, np.arange(5))
    np.testing.assert_allclose(result.signals[PROGRESS_SIGNAL], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert result.signals[PROGRESS_SIGNAL].dtype == np.float32
    assert result.descriptors[PROGRESS_SIGNAL].bounds == (0.0, 1.0)
    assert scorer.name == "topreward"
    assert scorer.provenance["scorer"]["options"]["num_samples"] == 3
    assert scorer.provenance["scorer"]["options"]["vlm_name"] == "fake/vlm"


def test_topreward_scores_publish_to_the_dataset(tmp_path):
    class FakeDataset(LeRobotDataset):
        absolute_to_relative_idx = None
        episodes = None
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

        def __getitem__(self, position: int):
            return {IMAGE_KEY: torch.full((3, 2, 2), float(position)), "task": "pick up the cube"}

    dataset = FakeDataset()
    dataset.add_score(TOPRewardFrameScorer(_FakeModel(), _FakeEncoder(), num_samples=3))

    assert list(dataset.read_score("topreward")[PROGRESS_SIGNAL]) == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_topreward_frame_scorer_supports_episode_filtered_dataset_views():
    dataset = _FakeDataset(
        [{"dataset_from_index": 0, "dataset_to_index": 5}, {"dataset_from_index": 5, "dataset_to_index": 7}]
    )
    dataset.absolute_to_relative_idx = {5: 0, 6: 1}
    encoder = _FakeEncoder()

    result = TOPRewardFrameScorer(_FakeModel(), encoder, num_samples=None).score_episode(dataset, 1)

    assert encoder.prefix_shapes == [(1, 1, 3, 2, 2), (1, 2, 3, 2, 2)]
    np.testing.assert_allclose(result.signals[PROGRESS_SIGNAL], [0.0, 1.0])


def test_topreward_frame_scorer_rejects_non_positive_num_samples():
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        TOPRewardFrameScorer(_FakeModel(), _FakeEncoder(), num_samples=0)
