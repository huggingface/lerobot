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
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.scores import load_scores
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import PolicyProcessorPipeline, ProcessorStep, batch_to_transition
from lerobot.rewards import score_dataset
from lerobot.rewards.robometer.modeling_robometer import RobometerOutput
from lerobot.rewards.robometer.scoring_robometer import (
    PROGRESS_SIGNAL,
    SUCCESS_PROBABILITY_SIGNAL,
    RobometerFrameScorer,
    build_subsample_indices,
    make_robometer_frame_scorer,
)

pytestmark = pytest.mark.dataset


@dataclass
class _FrameIdsEncoderStep(ProcessorStep):
    """Stand in for RoboMeter's encoder: write each frame's first pixel as `observation.encoded`."""

    image_key: str = "observation.images.top"
    task_key: str = "task"
    tasks: list[str | None] = field(default_factory=list)

    def __call__(self, transition):
        self.tasks.append(transition[TransitionKey.COMPLEMENTARY_DATA].get(self.task_key))
        observation = dict(transition[TransitionKey.OBSERVATION])
        observation["observation.encoded"] = observation[self.image_key][:, :, 0, 0, 0]
        return {**transition, TransitionKey.OBSERVATION: observation}

    def transform_features(self, features):
        return features


def _make_preprocessor(task_key: str = "task") -> PolicyProcessorPipeline:
    return PolicyProcessorPipeline(
        steps=[_FrameIdsEncoderStep(task_key=task_key)],
        to_transition=partial(batch_to_transition, complementary_keys=(task_key,)),
    )


def test_build_subsample_indices_matches_upstream_frame_steps():
    actual = build_subsample_indices(6, 4)

    assert [indices.tolist() for indices in actual] == [
        np.linspace(0, frame_index, 4, dtype=int).tolist() for frame_index in range(6)
    ]
    assert [indices.tolist() for indices in actual] == [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2, 4],
        [0, 1, 3, 5],
    ]


def test_build_subsample_indices_keeps_prefix_endpoints():
    for current_frame, indices in enumerate(build_subsample_indices(7, 4)):
        assert indices.shape == (4,)
        assert indices.dtype == np.int64
        assert indices[0] == 0
        assert indices[-1] == current_frame


def test_make_robometer_frame_scorer_builds_processor_without_frame_cap(monkeypatch):
    import lerobot.rewards.robometer.scoring_robometer as scoring_robometer
    from lerobot.rewards.robometer import RobometerConfig

    config = RobometerConfig(
        device="cpu", vlm_config={"model_type": "fake", "text_config": {"vocab_size": 10}}
    )
    model = SimpleNamespace(config=config)
    preprocessor = object()
    received = {}

    def fake_make_processors(received_config):
        received["config"] = received_config
        return preprocessor, object()

    monkeypatch.setattr(scoring_robometer, "make_robometer_pre_post_processors", fake_make_processors)

    scorer = make_robometer_frame_scorer(model)

    assert scorer.preprocessor is preprocessor
    assert received["config"].max_frames is None
    assert config.max_frames == 8


@pytest.mark.parametrize("task_key", ["task", "missing_task", "instruction"])
def test_robometer_frame_scorer_returns_dense_namespaced_signals(task_key: str):
    image_key = "observation.images.top"
    grad_modes: list[bool] = []

    class FakeDataset:
        absolute_to_relative_idx = None
        meta = SimpleNamespace(episodes=[{"dataset_from_index": 0, "dataset_to_index": 3}])

        def __getitem__(self, index: int):
            return {
                image_key: torch.full((3, 2, 2), float(index)),
                "task": "pick up the cube",
                "instruction": f"custom task at frame {index}",
            }

    class FakeModel:
        config = SimpleNamespace(
            type="robometer",
            image_key=image_key,
            task_key=task_key,
            default_task=None,
            pretrained_path="test/robometer",
            pretrained_revision="model-revision",
        )

        def eval(self):
            return self

        def predict_progress(self, batch):
            grad_modes.append(torch.is_grad_enabled())
            progress = batch["observation.encoded"].float() / 2.0
            return RobometerOutput(
                progress=progress,
                success_probability=progress / 2.0,
            )

    dataset = FakeDataset()
    model = FakeModel()
    preprocessor = _make_preprocessor(task_key=task_key)
    scorer = RobometerFrameScorer(model, preprocessor, batch_size=2, num_subsampled_frames=2)

    result = scorer.score_episode(dataset, episode_index=0)

    assert grad_modes == [False, False]

    np.testing.assert_array_equal(result.frame_indices, np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(result.signals[PROGRESS_SIGNAL], np.asarray([0.0, 0.5, 1.0]))
    np.testing.assert_array_equal(result.signals[SUCCESS_PROBABILITY_SIGNAL], np.asarray([0.0, 0.25, 0.5]))
    assert set(scorer.descriptors) == {PROGRESS_SIGNAL, SUCCESS_PROBABILITY_SIGNAL}
    assert scorer.descriptors[PROGRESS_SIGNAL].bounds == (0.0, 1.0)
    assert scorer.descriptors[PROGRESS_SIGNAL].greater_is_better is True
    task = {"task": "pick up the cube", "instruction": "custom task at frame 0"}.get(task_key)
    assert preprocessor.steps[0].tasks == [task, task]
    assert scorer.name == "robometer"
    assert scorer.provenance == {
        "model": {"type": "robometer", "id": "test/robometer", "revision": "model-revision"},
        "scorer": {
            "version": 1,
            "options": {
                "image_key": image_key,
                "task_key": task_key,
                "default_task": None,
                "num_subsampled_frames": 2,
            },
        },
    }

    episode_frames = torch.stack([dataset[index][image_key] for index in range(3)])
    prefix_batch = torch.stack([episode_frames[indices] for indices in build_subsample_indices(3, 2)])
    direct = model.predict_progress(preprocessor({image_key: prefix_batch, task_key: task}))
    np.testing.assert_allclose(
        result.signals[PROGRESS_SIGNAL],
        direct.progress[:, -1].numpy(),
    )
    np.testing.assert_allclose(
        result.signals[SUCCESS_PROBABILITY_SIGNAL],
        direct.success_probability[:, -1].numpy(),
    )


@pytest.mark.parametrize("task_key", ["task", "instruction"])
@pytest.mark.parametrize("filtered", [False, True])
def test_robometer_frame_scorer_supports_episode_filtered_dataset_views(task_key: str, filtered: bool):
    image_key = "observation.images.top"

    class FakeDataset:
        def __init__(self) -> None:
            self.meta = SimpleNamespace(
                episodes=[
                    {"dataset_from_index": 0, "dataset_to_index": 5},
                    {"dataset_from_index": 5, "dataset_to_index": 7},
                ],
            )
            # A filtered view holds only episode 1, so its rows start at 0.
            self.absolute_to_relative_idx = {5: 0, 6: 1} if filtered else None
            self.row_requests: list[int] = []

        def __getitem__(self, position: int):
            self.row_requests.append(position)
            absolute_index = position + 5 if filtered else position
            return {
                image_key: torch.full((3, 2, 2), float(absolute_index)),
                "task": "selected task",
                "instruction": "selected instruction",
            }

    class FakeModel:
        config = SimpleNamespace(image_key=image_key, task_key=task_key)

        def eval(self):
            return self

        def predict_progress(self, batch):
            values = batch["observation.encoded"].float() / 10.0
            return RobometerOutput(progress=values, success_probability=values)

    preprocessor = _make_preprocessor(task_key=task_key)
    scorer = RobometerFrameScorer(FakeModel(), preprocessor, num_subsampled_frames=2)
    dataset = FakeDataset()

    result = scorer.score_episode(dataset, episode_index=1)

    np.testing.assert_allclose(result.signals[PROGRESS_SIGNAL], [0.5, 0.6])
    assert sorted(set(dataset.row_requests)) == ([0, 1] if filtered else [5, 6])
    assert preprocessor.steps[0].tasks == [
        "selected instruction" if task_key == "instruction" else "selected task"
    ]


def test_robometer_frame_scorer_publishes_a_dataset_score(tmp_path):
    image_key = "observation.images.top"

    class FakeDataset(LeRobotDataset):
        absolute_to_relative_idx = None
        episodes = None
        _storage_root = None
        reader = SimpleNamespace()
        writer = None

        def __init__(self) -> None:
            self.root = tmp_path
            self.repo_id = "test/dataset"
            self.revision = "dataset-revision"
            self.meta = SimpleNamespace(
                total_episodes=1,
                total_frames=3,
                storage_format="lerobot",
                episodes=[{"dataset_from_index": 0, "dataset_to_index": 3}],
            )

        def __getitem__(self, index: int):
            return {image_key: torch.full((3, 2, 2), float(index)), "task": "pick up the cube"}

    class FakeModel:
        config = SimpleNamespace(
            type="robometer",
            image_key=image_key,
            task_key="task",
            default_task=None,
            pretrained_path="test/robometer",
            pretrained_revision="model-revision",
        )

        def eval(self):
            return self

        def predict_progress(self, batch):
            progress = batch["observation.encoded"].float() / 2.0
            return RobometerOutput(
                progress=progress,
                success_probability=progress / 2.0,
            )

    scorer = RobometerFrameScorer(FakeModel(), _make_preprocessor(), batch_size=2, num_subsampled_frames=2)
    dataset = FakeDataset()
    score_dataset(dataset, scorer)

    scores = load_scores(dataset, "robometer")
    assert list(scores[PROGRESS_SIGNAL]) == [0.0, 0.5, 1.0]
    assert list(scores[SUCCESS_PROBABILITY_SIGNAL]) == [0.0, 0.25, 0.5]
    assert scores.features[PROGRESS_SIGNAL].dtype == "float32"
    assert scores.features[SUCCESS_PROBABILITY_SIGNAL].dtype == "float32"
    assert scores.column_names == [
        "index",
        "episode_index",
        "frame_index",
        PROGRESS_SIGNAL,
        SUCCESS_PROBABILITY_SIGNAL,
    ]
