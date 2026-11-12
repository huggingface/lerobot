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
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import pandas as pd

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import PolicyProcessorPipeline, ProcessorStep
from lerobot.rewards.sarm import SARMOutput, scoring_sarm
from lerobot.rewards.sarm.scoring_sarm import (
    DENSE_PROGRESS_SIGNAL,
    SPARSE_PROGRESS_SIGNAL,
    SARMFrameScorer,
    make_sarm_frame_scorer,
)

pytestmark = pytest.mark.dataset

IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"
# Two past frames, the current one, two future ones, then one rewind placeholder.
DELTAS = [-2, -1, 0, 1, 2, -3]


@dataclass
class _WindowEncoderStep(ProcessorStep):
    """Stand in for SARM's encoder: expose each window's frame ids as `video_features`."""

    calls: list[dict] = field(default_factory=list)

    def __call__(self, transition):
        observation = dict(transition[TransitionKey.OBSERVATION])
        complementary = transition[TransitionKey.COMPLEMENTARY_DATA]
        self.calls.append({"task": complementary["task"], "index": complementary["index"].tolist()})
        window_ids = observation[IMAGE_KEY][:, :, 0, 0, 0]
        observation["video_features"] = window_ids
        observation["text_features"] = torch.zeros(len(window_ids), 1)
        observation["state_features"] = observation.get(STATE_KEY)
        observation["lengths"] = torch.full((len(window_ids),), 5)
        return {**transition, TransitionKey.OBSERVATION: observation}

    def transform_features(self, features):
        return features


class _FakeModel:
    def __init__(self, *, dual_heads: bool = False, fail_on: int | None = None):
        self.config = SimpleNamespace(
            type="sarm",
            pretrained_path="test/sarm",
            pretrained_revision=None,
            image_key=IMAGE_KEY,
            state_key=STATE_KEY,
            n_obs_steps=4,
            observation_delta_indices=DELTAS,
            uses_dual_heads=dual_heads,
        )
        self.fail_on = fail_on
        self.heads_seen = []

    def eval(self):
        return self

    def predict_progress(self, batch, *, head_mode):
        self.heads_seen.append(head_mode)
        window_ids = batch["video_features"].float()
        if self.fail_on is not None and (window_ids[:, 2] == self.fail_on).any():
            raise RuntimeError("bad frame")
        # Progress is the frame id divided by 10, plus 0.5 for the dense head.
        progress = window_ids / 10.0 + (0.5 if head_mode == "dense" else 0.0)
        batch_size, steps = progress.shape
        return SARMOutput(
            progress=progress,
            stage_probabilities=torch.ones(batch_size, steps, 1),
            stage_confidence=torch.ones(batch_size, steps),
            valid_mask=torch.ones(batch_size, steps, dtype=torch.bool),
        )


class _FakeDataset:
    def __init__(self, num_frames: int = 5, *, with_state: bool = True):
        self.meta = SimpleNamespace(
            episodes=[{"dataset_from_index": 0, "dataset_to_index": num_frames}],
            stats={"observation.state": {}},
            tasks=pd.DataFrame({"task_index": [0, 1]}, index=["another task", "pick up the cube"]),
            features={IMAGE_KEY: {}, "task_index": {}, **({STATE_KEY: {}} if with_state else {})},
        )
        self.frame_requests: list[tuple[int, list[str], list[int]]] = []

    def get_frames(
        self, episode_index: int, keys: Sequence[str], frame_indices: Sequence[int]
    ) -> dict[str, torch.Tensor]:
        assert set(keys) <= self.meta.features.keys()
        self.frame_requests.append((episode_index, list(keys), list(frame_indices)))
        return {
            key: torch.stack(
                [
                    torch.tensor(1)
                    if key == "task_index"
                    else torch.full((3, 2, 2) if key == IMAGE_KEY else (2,), float(position))
                    for position in frame_indices
                ]
            )
            for key in keys
        }


@pytest.fixture
def encoder(monkeypatch):
    step = _WindowEncoderStep()
    received = {}

    def fake_make_processors(config, *, dataset_stats=None, dataset_meta=None):
        received.update(dataset_stats=dataset_stats, dataset_meta=dataset_meta)
        return PolicyProcessorPipeline(steps=[step]), object()

    monkeypatch.setattr(scoring_sarm, "make_sarm_pre_post_processors", fake_make_processors)
    step.received = received
    return step


@pytest.mark.parametrize("with_state", [True, False])
def test_sarm_frame_scorer_keeps_the_window_center_of_each_frame(encoder, with_state: bool):
    model = _FakeModel()
    scorer = SARMFrameScorer(model, batch_size=2)
    dataset = _FakeDataset(with_state=with_state)

    result = scorer.score_episode(dataset, 0)

    np.testing.assert_array_equal(result.frame_indices, np.arange(5))
    np.testing.assert_allclose(result.signals[SPARSE_PROGRESS_SIGNAL], [0.0, 0.1, 0.2, 0.3, 0.4])
    assert set(result.signals) == {SPARSE_PROGRESS_SIGNAL}
    assert result.signals[SPARSE_PROGRESS_SIGNAL].dtype == np.float32
    assert [call["index"] for call in encoder.calls] == [[0, 1], [2, 3], [4]]
    assert encoder.calls[0]["task"] == "pick up the cube"
    assert encoder.received == {"dataset_stats": {"observation.state": {}}, "dataset_meta": None}
    keys = [IMAGE_KEY, STATE_KEY] if with_state else [IMAGE_KEY]
    assert dataset.frame_requests == [(0, ["task_index"], [0]), (0, keys, list(range(5)))]
    assert set(scorer.descriptors) == {SPARSE_PROGRESS_SIGNAL}
    assert scorer.descriptors[SPARSE_PROGRESS_SIGNAL].greater_is_better is True
    assert scorer.name == "sarm"
    assert scorer.provenance["scorer"]["options"] == {
        "image_key": IMAGE_KEY,
        "state_key": STATE_KEY,
        "heads": ["sparse"],
        "stride": 1,
    }


def test_sarm_frame_scorer_clamps_windows_to_the_episode(encoder, monkeypatch):
    windows = []
    original = scoring_sarm.delta_window

    def recording_delta_window(*args):
        result = original(*args)
        windows.append(result[0])
        return result

    monkeypatch.setattr(scoring_sarm, "delta_window", recording_delta_window)
    SARMFrameScorer(_FakeModel()).score_episode(_FakeDataset(num_frames=3), 0)

    assert windows == [[0, 0, 0, 1, 2, 0], [0, 0, 1, 2, 2, 0], [0, 1, 2, 2, 2, 0]]


def test_sarm_frame_scorer_interpolates_between_strided_frames(encoder):
    result = SARMFrameScorer(_FakeModel(), stride=3).score_episode(_FakeDataset(num_frames=5), 0)

    assert [call["index"] for call in encoder.calls] == [[0, 3, 4]]
    np.testing.assert_allclose(result.signals[SPARSE_PROGRESS_SIGNAL], [0.0, 0.1, 0.2, 0.3, 0.4], atol=1e-6)


def test_sarm_frame_scorer_scores_both_heads_of_a_dual_head_model(encoder):
    model = _FakeModel(dual_heads=True)
    scorer = SARMFrameScorer(model, head_mode="both")

    result = scorer.score_episode(_FakeDataset(num_frames=2), 0)

    np.testing.assert_allclose(result.signals[SPARSE_PROGRESS_SIGNAL], [0.0, 0.1])
    np.testing.assert_allclose(result.signals[DENSE_PROGRESS_SIGNAL], [0.5, 0.6])
    assert set(scorer.descriptors) == {SPARSE_PROGRESS_SIGNAL, DENSE_PROGRESS_SIGNAL}


def test_sarm_frame_scorer_falls_back_to_the_sparse_head_without_dual_heads(encoder):
    scorer = SARMFrameScorer(_FakeModel(dual_heads=False), head_mode="dense")

    assert scorer.heads == ["sparse"]
    assert set(scorer.descriptors) == {SPARSE_PROGRESS_SIGNAL}


def test_sarm_frame_scorer_stores_nan_for_frames_that_fail(encoder):
    scorer = SARMFrameScorer(_FakeModel(fail_on=2), batch_size=1)
    result = scorer.score_episode(_FakeDataset(num_frames=4), 0)

    np.testing.assert_allclose(result.signals[SPARSE_PROGRESS_SIGNAL], [0.0, 0.1, np.nan, 0.3])
    assert scorer.descriptors[SPARSE_PROGRESS_SIGNAL].allow_nan is True


def test_make_sarm_frame_scorer_forwards_options():
    scorer = make_sarm_frame_scorer(_FakeModel(dual_heads=True), head_mode="dense", stride=2, batch_size=4)

    assert (scorer.head_mode, scorer.stride, scorer.batch_size) == ("dense", 2, 4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"head_mode": "middle"}, "head_mode"), ({"stride": 0}, "stride"), ({"batch_size": 0}, "batch_size")],
)
def test_sarm_frame_scorer_rejects_invalid_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SARMFrameScorer(_FakeModel(), **kwargs)
