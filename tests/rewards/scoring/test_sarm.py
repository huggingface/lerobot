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

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import PolicyProcessorPipeline, ProcessorStep
from lerobot.rewards.sarm import SARMConfig, SARMOutput, processor_sarm, scoring_sarm
from lerobot.rewards.sarm.scoring_sarm import (
    DENSE_PROGRESS_SIGNAL,
    DENSE_STAGE_SIGNAL,
    SPARSE_PROGRESS_SIGNAL,
    SPARSE_STAGE_SIGNAL,
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
    def __init__(
        self,
        *,
        dual_heads: bool = False,
        fail_on: int | None = None,
        head_mode: str = "sparse",
        stride: int = 1,
    ):
        self.config = SimpleNamespace(
            type="sarm",
            pretrained_path="test/sarm",
            pretrained_revision=None,
            device="cpu",
            image_key=IMAGE_KEY,
            state_key=STATE_KEY,
            n_obs_steps=4,
            observation_delta_indices=DELTAS,
            uses_dual_heads=dual_heads,
            head_mode=head_mode,
            stride=stride,
        )
        self.fail_on = fail_on
        self.heads_seen = []

    def eval(self):
        return self

    def predict_progress(self, batch, *, head_mode):
        assert not torch.is_grad_enabled()
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
    absolute_to_relative_idx: dict[int, int] | None = None

    def __init__(self, num_frames: int = 5, *, with_state: bool = True):
        self.meta = SimpleNamespace(
            episodes=[{"dataset_from_index": 0, "dataset_to_index": num_frames}],
            stats={"observation.state": {}},
        )
        self.with_state = with_state
        self.row_requests: list[int] = []

    def __getitem__(self, position: int):
        self.row_requests.append(position)
        sample = {IMAGE_KEY: torch.full((3, 2, 2), float(position)), "task": "pick up the cube"}
        if self.with_state:
            sample[STATE_KEY] = torch.full((2,), float(position))
        return sample


@pytest.fixture
def encoder(monkeypatch):
    step = _WindowEncoderStep()
    received = {}

    def fake_from_pretrained(path, config_filename, *, revision, overrides):
        received.update(path=path, config_filename=config_filename, revision=revision, overrides=overrides)
        return PolicyProcessorPipeline(steps=[step])

    monkeypatch.setattr(
        scoring_sarm, "PolicyProcessorPipeline", SimpleNamespace(from_pretrained=fake_from_pretrained)
    )
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
    # The checkpoint's saved preprocessor, with its training normalization statistics, is used.
    assert encoder.received == {
        "path": "test/sarm",
        "config_filename": "policy_preprocessor.json",
        "revision": None,
        "overrides": {
            "SARMEncodingProcessorStep": {"config": model.config},
            "device_processor": {"device": "cpu"},
        },
    }
    # Each frame is read once, for both the image and the state.
    assert dataset.row_requests == list(range(5))
    assert set(scorer.descriptors) == {SPARSE_PROGRESS_SIGNAL}
    assert scorer.descriptors[SPARSE_PROGRESS_SIGNAL].greater_is_better is True
    assert scorer.name == "sarm"


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
    result = SARMFrameScorer(_FakeModel(stride=3)).score_episode(_FakeDataset(num_frames=5), 0)

    assert [call["index"] for call in encoder.calls] == [[0, 3, 4]]
    np.testing.assert_allclose(result.signals[SPARSE_PROGRESS_SIGNAL], [0.0, 0.1, 0.2, 0.3, 0.4], atol=1e-6)


def test_sarm_frame_scorer_scores_both_heads_of_a_dual_head_model(encoder):
    model = _FakeModel(dual_heads=True, head_mode="both")
    scorer = SARMFrameScorer(model)

    result = scorer.score_episode(_FakeDataset(num_frames=2), 0)

    np.testing.assert_allclose(result.signals[SPARSE_PROGRESS_SIGNAL], [0.0, 0.1])
    np.testing.assert_allclose(result.signals[DENSE_PROGRESS_SIGNAL], [0.5, 0.6])
    assert set(scorer.descriptors) == {SPARSE_PROGRESS_SIGNAL, DENSE_PROGRESS_SIGNAL}


def test_sarm_frame_scorer_falls_back_to_the_sparse_head_without_dual_heads(encoder):
    scorer = SARMFrameScorer(_FakeModel(dual_heads=False, head_mode="dense"))

    assert scorer.heads == ["sparse"]
    assert set(scorer.descriptors) == {SPARSE_PROGRESS_SIGNAL}


def test_sarm_frame_scorer_stores_nan_for_frames_that_fail(encoder):
    scorer = SARMFrameScorer(_FakeModel(fail_on=2), batch_size=1)
    result = scorer.score_episode(_FakeDataset(num_frames=4), 0)

    np.testing.assert_allclose(result.signals[SPARSE_PROGRESS_SIGNAL], [0.0, 0.1, np.nan, 0.3])
    assert scorer.descriptors[SPARSE_PROGRESS_SIGNAL].allow_nan is True


class _StagedModel(_FakeModel):
    """Predict subtask `frame id // 2` (capped at the last subtask) at each window position."""

    def __init__(self, names: list[str], **kwargs):
        super().__init__(**kwargs)
        self.config.sparse_subtask_names = names
        self.config.dense_subtask_names = names

    def predict_progress(self, batch, *, head_mode):
        output = super().predict_progress(batch, head_mode=head_mode)
        num_stages = len(self.config.sparse_subtask_names)
        stage = (batch["video_features"].long() // 2).clamp(max=num_stages - 1)
        return replace(output, stage_probabilities=torch.nn.functional.one_hot(stage, num_stages).float())


def test_sarm_frame_scorer_records_the_predicted_subtask_of_each_frame(encoder):
    scorer = SARMFrameScorer(_StagedModel(["grasp", "fold", "place"]), batch_size=2)

    result = scorer.score_episode(_FakeDataset(num_frames=6), 0)

    np.testing.assert_array_equal(result.signals[SPARSE_STAGE_SIGNAL], [0, 0, 1, 1, 2, 2])
    assert result.signals[SPARSE_STAGE_SIGNAL].dtype == np.float32
    descriptor = scorer.descriptors[SPARSE_STAGE_SIGNAL]
    assert descriptor.labels == ("grasp", "fold", "place")
    assert (descriptor.greater_is_better, descriptor.bounds) == (None, (0.0, 2.0))


def test_sarm_frame_scorer_keeps_the_last_scored_subtask_between_strided_frames(encoder):
    result = SARMFrameScorer(_StagedModel(["grasp", "fold", "place"], stride=3)).score_episode(
        _FakeDataset(num_frames=6), 0
    )

    # Frames 0, 3 and 5 are scored, as subtasks 0, 1 and 2.
    np.testing.assert_array_equal(result.signals[SPARSE_STAGE_SIGNAL], [0, 0, 0, 1, 1, 2])


def test_sarm_frame_scorer_records_subtasks_of_both_heads(encoder):
    scorer = SARMFrameScorer(_StagedModel(["grasp", "place"], dual_heads=True, head_mode="both"))

    assert set(scorer.descriptors) == {
        SPARSE_PROGRESS_SIGNAL,
        SPARSE_STAGE_SIGNAL,
        DENSE_PROGRESS_SIGNAL,
        DENSE_STAGE_SIGNAL,
    }
    result = scorer.score_episode(_FakeDataset(num_frames=2), 0)

    assert set(result.signals) == set(scorer.descriptors)
    np.testing.assert_array_equal(result.signals[DENSE_STAGE_SIGNAL], [0, 0])


def test_sarm_frame_scorer_skips_the_subtask_of_a_single_stage_head(encoder):
    scorer = SARMFrameScorer(_StagedModel(["task"]))

    assert set(scorer.descriptors) == {SPARSE_PROGRESS_SIGNAL}
    assert set(scorer.score_episode(_FakeDataset(num_frames=2), 0).signals) == {SPARSE_PROGRESS_SIGNAL}


def test_sarm_frame_scorer_requires_a_trained_checkpoint(encoder):
    model = _FakeModel()
    model.config.pretrained_path = None

    with pytest.raises(ValueError, match="trained checkpoint"):
        SARMFrameScorer(model).score_episode(_FakeDataset(), 0)


def test_make_sarm_frame_scorer_reads_options_from_the_model_config():
    scorer = make_sarm_frame_scorer(_FakeModel(dual_heads=True, head_mode="dense", stride=2))

    assert scorer.heads == ["dense"]


def test_sarm_frame_scorer_provenance_records_settings_that_change_the_scores():
    config = SARMConfig(device="cpu", stride=2, frame_gap=10)

    options = SARMFrameScorer(SimpleNamespace(config=config)).provenance["scorer"]["options"]

    assert (options["stride"], options["frame_gap"], options["n_obs_steps"]) == (2, 10, config.n_obs_steps)
    assert not {"device", "pretrained_path"} & set(options)


class _StateFeaturesStep(ProcessorStep):
    """Stand in for SARM's CLIP encoder: pass the normalized state through as `state_features`."""

    def __init__(self, config, image_key=None, dataset_meta=None, dataset_stats=None):
        super().__init__()
        self.config = config

    def __call__(self, transition):
        observation = dict(transition[TransitionKey.OBSERVATION])
        state = observation[self.config.state_key]
        batch_size, steps = state.shape[:2]
        observation["video_features"] = torch.zeros(batch_size, steps, 1)
        observation["text_features"] = torch.zeros(batch_size, 1)
        observation["state_features"] = state
        observation["lengths"] = torch.full((batch_size,), steps)
        return {**transition, TransitionKey.OBSERVATION: observation}

    def transform_features(self, features):
        return features


class _StateModel:
    """Predict the mean of the normalized state as progress."""

    def __init__(self, config):
        self.config = config

    def eval(self):
        return self

    def predict_progress(self, batch, *, head_mode):
        progress = batch["state_features"].float().mean(-1)
        batch_size, steps = progress.shape
        return SARMOutput(
            progress=progress,
            stage_probabilities=torch.ones(batch_size, steps, 1),
            stage_confidence=torch.ones(batch_size, steps),
            valid_mask=torch.ones(batch_size, steps, dtype=torch.bool),
        )


def test_sarm_frame_scorer_normalizes_with_the_checkpoint_statistics(tmp_path, monkeypatch):
    # Save SARM's preprocessor with the training statistics, as `lerobot-train` does. The stand-in
    # encoder is saved and loaded under the real encoder's import path.
    encoder_class = type(
        "SARMEncodingProcessorStep", (_StateFeaturesStep,), {"__module__": processor_sarm.__name__}
    )
    monkeypatch.setattr(processor_sarm, "SARMEncodingProcessorStep", encoder_class)
    config = SARMConfig(device="cpu")
    training_stats = {STATE_KEY: {"mean": torch.full((2,), 1.0), "std": torch.full((2,), 2.0)}}
    preprocessor, _ = processor_sarm.make_sarm_pre_post_processors(config, dataset_stats=training_stats)
    preprocessor.save_pretrained(tmp_path)
    config.pretrained_path = str(tmp_path)
    dataset = _FakeDataset(num_frames=3)
    # Statistics of the scored dataset, which change when episodes are appended, are not used.
    dataset.meta.stats = {STATE_KEY: {"mean": torch.full((2,), 100.0), "std": torch.ones(2)}}

    result = SARMFrameScorer(_StateModel(config)).score_episode(dataset, 0)

    # The state of frame `f` is `f`, normalized with the training statistics to `(f - 1) / 2`.
    np.testing.assert_allclose(result.signals[SPARSE_PROGRESS_SIGNAL], [-0.5, 0.0, 0.5])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"head_mode": "middle"}, "head_mode"),
        ({"stride": 0}, "stride"),
    ],
)
def test_sarm_config_rejects_invalid_scoring_options(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SARMConfig(device="cpu", **kwargs)


def test_sarm_frame_scorer_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        SARMFrameScorer(_FakeModel(), batch_size=0)
