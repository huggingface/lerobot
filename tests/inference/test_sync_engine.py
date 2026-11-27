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
"""The one chunk loop: windows from the config, chunk playback, predictions per chunk."""

from dataclasses import dataclass, replace

import pytest
import torch

from lerobot.configs.sampling import FrameSampler
from lerobot.configs.types import Deltas, FeatureType
from lerobot.inference import FeatureHistory, SyncInferenceEngine, policy_history
from lerobot.lerobot_types import PolicyOutput
from lerobot.policies import ExecutionMode
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE, PREDICTION
from tests.inference.policy_helpers import ConformingPolicy, tiny_config
from tests.inference.rtc_helpers import Pipeline


@dataclass
class WindowConfig(ACTConfig):
    """A config whose training samples reach two frames into the past and one into the future."""

    @property
    def training_sampler(self) -> FrameSampler:
        window = (-4, -2, 0, 2)
        return FrameSampler(
            Deltas(
                observation={FeatureType.STATE: window, FeatureType.ENV: window},
                action=range(self.chunk_size),
            )
        )

    @property
    def inference_deltas(self) -> Deltas:
        window = (-4, -2, 0)
        return Deltas(observation={FeatureType.STATE: window, FeatureType.ENV: window})


def window_config(**kwargs) -> WindowConfig:
    return WindowConfig(**{**vars(tiny_config()), **kwargs})


class RecordingPolicy(ConformingPolicy):
    """Records every chunk call; returns actions encoding the call index."""

    name = "recording_sync_engine"

    def __init__(self, config, spec_changes=None):
        super().__init__(config)
        self.batches: list[dict] = []
        self.spec_changes = spec_changes or {}

    def chunk_inference_spec(self):
        return replace(super().chunk_inference_spec(), **self.spec_changes)

    def predict_action_chunk(self, batch) -> PolicyOutput:
        self.batches.append({key: value.clone() for key, value in batch.items() if torch.is_tensor(value)})
        steps = self.chunk_inference_spec().prediction_steps
        return {
            ACTION: torch.arange(steps, dtype=torch.float32).repeat(3, 1).T[None] + 100 * len(self.batches)
        }


def test_inputs_are_the_offsets_of_the_inference_deltas():
    @dataclass
    class WithCommands(WindowConfig):
        @property
        def inference_deltas(self) -> Deltas:
            observation = {**super().inference_deltas.observation, FeatureType.ACTION: (-2, -1)}
            return Deltas(observation=observation)

    config = WithCommands(**vars(tiny_config()))
    assert config.inference_deltas.of(OBS_STATE, FeatureType.STATE) == (-4, -2, 0)
    assert tiny_config().inference_deltas.of(OBS_STATE, FeatureType.STATE) is None  # the current frame
    history = policy_history(config)
    assert history.observations == (
        FeatureHistory(OBS_STATE, (-4, -2, 0)),
        FeatureHistory(OBS_ENV_STATE, (-4, -2, 0)),
    )
    assert history.actions == (-2, -1)


def test_a_window_reaching_the_past_must_include_the_current_frame():
    @dataclass
    class Broken(ACTConfig):
        @property
        def inference_deltas(self) -> Deltas:
            return Deltas(observation={FeatureType.STATE: (-2, -1)})

    with pytest.raises(ValueError, match="current frame"):
        policy_history(Broken(**vars(tiny_config())))


@pytest.mark.dataset
def test_chunk_batches_are_the_training_samples(tmp_path, empty_lerobot_dataset_factory):
    """At every chunk boundary the policy sees what ``LeRobotDataset[t]`` holds for its offsets ``<= 0``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps, frames = 10, 14
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (3,), "names": ["a", "b", "c"]},
        OBS_ENV_STATE: {"dtype": "float32", "shape": (3,), "names": ["a", "b", "c"]},
        ACTION: {"dtype": "float32", "shape": (3,), "names": ["a", "b", "c"]},
    }
    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "ds", features=features, use_videos=False, fps=fps
    )
    for t in range(frames):
        value = torch.full((3,), float(t))
        dataset.add_frame({OBS_STATE: value, OBS_ENV_STATE: -value, ACTION: value, "task": "x"})
    dataset.save_episode()
    dataset.finalize()
    config = window_config()
    loaded = LeRobotDataset(dataset.repo_id, root=dataset.root, sampler=config.training_sampler)

    policy = RecordingPolicy(config)
    executor = SyncInferenceEngine(policy, Pipeline(), Pipeline())
    calls = []
    for t in range(frames):
        if not executor.queued_actions:
            calls.append(t)
        sample = loaded[t]
        executor.step(
            {OBS_STATE: sample[OBS_STATE][2:3], OBS_ENV_STATE: sample[OBS_ENV_STATE][2:3]}, float(t)
        )

    assert calls == list(range(0, frames, 3))  # n_action_steps of each 8-action chunk
    for t, batch in zip(calls, policy.batches, strict=True):
        expected = loaded[t]
        for key in (OBS_STATE, OBS_ENV_STATE):
            torch.testing.assert_close(batch[key][0], expected[key][:3])  # offsets -4, -2, 0
            assert batch[f"{key}_is_pad"][0].tolist() == expected[f"{key}_is_pad"][:3].tolist()


def test_chunks_play_back_their_execution_steps_and_predictions_ride_on_their_first_action():
    class Predicting(RecordingPolicy):
        name = "predicting_executor"

        def predict_action_chunk(self, batch) -> PolicyOutput:
            actions = super().predict_action_chunk(batch)[ACTION]
            return {ACTION: actions, PREDICTION: {"language": {"subtask": [f"chunk {len(self.batches)}"]}}}

    policy = Predicting(tiny_config())
    executor = SyncInferenceEngine(policy, Pipeline(), Pipeline())
    actions, predictions = [], []
    for t in range(7):
        actions.append(
            executor.step({OBS_STATE: torch.zeros(1, 3), OBS_ENV_STATE: torch.zeros(1, 3)}, float(t))
        )
        predictions.append(executor.prediction)
    assert [a[0, 0].item() for a in actions] == [100, 101, 102, 200, 201, 202, 300]
    subtasks = [p["language"]["subtask"][0] if p else None for p in predictions]
    assert subtasks == ["chunk 1", None, None, "chunk 2", None, None, "chunk 3"]


def test_a_chunk_of_the_wrong_length_is_rejected():
    policy = RecordingPolicy(tiny_config(), {"prediction_steps": 9})
    policy.predict_action_chunk = lambda batch: {ACTION: torch.zeros(1, 8, 3)}
    executor = SyncInferenceEngine(policy, Pipeline(), Pipeline())
    with pytest.raises(ValueError, match=r"must return \[B, 9, A\]"):
        executor.step({OBS_STATE: torch.zeros(1, 3), OBS_ENV_STATE: torch.zeros(1, 3)}, 0.0)


def test_complete_execution_plays_every_returned_action_of_shorter_chunks():
    changes = {"execution_steps": 8, "modes": (ExecutionMode.CHUNK,), "complete_execution": True}
    policy = RecordingPolicy(window_config(), changes)
    lengths = iter([4, 8, 8])
    policy.predict_action_chunk = lambda batch: (
        policy.batches.append(dict(batch)) or {ACTION: torch.zeros(1, next(lengths), 3)}
    )
    executor = SyncInferenceEngine(policy, Pipeline(), Pipeline())
    calls = []
    for t in range(13):
        if not executor.queued_actions:
            calls.append(t)
        executor.step({OBS_STATE: torch.zeros(1, 3), OBS_ENV_STATE: torch.full((1, 3), float(t))}, float(t))
    assert calls == [0, 4, 12]
    # Each call observes the result of the previous chunk's last action: its window ends there.
    assert policy.batches[1][OBS_ENV_STATE][0, :, 0].tolist() == [0.0, 2.0, 4.0]
    assert policy.batches[2][OBS_ENV_STATE][0, :, 0].tolist() == [8.0, 10.0, 12.0]


def test_dropping_queued_actions_predicts_on_the_next_tick():
    policy = RecordingPolicy(tiny_config())
    executor = SyncInferenceEngine(policy, Pipeline(), Pipeline())
    observation = {OBS_STATE: torch.zeros(1, 3), OBS_ENV_STATE: torch.zeros(1, 3)}
    executor.step(observation, 0.0)
    executor.drop_queued_actions()
    assert executor.step(observation, 1.0)[0, 0].item() == 200
    executor.reset()
    assert policy.resets == 2 and executor.step(observation, 0.0)[0, 0].item() == 300
