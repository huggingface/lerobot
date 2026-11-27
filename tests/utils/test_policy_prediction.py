#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from types import SimpleNamespace

import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor import (
    DataProcessorPipeline,
    PolicyProcessorStep,
    ProcessorStep,
    batch_to_transition,
    create_transition,
    transition_to_batch,
)
from lerobot.utils.constants import ACTION, PREDICTION

# A batched prediction for two environments.
PREDICTION_B2 = {
    "observation": {"observation.images.top": torch.zeros(2, 3, 4, 6)},
    "language": {"subtask": ["reach", "grasp"]},
    "boxes": {
        "observation.images.top": [
            {"detections": []},
            {"detections": [{"label": "cube", "bbox": [0, 0, 1, 1]}]},
        ]
    },
}


def _policy(output):
    return SimpleNamespace(predict_action_chunk=lambda batch: output, reset=lambda: None)


def test_the_policy_step_keeps_the_chunk_in_the_prediction_and_starts_the_command_from_a_copy():
    action = torch.ones(2, 8, 4)
    transition = PolicyProcessorStep(_policy({ACTION: action, PREDICTION: PREDICTION_B2}))(
        create_transition(observation={"observation.state": torch.zeros(2, 3)})
    )
    prediction = transition[TransitionKey.PREDICTION]
    assert prediction[ACTION] is action
    assert all(prediction[key] is value for key, value in PREDICTION_B2.items())
    assert torch.equal(transition[TransitionKey.ACTION], action)
    assert transition[TransitionKey.ACTION] is not action  # postprocessing never touches the prediction


def test_prediction_round_trips_through_the_batch_converters():
    batch = {"observation.state": torch.ones(2, 3), ACTION: torch.ones(2, 4), PREDICTION: PREDICTION_B2}
    assert transition_to_batch(batch_to_transition(batch))[PREDICTION] is PREDICTION_B2


def test_batch_without_prediction_gets_no_prediction_key():
    batch = {"observation.state": torch.ones(2, 3), ACTION: torch.ones(2, 4)}
    assert PREDICTION not in transition_to_batch(batch_to_transition(batch))


class _RecordPrediction(ProcessorStep):
    """A step that records the prediction it sees."""

    def __init__(self):
        self.seen = []

    def __call__(self, transition):
        self.seen.append(transition[TransitionKey.PREDICTION])
        return transition

    def transform_features(self, features):
        return features


def test_prediction_reaches_processor_steps_and_comes_back_out():
    step = _RecordPrediction()
    pipeline = DataProcessorPipeline(
        steps=[step], to_transition=batch_to_transition, to_output=transition_to_batch
    )
    output = pipeline({ACTION: torch.ones(2, 4), PREDICTION: PREDICTION_B2})

    assert step.seen == [PREDICTION_B2]
    assert output[PREDICTION] is PREDICTION_B2


def test_log_visualization_data_shows_the_step_prediction(monkeypatch):
    from lerobot.utils import visualization_utils

    logged = []
    monkeypatch.setattr(visualization_utils, "log_rerun_data", lambda **kwargs: logged.append(kwargs))
    prediction = {"language": {"subtask": ["reach"]}}
    step = create_transition(action={"m.pos": 1.0}, prediction=prediction)

    visualization_utils.log_visualization_data("rerun", action={"m.pos": 1.0}, policy_step=step)
    visualization_utils.log_visualization_data("rerun", action={"m.pos": 1.0}, policy_step=None)

    assert logged[0]["prediction"] == prediction
    assert logged[1]["prediction"] is None


def test_stacked_predicted_frames_play_back_at_their_actions():
    from lerobot.utils.visualization_utils import PredictionPlayback

    playback = PredictionPlayback()
    frames = torch.arange(3.0)[None, :, None].expand(1, 3, 2)  # [B=1, F=3, ...], frame f holds f
    chunk = {"observation": {"observation.images.top": frames}, "observation_steps": [0, 2, 3]}
    chunk["language"] = {"subtask": ["reach"]}

    shown = [playback(chunk)] + [playback(None) for _ in range(4)]
    assert shown[0]["language"] == {"subtask": ["reach"]}
    assert "observation_steps" not in shown[0]
    indices = [None if s is None else s["observation"]["observation.images.top"][0, 0].item() for s in shown]
    assert indices == [0.0, None, 1.0, 2.0, None]
    assert playback({"language": {"subtask": ["grasp"]}}) == {"language": {"subtask": ["grasp"]}}
    assert playback(None) is None  # a new chunk without frames ends the playback
