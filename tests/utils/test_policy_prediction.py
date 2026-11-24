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
"""What a policy predicts besides its action: ``prediction.*`` keys, from the policy to the viewers."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.inference import rename_prediction
from lerobot.inference.execution import prediction_at
from lerobot.inference.prediction import validate_prediction
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import PolicyProcessorStep, batch_to_transition, create_transition, transition_to_batch
from lerobot.utils.constants import ACTION, PREDICTED_ACTION, PREDICTION

# What a policy predicted for two environments.
PREDICTION_B2 = {
    "prediction.images.top": torch.zeros(2, 3, 3, 4, 6),
    "prediction.images.top_steps": [0, 2, 5],
    "prediction.subtask": ["reach", "grasp"],
    "prediction.boxes.top": [
        {"detections": []},
        {"detections": [{"label": "cube", "bbox": [0.0, 0.0, 1.0, 1.0]}]},
    ],
}
DECLARED = {
    "prediction.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 6)),
    "prediction.subtask": PolicyFeature(type=FeatureType.LANGUAGE, shape=()),
    "prediction.boxes.top": PolicyFeature(type=FeatureType.LANGUAGE, shape=()),
}


def _policy(output):
    return SimpleNamespace(predict_action_chunk=lambda batch: output, reset=lambda: None)


def test_the_policy_step_keeps_the_output_as_the_prediction_and_starts_the_command_from_a_copy():
    action = torch.ones(2, 8, 4)
    transition = PolicyProcessorStep(_policy({ACTION: action, **PREDICTION_B2}))(
        create_transition(observation={"observation.state": torch.zeros(2, 3)})
    )
    prediction = transition[TransitionKey.PREDICTION]
    assert prediction[PREDICTED_ACTION] is action
    assert all(prediction[key] is value for key, value in PREDICTION_B2.items())
    assert torch.equal(transition[TransitionKey.ACTION], action)
    assert transition[TransitionKey.ACTION] is not action  # postprocessing never touches the prediction


def test_the_policy_step_rejects_outputs_outside_the_prediction_namespace():
    with pytest.raises(ValueError, match=r"prediction\.\*"):
        PolicyProcessorStep(_policy({ACTION: torch.ones(1, 8, 4), "subtask": ["reach"]}))(create_transition())


def test_prediction_round_trips_through_the_batch_converters():
    batch = {"observation.state": torch.ones(2, 3), ACTION: torch.ones(2, 4), PREDICTION: PREDICTION_B2}
    assert transition_to_batch(batch_to_transition(batch))[PREDICTION] is PREDICTION_B2


def test_predictions_must_be_declared_and_shaped_as_declared():
    assert (
        validate_prediction(PREDICTION_B2, DECLARED, batch_size=2, chunk_length=8).keys()
        == PREDICTION_B2.keys()
    )
    with pytest.raises(ValueError, match="prediction_features omit"):
        validate_prediction({"prediction.plan": ["x", "y"]}, DECLARED, batch_size=2, chunk_length=8)
    with pytest.raises(ValueError, match="must be a"):
        frames = {**PREDICTION_B2, "prediction.images.top": torch.zeros(2, 3, 3, 8, 8)}
        validate_prediction(frames, DECLARED, batch_size=2, chunk_length=8)
    with pytest.raises(ValueError, match="ascending chunk indices"):
        late = {**PREDICTION_B2, "prediction.images.top_steps": [0, 2, 8]}
        validate_prediction(late, DECLARED, batch_size=2, chunk_length=8)
    with pytest.raises(ValueError, match="one value per environment"):
        validate_prediction({"prediction.subtask": ["reach"]}, DECLARED, batch_size=2, chunk_length=8)


def test_each_action_shows_the_frame_it_belongs_to():
    frames = torch.arange(3.0).reshape(1, 3, 1)  # frame f holds f
    prediction = {
        "prediction.images.top": frames,
        "prediction.images.top_steps": [0, 2, 3],
        "prediction.subtask": ["reach"],
    }
    shown = [prediction_at(prediction, index, first=index == 0) for index in range(5)]
    assert shown[0]["prediction.subtask"] == ["reach"]
    assert [None if s is None else s["prediction.images.top"].item() for s in shown] == [
        0.0,
        None,
        1.0,
        2.0,
        None,
    ]
    # A chunk whose first actions were dropped (RTC) starts from the frame already reached.
    assert prediction_at(prediction, 2, first=True)["prediction.images.top"].item() == 1.0


def test_predictions_name_cameras_as_the_robot_does():
    names = {"observation.images.top": "observation.images.front"}
    renamed = rename_prediction(PREDICTION_B2, names)
    assert set(renamed) == {
        "prediction.images.front",
        "prediction.images.front_steps",
        "prediction.subtask",
        "prediction.boxes.front",
    }


def test_log_visualization_data_shows_the_step_prediction(monkeypatch):
    from lerobot.utils import visualization_utils

    logged = []
    monkeypatch.setattr(visualization_utils, "log_rerun_data", lambda **kwargs: logged.append(kwargs))
    prediction = {"prediction.subtask": ["reach"]}
    step = create_transition(action={"m.pos": 1.0}, prediction=prediction)

    visualization_utils.log_visualization_data("rerun", action={"m.pos": 1.0}, policy_step=step)
    visualization_utils.log_visualization_data("rerun", action={"m.pos": 1.0}, policy_step=None)

    assert logged[0]["prediction"] == prediction
    assert logged[1]["prediction"] is None
