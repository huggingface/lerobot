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

import pytest
import torch

from lerobot.lerobot_types import PolicyOutput, TransitionKey
from lerobot.processor import (
    DataProcessorPipeline,
    ProcessorStep,
    batch_to_transition,
    create_transition,
    policy_output_to_transition,
    transition_to_batch,
    transition_to_policy_action,
    transition_to_prediction,
)
from lerobot.utils.constants import ACTION, PREDICTION
from tests.fixtures.dummy_checkpoint_policy import make_dummy_policy

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


def test_bare_action_becomes_a_transition_without_prediction():
    action = torch.ones(1, 4)
    transition = policy_output_to_transition(action)

    assert transition_to_policy_action(transition) is action
    assert transition_to_prediction(transition) == {}


def test_policy_output_becomes_a_transition_with_its_batched_prediction():
    action = torch.ones(2, 4)
    output: PolicyOutput = {ACTION: action, PREDICTION: PREDICTION_B2}
    transition = policy_output_to_transition(output)

    assert transition_to_policy_action(transition) is action
    assert transition_to_prediction(transition) is PREDICTION_B2
    assert transition[TransitionKey.OBSERVATION] is None


def test_policy_output_without_prediction_has_an_empty_one():
    transition = policy_output_to_transition({ACTION: torch.ones(1, 4)})
    assert transition_to_prediction(transition) == {}


def test_policy_output_action_must_be_a_tensor():
    with pytest.raises(ValueError, match="PolicyAction"):
        policy_output_to_transition({ACTION: [1.0, 2.0]})


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
        self.seen.append(transition_to_prediction(transition))
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


def test_policy_without_prediction_still_returns_a_bare_action():
    policy = make_dummy_policy()
    assert isinstance(policy.select_action({"observation.state": torch.ones(1, 4)}), torch.Tensor)


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
