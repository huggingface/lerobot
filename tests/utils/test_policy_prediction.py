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

import torch

from lerobot.processor import (
    create_transition,
    policy_output_to_transition,
    transition_to_policy_action,
    transition_to_prediction,
)
from tests.fixtures.dummy_checkpoint_policy import make_dummy_policy


def test_bare_action_becomes_a_transition_without_prediction():
    action = torch.ones(1, 4)
    transition = policy_output_to_transition(action)

    assert transition_to_policy_action(transition) is action
    assert transition_to_prediction(transition) == {}


def test_transition_passes_through():
    transition = create_transition(action=torch.ones(1, 4), prediction={"language": {"subtask": "reach"}})
    assert policy_output_to_transition(transition) is transition
    assert transition_to_prediction(transition) == {"language": {"subtask": "reach"}}


def test_policy_without_prediction_still_returns_a_bare_action():
    policy = make_dummy_policy()
    assert isinstance(policy.select_action({"observation.state": torch.ones(1, 4)}), torch.Tensor)


def test_log_visualization_data_shows_the_step_prediction(monkeypatch):
    from lerobot.utils import visualization_utils

    logged = []
    monkeypatch.setattr(visualization_utils, "log_rerun_data", lambda **kwargs: logged.append(kwargs))
    prediction = {"language": {"subtask": "reach"}}
    step = create_transition(action={"m.pos": 1.0}, prediction=prediction)

    visualization_utils.log_visualization_data("rerun", action={"m.pos": 1.0}, policy_step=step)
    visualization_utils.log_visualization_data("rerun", action={"m.pos": 1.0}, policy_step=None)

    assert logged[0]["prediction"] == prediction
    assert logged[1]["prediction"] is None
