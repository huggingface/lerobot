#!/usr/bin/env python

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

"""Isaac-GR00T N1.7 raw-state dropout training contract.

Isaac-GR00T zeroes the entire proprioceptive state of a sample with probability
``state_dropout_prob`` (configured in the checkpoint's processor sidecar) during
training only. Baseline LeRobot kept the processor deterministic, so this
regularization never activated. These tests pin the train/eval split.
"""

import torch

from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_STATE


def _make_transition():
    return {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["Move", "Move"]},
    }


def test_groot_n1_7_training_applies_raw_state_dropout_before_encoder():
    step = GrootN17PackInputsStep(
        max_state_dim=4,
        max_action_dim=4,
        normalize_min_max=False,
        training=True,
        state_dropout_prob=1.0,
    )

    output = step(_make_transition())

    expected = torch.zeros(2, 1, 4)
    torch.testing.assert_close(output[TransitionKey.OBSERVATION]["state"], expected)


def test_groot_n1_7_training_state_dropout_is_disabled_under_no_grad():
    step = GrootN17PackInputsStep(
        max_state_dim=4,
        max_action_dim=4,
        normalize_min_max=False,
        training=True,
        state_dropout_prob=1.0,
    )

    with torch.no_grad():
        output = step(_make_transition())

    expected = torch.tensor([[[1.0, 2.0, 0.0, 0.0]], [[3.0, 4.0, 0.0, 0.0]]])
    torch.testing.assert_close(output[TransitionKey.OBSERVATION]["state"], expected)


def test_groot_n1_7_eval_mode_state_dropout_is_inactive():
    step = GrootN17PackInputsStep(
        max_state_dim=4,
        max_action_dim=4,
        normalize_min_max=False,
        training=False,
        state_dropout_prob=1.0,
    )

    output = step(_make_transition())

    expected = torch.tensor([[[1.0, 2.0, 0.0, 0.0]], [[3.0, 4.0, 0.0, 0.0]]])
    torch.testing.assert_close(output[TransitionKey.OBSERVATION]["state"], expected)


def test_groot_n1_7_pack_step_serializes_dropout_prob_but_not_training_mode():
    step = GrootN17PackInputsStep(
        max_state_dim=4,
        max_action_dim=4,
        normalize_min_max=False,
        training=True,
        state_dropout_prob=0.2,
    )

    serialized = step.get_config()
    restored = GrootN17PackInputsStep(**serialized)

    assert "training" not in serialized
    assert serialized["state_dropout_prob"] == 0.2
    assert restored.training is False
    assert restored.state_dropout_prob == 0.2
