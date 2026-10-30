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

import numpy as np
import pytest
import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.processor.gym_action_processor import Numpy2TorchActionProcessorStep
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY


@pytest.mark.parametrize("has_action", [False, True])
def test_numpy_conversion_preserves_input_complementary_data(has_action):
    teleop_action = np.array([1.0, 2.0], dtype=np.float32)
    metadata = {"episode": 3}
    complementary_data = {TELEOP_ACTION_KEY: teleop_action, "metadata": metadata}
    transition = {TransitionKey.COMPLEMENTARY_DATA: complementary_data}
    action = np.array([3.0, 4.0], dtype=np.float64)
    if has_action:
        transition[TransitionKey.ACTION] = action

    result = Numpy2TorchActionProcessorStep()(transition)

    assert transition[TransitionKey.COMPLEMENTARY_DATA] is complementary_data
    assert complementary_data[TELEOP_ACTION_KEY] is teleop_action
    assert isinstance(teleop_action, np.ndarray)
    assert result[TransitionKey.COMPLEMENTARY_DATA] is not complementary_data
    torch.testing.assert_close(
        result[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY], torch.tensor([1.0, 2.0])
    )
    assert result[TransitionKey.COMPLEMENTARY_DATA]["metadata"] is metadata
    if has_action:
        assert transition[TransitionKey.ACTION] is action
        torch.testing.assert_close(result[TransitionKey.ACTION], torch.from_numpy(action))


def test_numpy_conversion_can_reuse_the_same_transition():
    teleop_action = np.array([1.0, 2.0], dtype=np.float32)
    transition = {TransitionKey.COMPLEMENTARY_DATA: {TELEOP_ACTION_KEY: teleop_action}}
    step = Numpy2TorchActionProcessorStep()
    first = step(transition)
    first[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY] = torch.zeros(2)
    second = step(transition)
    torch.testing.assert_close(
        second[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY], torch.tensor([1.0, 2.0])
    )


@pytest.mark.parametrize("complementary_data", [None, {}, {"label": "pick"}])
def test_missing_teleop_action_is_unchanged(complementary_data):
    transition = {TransitionKey.COMPLEMENTARY_DATA: complementary_data}
    result = Numpy2TorchActionProcessorStep()(transition)
    assert result == transition
