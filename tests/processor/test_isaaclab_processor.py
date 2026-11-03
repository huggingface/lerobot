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

import pytest
import torch

from lerobot.processor import create_transition
from lerobot.processor.env_processor import IsaaclabArenaProcessorStep
from lerobot.utils.constants import OBS_STATE


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "sliced"])
def test_flatten_state_components_preserves_batch_and_feature_order(layout):
    component = torch.arange(48, dtype=torch.float64).reshape(2, 4, 6)
    if layout == "transposed":
        component = component.transpose(1, 2)
    elif layout == "sliced":
        component = component[:, ::2, :]
    positions = torch.tensor([[100.0, 101.0], [200.0, 201.0]])
    observation = {"observation.policy": {"positions": positions, "history": component}}
    step = IsaaclabArenaProcessorStep(state_keys=("history", "positions"), camera_keys=())

    result = step(create_transition(observation=observation))["observation"][OBS_STATE]

    expected = torch.stack(
        [torch.cat((component[i].contiguous().view(-1), positions[i])) for i in range(2)]
    ).float()
    torch.testing.assert_close(result, expected)
    assert observation["observation.policy"]["history"] is component
