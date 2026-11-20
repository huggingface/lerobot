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

import torch

from lerobot.lerobot_types import TransitionKey
from lerobot.policies.pi0_fast.processor_pi0_fast import Pi0FastPrepareStateAndLanguageTokenizerProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.utils.constants import OBS_STATE


def test_state_prompt_step_gives_the_same_prompt_on_every_call():
    step = Pi0FastPrepareStateAndLanguageTokenizerProcessorStep()
    transition = create_transition(
        observation={OBS_STATE: torch.tensor([[-1.0, 0.0, 1.0]])},
        complementary_data={"task": ["pick_up the cube"]},
    )

    first = step(transition)
    second = step(transition)

    assert first[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["Task: pick up the cube, State: 0 128 255;\n"]
    assert second[TransitionKey.COMPLEMENTARY_DATA]["task"] == first[TransitionKey.COMPLEMENTARY_DATA]["task"]
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["pick_up the cube"]
