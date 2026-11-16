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

from lerobot.policies.rtc.relative import reanchor_relative_rtc_prefix
from lerobot.processor import TransitionKey, create_transition
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    to_absolute_actions,
    to_relative_actions,
)
from lerobot.utils.constants import OBS_STATE


@pytest.mark.parametrize("layout", ["current_first", "oldest_first", "current_in_middle"])
def test_relative_action_helpers_use_the_current_frame(layout):
    state = torch.tensor([[[1.0, 10.0], [3.0, 30.0]], [[2.0, 20.0], [4.0, 40.0]]])
    actions = torch.tensor([[[6.0, 50.0], [7.0, 60.0]], [[8.0, 70.0], [9.0, 80.0]]])
    expected = torch.tensor([[[3.0, 50.0], [4.0, 60.0]], [[4.0, 70.0], [5.0, 80.0]]])
    mask = [True, False]
    kwargs = {}
    if layout == "current_first":
        state = state.flip(dims=(1,))
    elif layout == "oldest_first":
        kwargs = {"state_anchor_index": -1}
    else:
        state = torch.cat((state, state[:, :1] + 10), dim=1)
        kwargs = {"state_anchor_index": 1}

    torch.testing.assert_close(to_relative_actions(actions, state, mask, **kwargs), expected)
    torch.testing.assert_close(to_absolute_actions(expected, state, mask, **kwargs), actions)


def test_relative_action_processor_and_rtc_share_the_configured_anchor():
    pre = RelativeActionsProcessorStep(enabled=True, state_anchor_index=-1)
    pre = RelativeActionsProcessorStep(**pre.get_config())
    post = AbsoluteActionsProcessorStep(enabled=True, relative_step=pre)
    state = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    actions = torch.tensor([[[7.0, 10.0], [8.0, 11.0]]])
    expected = torch.tensor([[[4.0, 6.0], [5.0, 7.0]]])

    relative = pre(create_transition(observation={OBS_STATE: state}, action=actions))
    torch.testing.assert_close(relative[TransitionKey.ACTION], expected)
    torch.testing.assert_close(post(relative)[TransitionKey.ACTION], actions)
    torch.testing.assert_close(reanchor_relative_rtc_prefix(actions, state, pre, None, "cpu"), expected)


def test_single_frame_state_does_not_use_the_temporal_anchor_index():
    state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    action = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    torch.testing.assert_close(
        to_relative_actions(action, state, [True, True], state_anchor_index=-1), expected
    )
    torch.testing.assert_close(
        to_absolute_actions(expected, state, [True, True], state_anchor_index=-1), action
    )
