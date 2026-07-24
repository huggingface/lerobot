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

import numpy as np
import pytest
import torch

from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    TransitionKey,
    create_transition,
    to_absolute_actions,
    to_relative_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_relative_actions_use_explicit_state_action_index_map_for_2d_actions():
    state = torch.tensor([[10.0, 0.1, 20.0, 0.2, 30.0]])
    actions = torch.tensor([[11.0, 22.0, 33.0]])
    mask = [True, True, True]
    state_action_index_map = [0, 2, 4]

    relative = to_relative_actions(actions, state, mask, state_action_index_map)

    torch.testing.assert_close(relative, torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(
        to_absolute_actions(relative, state, mask, state_action_index_map),
        actions,
    )


def test_relative_actions_round_trip_chunked_actions_with_explicit_state_action_index_map():
    state = torch.tensor([[10.0, 0.1, 20.0, 0.2, 30.0, 0.3, 40.0, 0.4, 50.0, 0.5, 60.0, 0.6, 99.0]])
    actions = torch.tensor(
        [
            [
                [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 7.0],
                [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 8.0],
            ]
        ]
    )
    mask = [True, True, True, True, True, True, False]
    state_action_index_map = [0, 2, 4, 6, 8, 10, 12]

    relative = to_relative_actions(actions, state, mask, state_action_index_map)

    expected_relative = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 8.0],
            ]
        ]
    )
    torch.testing.assert_close(relative, expected_relative)
    torch.testing.assert_close(
        to_absolute_actions(relative, state, mask, state_action_index_map),
        actions,
    )


def test_relative_and_absolute_processor_steps_share_explicit_state_action_index_map():
    state = torch.tensor([[10.0, 0.1, 20.0, 0.2, 30.0]])
    action = torch.tensor([[11.0, 22.0, 33.0]])
    relative_step = RelativeActionsProcessorStep(
        enabled=True,
        state_action_index_map=[0, 2, 4],
    )
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)

    relative_transition = relative_step(create_transition(observation={OBS_STATE: state}, action=action))
    torch.testing.assert_close(relative_transition[TransitionKey.ACTION], torch.tensor([[1.0, 2.0, 3.0]]))

    absolute_transition = absolute_step(create_transition(action=relative_transition[TransitionKey.ACTION]))
    torch.testing.assert_close(absolute_transition[TransitionKey.ACTION], action)


@pytest.mark.parametrize("state_action_index_map", ([0], [0, -1], [0, 5]))
def test_relative_actions_reject_invalid_state_action_index_map(state_action_index_map):
    state = torch.tensor([[10.0, 20.0]])
    actions = torch.tensor([[11.0, 22.0]])
    mask = [True, True]

    with pytest.raises(ValueError, match="state_action_index_map"):
        to_relative_actions(actions, state, mask, state_action_index_map)


def test_compute_relative_action_stats_uses_explicit_state_action_index_map():
    compute_stats = pytest.importorskip("lerobot.datasets.compute_stats", exc_type=ImportError)
    hf_dataset = {
        ACTION: np.array(
            [
                [11.0, 22.0, 7.0],
                [12.0, 24.0, 8.0],
                [13.0, 26.0, 9.0],
            ],
            dtype=np.float32,
        ),
        OBS_STATE: np.array(
            [
                [10.0, 0.1, 20.0, 0.2, 99.0],
                [11.0, 0.1, 22.0, 0.2, 99.0],
                [12.0, 0.1, 24.0, 0.2, 99.0],
            ],
            dtype=np.float32,
        ),
        "episode_index": np.array([0, 0, 0]),
    }
    features = {
        ACTION: {
            "shape": (3,),
            "names": ["joint_0.pos", "joint_1.pos", "gripper"],
        }
    }

    stats = compute_stats.compute_relative_action_stats(
        hf_dataset=hf_dataset,
        features=features,
        chunk_size=2,
        exclude_joints=["gripper"],
        state_action_index_map=[0, 2, 4],
        num_workers=0,
    )

    np.testing.assert_allclose(stats["mean"], np.array([1.5, 3.0, 8.0]), rtol=1e-6)
