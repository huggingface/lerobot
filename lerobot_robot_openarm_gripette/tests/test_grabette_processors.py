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

from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.lerobot_types import TransitionKey
from lerobot_robot_openarm_gripette.grabette_processors import (
    GrabetteRelativePoseToJoints,
    GripettePolicyObservation,
)


def _observation():
    return {
        **{f"joint_{i}.pos": float(i) for i in range(1, 8)},
        "proximal.pos": 0.8,
        "distal.pos": 0.9,
        "camera": np.zeros((4, 4, 3), dtype=np.uint8),
    }


def _action(x=0.0):
    return {
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "r6d_0": 1.0,
        "r6d_1": 0.0,
        "r6d_2": 0.0,
        "r6d_3": 0.0,
        "r6d_4": 1.0,
        "r6d_5": 0.0,
        "strategy": 0.25,
        "closure": 0.5,
    }


def test_gripette_observation_encodes_policy_state_and_preserves_camera():
    processed = GripettePolicyObservation()({TransitionKey.OBSERVATION: _observation()})[
        TransitionKey.OBSERVATION
    ]

    assert set(processed) == {"camera", "strategy", "closure"}
    np.testing.assert_array_equal(processed["camera"], np.zeros((4, 4, 3), dtype=np.uint8))
    assert processed["strategy"] == pytest.approx(0.3)
    assert processed["closure"] == pytest.approx(0.4)


def test_relative_pose_processor_holds_anchor_for_chunk_and_decodes_gripper():
    kinematics = MagicMock()
    kinematics.forward_kinematics.return_value = np.eye(4)
    kinematics.inverse_kinematics.side_effect = lambda q, *_args, **_kwargs: q + 1.0
    depths = iter([2, 1, 0, 2])
    processor = GrabetteRelativePoseToJoints(
        kinematics=kinematics,
        joint_signs=(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    )
    processor.bind_action_queue(lambda: next(depths), n_action_steps=3)

    outputs = []
    for x in (0.1, 0.2, 0.3, 0.4):
        transition = processor(
            {
                TransitionKey.OBSERVATION: _observation(),
                TransitionKey.ACTION: _action(x),
            }
        )
        outputs.append(transition[TransitionKey.ACTION])

    assert kinematics.forward_kinematics.call_count == 2
    assert outputs[0]["joint_1.pos"] == 0.0
    assert outputs[0]["joint_2.pos"] == 3.0
    assert outputs[0]["proximal.pos"] == 1.0
    assert outputs[0]["distal.pos"] == 0.75


def test_relative_pose_processor_rejects_unsafe_ik_jump():
    kinematics = MagicMock()
    kinematics.forward_kinematics.return_value = np.eye(4)
    kinematics.inverse_kinematics.side_effect = lambda q, *_args, **_kwargs: q + 6.0
    processor = GrabetteRelativePoseToJoints(kinematics=kinematics)

    with pytest.raises(RuntimeError, match="joint_1 by 6.0 deg"):
        processor(
            {
                TransitionKey.OBSERVATION: _observation(),
                TransitionKey.ACTION: _action(),
            }
        )
