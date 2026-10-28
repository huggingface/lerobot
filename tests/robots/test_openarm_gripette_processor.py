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

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.lerobot_types import TransitionKey
from lerobot.robots.openarm_gripette_follower.robot_grabette_processor import (
    GrabetteRelativePoseToJoints,
    GripettePolicyObservation,
)


class FakeGraspProjection:
    def encode(self, proximal: float, distal: float) -> tuple[float, float]:
        return distal / 3.0, proximal / 2.0

    def decode(self, strategy: float, closure: float) -> tuple[float, float]:
        return closure * 2.0, strategy * 3.0


def _observation():
    return {
        **{f"joint_{i}.pos": float(i) for i in range(1, 8)},
        "proximal.pos": 0.8,
        "distal.pos": 0.9,
        "camera": np.zeros((4, 4, 3), dtype=np.uint8),
    }


def _action(x=0.0):
    rot6d = dict(zip([f"r6d_{i}" for i in range(6)], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], strict=True))
    return {"x": x, "y": 0.0, "z": 0.0, **rot6d, "strategy": 0.25, "closure": 0.5}


def _kinematics(ik_step: float) -> MagicMock:
    kinematics = MagicMock()
    kinematics.forward_kinematics.return_value = np.eye(4)
    kinematics.inverse_kinematics.side_effect = lambda q, *_args, **_kwargs: q + ik_step
    return kinematics


def test_gripette_observation_encodes_policy_state_and_preserves_camera():
    step = GripettePolicyObservation(projection=FakeGraspProjection())
    processed = step({TransitionKey.OBSERVATION: _observation()})[TransitionKey.OBSERVATION]

    assert set(processed) == {"camera", "strategy", "closure"}
    assert processed["strategy"] == pytest.approx(0.3)
    assert processed["closure"] == pytest.approx(0.4)


def test_relative_pose_holds_anchor_for_chunk_and_decodes_gripper(caplog):
    caplog.set_level(logging.INFO)
    kinematics = _kinematics(ik_step=1.0)
    depths = iter([2, 1, 0, 2])
    step = GrabetteRelativePoseToJoints(
        kinematics=kinematics,
        joint_signs=(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        projection=FakeGraspProjection(),
    )
    step.bind_action_queue(lambda: next(depths), n_action_steps=3)

    outputs = [
        step({TransitionKey.OBSERVATION: _observation(), TransitionKey.ACTION: _action(x)})[
            TransitionKey.ACTION
        ]
        for x in (0.1, 0.2, 0.3, 0.4)
    ]

    assert kinematics.forward_kinematics.call_count == 2
    assert outputs[0]["joint_1.pos"] == 0.0
    assert outputs[0]["joint_2.pos"] == 3.0
    assert (outputs[0]["proximal.pos"], outputs[0]["distal.pos"]) == (1.0, 0.75)
    assert any("X=+0.1000 m" in message for message in caplog.messages)


def test_relative_pose_rejects_unsafe_ik_jump():
    step = GrabetteRelativePoseToJoints(kinematics=_kinematics(ik_step=6.0), projection=FakeGraspProjection())

    with pytest.raises(RuntimeError, match="joint_1 by 6.0 deg"):
        step({TransitionKey.OBSERVATION: _observation(), TransitionKey.ACTION: _action()})
