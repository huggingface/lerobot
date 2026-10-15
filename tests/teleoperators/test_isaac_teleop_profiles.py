#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Two ``RobotProfile`` facts that fail silently rather than loudly.

Everything else about the twin announces itself: an unknown ``preview_arm`` raises out of
``build_twin``, and a broken scene raises out of the compiler. These two do not.
"""

import numpy as np
import pytest

from lerobot.teleoperators.isaac_teleop.robot_profiles import ROBOT_PROFILES, RobotProfile


@pytest.mark.parametrize("key", sorted(ROBOT_PROFILES))
def test_reset_pose_covers_every_motor(key):
    """``_step_reset`` builds its target with ``reset_pose.get(name, <where the arm is>)``
    over ``motor_names``, so a missing entry leaves that joint wherever it was instead of
    parking it -- a declutch that silently half-moves."""
    profile: RobotProfile = ROBOT_PROFILES[key]
    assert set(profile.reset_pose) == set(profile.motor_names)


@pytest.mark.parametrize("key", sorted(ROBOT_PROFILES))
def test_reset_pose_is_the_preview_arms_home_pose(key):
    """The preview holds the pose the hardware parks at, so the operator sees the arm hold
    what it will hold. Read from isaacteleop rather than copied, so a q_home tuned upstream
    fails here too -- that is the direction this last drifted, and it is invisible until
    somebody is wearing a headset.
    """
    profile: RobotProfile = ROBOT_PROFILES[key]
    if profile.preview_arm is None:
        pytest.skip("no twin, so nothing to agree with")
    viz = pytest.importorskip("isaacteleop.viz.robot", reason="needs isaacteleop with the robot twin")
    preview = viz.PREVIEW_ARMS[profile.preview_arm]
    # The preview names its own joints (jointN on the reBot) and skips the gripper where it
    # is not a hinge; zip pairs them onto motor_names in the positional order
    # RobotProfile.urdf_joint_names already documents.
    mismatched = {
        motor: (profile.reset_pose[motor], round(float(want), 4))
        for motor, want in zip(profile.motor_names, np.degrees(preview.q_home), strict=False)
        if profile.reset_pose[motor] != pytest.approx(want)
    }
    assert not mismatched, (
        f"{key}: reset_pose disagrees with the {profile.preview_arm} preview's q_home at "
        f"{mismatched} (reset_pose, q_home); the operator would be shown a pose the arm "
        "will not hold"
    )
