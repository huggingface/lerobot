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

from lerobot.robots.nero_follower.config_nero_follower import NEOKeyboardEEConfig
from lerobot.robots.nero_follower.robot_kinematic_processor import NEROKeyboardEEToJoints


class FakeKinematics:
    def __init__(self):
        self.last_desired_pose = None

    def forward_kinematics(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = np.array([0.0, 0.0, 0.3], dtype=float)
        return pose

    def inverse_kinematics(self, current_joint_pos_rad: np.ndarray, desired_ee_pose: np.ndarray) -> np.ndarray:
        self.last_desired_pose = desired_ee_pose.copy()
        return current_joint_pos_rad + 0.01


def _make_observation() -> dict[str, float]:
    obs = {f"joint{i}.pos": i * 0.1 for i in range(1, 8)}
    obs["gripper.pos"] = 20.0
    return obs


def test_deadman_disabled_returns_empty_action():
    step = NEROKeyboardEEToJoints(
        kinematics=FakeKinematics(),
        joint_names=[f"joint{i}" for i in range(1, 8)],
        config=NEOKeyboardEEConfig(),
    )

    out = step(
        {
            "observation": _make_observation(),
            "action": {
                "enabled": 0.0,
                "target_x": 0.2,
                "target_y": 0.0,
                "target_z": 0.0,
                "target_wx": 0.0,
                "target_wy": 0.0,
                "target_wz": 0.0,
                "gripper_vel": 1.0,
            },
        }
    )["action"]

    assert out == {}


def test_enabled_action_runs_ik_and_gripper_update():
    cfg = NEOKeyboardEEConfig(max_linear_step_m=0.01, max_angular_step_rad=0.2, gripper_delta_per_step=2.0)
    kin = FakeKinematics()
    step = NEROKeyboardEEToJoints(
        kinematics=kin,
        joint_names=[f"joint{i}" for i in range(1, 8)],
        config=cfg,
    )
    out = step(
        {
            "observation": _make_observation(),
            "action": {
                "enabled": 1.0,
                "target_x": 0.5,
                "target_y": 0.0,
                "target_z": 0.0,
                "target_wx": 0.0,
                "target_wy": 0.0,
                "target_wz": 0.0,
                "gripper_vel": 1.0,
            },
        }
    )["action"]

    assert all(f"joint{i}.pos" in out for i in range(1, 8))
    assert out["gripper.pos"] == 22.0
    assert kin.last_desired_pose is not None
    assert float(kin.last_desired_pose[0, 3]) == 0.01
