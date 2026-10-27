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

"""Exoskeleton-to-Asimov retargeting. Needs pinocchio and downloads the robot URDFs once."""

from dataclasses import dataclass

import draccus
import numpy as np
import pytest

from lerobot.robots.asimov.asimov import ARM_ACTIONS
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.asimov_exo import AsimovExoTeleoperatorConfig

EXO_ZERO = dict.fromkeys(("shoulder_pitch", "shoulder_yaw", "shoulder_roll", "elbow_flex", "wrist_roll"), 0.0)


@pytest.fixture(scope="module")
def ik():
    pytest.importorskip("pinocchio")
    from lerobot.robots.asimov.asimov_kinematics import AsimovArmIK

    return AsimovArmIK()


@pytest.fixture(scope="module")
def helper():
    pytest.importorskip("pinocchio")
    from lerobot.teleoperators.asimov_exo import AsimovExoIKHelper

    return AsimovExoIKHelper()


def test_cli_config():
    @dataclass
    class Config:
        teleop: TeleoperatorConfig

    config = draccus.parse(
        Config,
        args=[
            "--teleop.type=asimov_exo",
            "--teleop.left_arm_config.port=/dev/ttyACM0",
            "--teleop.right_arm_config.port=/dev/ttyACM1",
        ],
    )
    assert isinstance(config.teleop, AsimovExoTeleoperatorConfig)


def test_ik_tracks_reachable_arm_motion(ik):
    rng = np.random.default_rng(0)
    idx = ik.arm_idx["left"][0] + ik.arm_idx["right"][0]
    lo, hi = ik.lower[idx], ik.upper[idx]
    phase, freq = rng.random(10) * 6, 0.2 + rng.random(10) * 0.5
    ik.q[:] = np.clip(np.zeros(ik.model.nq), ik.lower, ik.upper)
    for k in range(150):
        reference = ik.q.copy()
        reference[idx] = (lo + hi) / 2 + 0.45 * (hi - lo) * np.sin(2 * np.pi * freq * k / 30 + phase)
        solved = ik.q.copy()
        ik.q[:] = reference
        targets = {side: ik.ee_pose(side) for side in ("left", "right")}
        ik.q[:] = solved
        ik.solve(targets)
        for side, target in targets.items():
            assert np.linalg.norm(ik.ee_pose(side)[:3, 3] - target[:3, 3]) < 5e-3


def test_ik_respects_limits_and_frozen_joints(ik):
    ik.set_arm_positions({"L_Wrist_Yaw": 1.0})
    unreachable = np.eye(4)
    unreachable[:3, 3] = [2.0, 2.0, 2.0]
    q = ik.solve({"left": unreachable, "right": None}, frozen={"L_Wrist_Yaw"})
    assert np.all(q >= ik.lower - 1e-9) and np.all(q <= ik.upper + 1e-9)
    assert ik.arm_positions()["L_Wrist_Yaw"] == pytest.approx(1.0)


def test_exo_rest_pose_maps_to_forward_forearms(helper):
    for _ in range(5):
        action = helper.compute_asimov_joints_from_exo(EXO_ZERO, EXO_ZERO)
    assert tuple(action) == ARM_ACTIONS
    # Exo zero is the G1 zero: elbows bent ~90°, forearms forward, mirrored left/right.
    assert action["L_Elbow.pos"] == pytest.approx(np.pi / 2, abs=0.2)
    assert action["R_Elbow.pos"] == pytest.approx(-action["L_Elbow.pos"], abs=1e-3)
    for side in ("left", "right"):
        reached = helper.ik.ee_pose(side)[:3, 3]
        assert np.linalg.norm(reached - helper.targets[side][:3, 3]) < 0.01


def test_exo_motion_moves_matching_asimov_joint(helper):
    for _ in range(5):
        rest = helper.compute_asimov_joints_from_exo(EXO_ZERO, EXO_ZERO)
    for _ in range(5):
        raised = helper.compute_asimov_joints_from_exo({**EXO_ZERO, "shoulder_pitch": 0.5}, EXO_ZERO)
    # Raising the exo arm forward raises Asimov's left arm (negative left shoulder pitch).
    assert raised["L_Shoulder_Pitch.pos"] < rest["L_Shoulder_Pitch.pos"] - 0.3
    assert raised["R_Shoulder_Pitch.pos"] == pytest.approx(rest["R_Shoulder_Pitch.pos"], abs=1e-3)
