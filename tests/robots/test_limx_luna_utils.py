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

"""Pure config and joint-layout tests for the LimX Luna robot.

Nothing here needs the ``limxsdk`` extra: these cover the registry entry, config validation and the
joint vector layout. Tests that drive the robot class live in ``test_limx_luna.py``.
"""

import numpy as np
import pytest

from lerobot.robots.config import RobotConfig
from lerobot.robots.limx_luna import LimxLuna, LimxLunaConfig
from lerobot.robots.limx_luna.joints import (
    DEFAULT_JOINT_NAMES,
    LUNA_DIM,
    joint_vector,
    pos_key,
    resolve_joint_names,
    resolve_joint_vector,
)

#: Placeholder gains for constructing configs in tests; not suitable for a real robot.
TEST_KP = [10.0] * LUNA_DIM
TEST_KD = [1.0] * LUNA_DIM


def make_config(**overrides) -> LimxLunaConfig:
    base = {"robot_ip": "127.0.0.1", "kp": TEST_KP, "kd": TEST_KD}
    base.update(overrides)
    return LimxLunaConfig(**base)


# ------------------------------------------------------------------- registry


def test_registered_with_lerobot():
    """The class must be discoverable, or --robot.type=limx_luna fails at parse time."""
    assert "limx_luna" in RobotConfig.get_known_choices()
    assert RobotConfig.get_choice_class("limx_luna") is LimxLunaConfig


def test_robot_declares_its_config_class():
    assert LimxLuna.config_class is LimxLunaConfig
    assert LimxLuna.name == "limx_luna"


# --------------------------------------------------------------------- config


def test_default_joint_names_match_the_luna_width():
    config = make_config()
    assert len(config.joint_names) == LUNA_DIM == 27


def test_default_joint_names_cover_legs_waist_head_and_arms():
    """The vector order is legs, waist, head, arms, matching the vendor controller and the URDF."""
    names = list(DEFAULT_JOINT_NAMES)
    assert names[:6] == [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ]
    assert names[12:17] == [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "head_yaw_joint",
        "head_pitch_joint",
    ]
    assert names[17:22] == [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_yaw_joint",
    ]
    assert names[22] == "right_shoulder_pitch_joint"
    assert names[-1] == "right_wrist_yaw_joint"
    assert not any("wrist_pitch" in name or "wrist_roll" in name for name in names)


def test_bad_joint_names_fail_at_construction():
    with pytest.raises(ValueError, match="joint_names must have 27"):
        make_config(joint_names=["a", "b"])


def test_duplicate_joint_names_rejected():
    names = list(make_config().joint_names)
    names[1] = names[0]
    with pytest.raises(ValueError, match="duplicates"):
        make_config(joint_names=names)


def test_gain_tables_are_validated():
    assert len(make_config().kp) == LUNA_DIM
    assert len(make_config().kd) == LUNA_DIM
    with pytest.raises(ValueError, match="kp must have 27"):
        make_config(kp=[1.0] * 3)
    with pytest.raises(ValueError, match="kd must have 27"):
        make_config(kd=[1.0] * 7)
    with pytest.raises(ValueError, match="non-negative"):
        make_config(kp=[-1.0] * LUNA_DIM)


@pytest.mark.parametrize("missing", ["kp", "kd"])
def test_gains_are_required(missing):
    """Luna has no vendor-published standing gains, so the config must not invent any."""
    overrides = {"robot_ip": "127.0.0.1", "kp": TEST_KP, "kd": TEST_KD}
    del overrides[missing]
    with pytest.raises(ValueError, match="no default gains"):
        LimxLunaConfig(**overrides)


def test_bad_start_pose_rejected():
    with pytest.raises(ValueError, match="start_pose must have 27"):
        make_config(start_pose=[0.0] * 12)


def test_bad_rates_rejected():
    with pytest.raises(ValueError, match="publish_rate must be positive"):
        make_config(publish_rate=0.0)
    with pytest.raises(ValueError, match="observation_timeout must be positive"):
        make_config(observation_timeout=0.0)
    with pytest.raises(ValueError, match="state_max_age"):
        make_config(state_max_age=-1.0)


def test_robot_type_defaults_to_leaving_the_sdk_alone():
    """`robot_type` is opt-in: when unset the driver must not set the environment variable."""
    assert make_config().robot_type is None


# -------------------------------------------------------------- joint vectors


def test_resolve_joint_names_falls_back_to_the_default():
    assert resolve_joint_names(None) == DEFAULT_JOINT_NAMES


def test_resolve_joint_vector_fills_neutral_values():
    assert resolve_joint_vector(None, "kp") == (0.0,) * LUNA_DIM


def test_pos_key_appends_the_suffix():
    assert pos_key("left_elbow_joint") == "left_elbow_joint.pos"


def test_joint_vector_follows_the_joint_name_order():
    names = make_config().joint_names
    action = {pos_key(name): float(index) for index, name in enumerate(names)}
    np.testing.assert_allclose(joint_vector(action, names), np.arange(LUNA_DIM))


def test_joint_vector_reports_missing_keys():
    with pytest.raises(KeyError, match="missing joint keys"):
        joint_vector({pos_key("left_hip_pitch_joint"): 0.0}, make_config().joint_names)


def test_joint_vector_rejects_non_finite_targets():
    action = {pos_key(name): 0.0 for name in make_config().joint_names}
    action[pos_key("waist_yaw_joint")] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        joint_vector(action, make_config().joint_names)
