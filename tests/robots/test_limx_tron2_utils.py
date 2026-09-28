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

"""Pure config and joint-layout tests for the LimX TRON2 robot.

Nothing here needs the ``tron2-env`` extra: these cover the registry entry, config validation and
the two vector layouts. Tests that drive the motion controller live in ``test_limx_tron2.py``.
"""

import numpy as np
import pytest

from lerobot.robots.config import RobotConfig
from lerobot.robots.limx_tron2 import LimxTron2, LimxTron2Config
from lerobot.robots.limx_tron2.joints import (
    SERVOJ_DIM,
    STATE_DIM,
    gripper_openings,
    servoj_to_state,
    servoj_vector,
    split_state,
    state_to_servoj,
)


def make_config(**overrides) -> LimxTron2Config:
    base = {"robot_ip": "127.0.0.1"}
    base.update(overrides)
    return LimxTron2Config(**base)


# ------------------------------------------------------------------- registry


def test_registered_with_lerobot():
    """The class must be discoverable, or --robot.type=limx_tron2 fails at parse time."""
    assert "limx_tron2" in RobotConfig.get_known_choices()
    assert RobotConfig.get_choice_class("limx_tron2") is LimxTron2Config


def test_robot_declares_its_config_class():
    assert LimxTron2.config_class is LimxTron2Config
    assert LimxTron2.name == "limx_tron2"


# --------------------------------------------------------------------- config


def test_default_joint_names_match_the_servoj_width():
    config = make_config()
    assert len(config.joint_names) == SERVOJ_DIM == 16
    assert config.gripper_names == ["left_gripper", "right_gripper"]


def test_bad_joint_names_fail_at_construction():
    with pytest.raises(ValueError, match="joint_names must have 16"):
        make_config(joint_names=["a", "b"])


def test_duplicate_joint_names_rejected():
    names = list(make_config().joint_names)
    names[1] = names[0]
    with pytest.raises(ValueError, match="duplicates"):
        make_config(joint_names=names)


def test_bad_bringup_pose_rejected():
    with pytest.raises(ValueError, match="init_joints must have 14"):
        make_config(init_joints=[0.0] * 7)
    with pytest.raises(ValueError, match="init_head must have 2"):
        make_config(init_head=[0.0] * 3)


def test_bad_rates_rejected():
    with pytest.raises(ValueError, match="publish_rate must be positive"):
        make_config(publish_rate=0.0)
    with pytest.raises(ValueError, match="observation_timeout must be positive"):
        make_config(observation_timeout=0.0)
    with pytest.raises(ValueError, match="state_max_age"):
        make_config(state_max_age=-1.0)


def test_grippers_are_opt_in():
    """Grippers stay out of the action space unless the units have been verified on hardware."""
    robot = LimxTron2(make_config())
    assert "left_gripper.pos" not in robot.observation_features

    robot_with = LimxTron2(make_config(include_grippers=True))
    assert "left_gripper.pos" in robot_with.observation_features
    assert "left_gripper.pos" in robot_with.action_features


# ------------------------------------------------------------- vector layouts


def test_state_to_servoj_drops_the_grippers():
    state = np.arange(STATE_DIM, dtype=float)
    servoj = state_to_servoj(state)
    # state = [L_arm(0:7), L_grip(7), R_arm(8:15), R_grip(15), head(16:18)]
    np.testing.assert_allclose(servoj, np.concatenate((state[0:7], state[8:15], state[16:18])))
    assert servoj.shape == (SERVOJ_DIM,)


def test_servoj_state_round_trip_preserves_arm_and_head():
    state = np.arange(STATE_DIM, dtype=float)
    servoj = state_to_servoj(state)
    left, _, right, _, head = split_state(state)
    rebuilt = servoj_to_state(servoj, left_gripper=0.0, right_gripper=0.0)
    np.testing.assert_allclose(state_to_servoj(rebuilt), servoj)
    np.testing.assert_allclose(rebuilt[0:7], left)
    np.testing.assert_allclose(rebuilt[8:15], right)
    np.testing.assert_allclose(rebuilt[16:18], head)


def test_state_validation_rejects_wrong_width():
    with pytest.raises(ValueError, match="expected a 18-value state vector"):
        split_state([0.0] * 5)


def test_servoj_vector_follows_the_joint_name_order():
    names = make_config().joint_names
    action = {f"{name}.pos": float(index) for index, name in enumerate(names)}
    np.testing.assert_allclose(servoj_vector(action, names), np.arange(SERVOJ_DIM))


def test_servoj_vector_reports_missing_keys():
    with pytest.raises(KeyError, match="missing joint keys"):
        servoj_vector({"left_arm_shoulder_pitch.pos": 0.0}, make_config().joint_names)


def test_gripper_openings_requires_both_sides():
    names = make_config().gripper_names
    assert gripper_openings({}, names) is None
    pair = {"left_gripper.pos": 0.25, "right_gripper.pos": 0.75}
    assert gripper_openings(pair, names) == (0.25, 0.75)
    with pytest.raises(KeyError, match="only one side"):
        gripper_openings({"left_gripper.pos": 0.25}, names)
