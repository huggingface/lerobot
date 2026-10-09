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

"""Pure config and joint-layout tests for the LimX TRON1 legged robots.

Nothing here needs the ``limxsdk`` extra: these cover the registry entry, config validation and the
joint vector layout for the three build families. Tests that drive the robot class live in
``test_limx_tron1.py``.
"""

import numpy as np
import pytest

from lerobot.robots.config import RobotConfig
from lerobot.robots.limx_tron1 import LimxTron1, LimxTron1Config
from lerobot.robots.limx_tron1.joints import (
    POINTFOOT,
    SOLEFOOT,
    VARIANT_DIMS,
    VARIANT_JOINT_NAMES,
    VARIANT_KD,
    VARIANT_KP,
    WHEELFOOT,
    joint_vector,
    pos_key,
    resolve_joint_names,
    resolve_joint_vector,
    variant_from_robot_type,
)


def make_config(**overrides) -> LimxTron1Config:
    base = {"robot_ip": "127.0.0.1"}
    base.update(overrides)
    return LimxTron1Config(**base)


# ------------------------------------------------------------------- registry


def test_registered_with_lerobot():
    """The class must be discoverable, or --robot.type=limx_tron1 fails at parse time."""
    assert "limx_tron1" in RobotConfig.get_known_choices()
    assert RobotConfig.get_choice_class("limx_tron1") is LimxTron1Config


def test_robot_declares_its_config_class():
    assert LimxTron1.config_class is LimxTron1Config
    assert LimxTron1.name == "limx_tron1"


# ------------------------------------------------------------- build families


def test_robot_type_prefix_selects_the_family():
    assert variant_from_robot_type(None) == POINTFOOT
    assert variant_from_robot_type("PF_TRON1A") == POINTFOOT
    assert variant_from_robot_type("PF_P441C") == POINTFOOT
    assert variant_from_robot_type("SF_TRON1A") == SOLEFOOT
    assert variant_from_robot_type("WF_TRON1A") == WHEELFOOT


def test_robot_type_prefix_is_case_insensitive():
    assert variant_from_robot_type("sf_tron1b") == SOLEFOOT
    assert variant_from_robot_type("WF_TRON1B") == WHEELFOOT


def test_unknown_robot_type_is_rejected():
    with pytest.raises(ValueError, match="unknown TRON1 build"):
        variant_from_robot_type("WL_P311D")


def test_family_widths():
    assert VARIANT_DIMS == {POINTFOOT: 6, SOLEFOOT: 8, WHEELFOOT: 8}


def test_joint_tables_match_the_vendor_header():
    """The SDK header (pointfoot.h) fixes the order: left leg first, then right leg."""
    assert list(VARIANT_JOINT_NAMES[POINTFOOT]) == [
        "abad_L_Joint",
        "hip_L_Joint",
        "knee_L_Joint",
        "abad_R_Joint",
        "hip_R_Joint",
        "knee_R_Joint",
    ]
    assert list(VARIANT_JOINT_NAMES[SOLEFOOT]) == [
        "abad_L_Joint",
        "hip_L_Joint",
        "knee_L_Joint",
        "ankle_L_Joint",
        "abad_R_Joint",
        "hip_R_Joint",
        "knee_R_Joint",
        "ankle_R_Joint",
    ]
    assert list(VARIANT_JOINT_NAMES[WHEELFOOT]) == [
        "abad_L_Joint",
        "hip_L_Joint",
        "knee_L_Joint",
        "wheel_L_Joint",
        "abad_R_Joint",
        "hip_R_Joint",
        "knee_R_Joint",
        "wheel_R_Joint",
    ]


def test_gain_defaults_match_the_vendor_params():
    # point foot: uniform 42 / 3.5.
    assert VARIANT_KP[POINTFOOT] == (42.0,) * 6
    assert VARIANT_KD[POINTFOOT] == (3.5,) * 6
    # sole foot: stiffness 45, damping 3.0 with a lower ankle damping of 1.5.
    assert VARIANT_KP[SOLEFOOT] == (45.0,) * 8
    assert VARIANT_KD[SOLEFOOT] == (3.0, 3.0, 3.0, 1.5, 3.0, 3.0, 3.0, 1.5)
    # wheel foot: stiffness 42, damping 2.5 with a lower wheel damping of 0.8.
    assert VARIANT_KP[WHEELFOOT] == (42.0,) * 8
    assert VARIANT_KD[WHEELFOOT] == (2.5, 2.5, 2.5, 0.8, 2.5, 2.5, 2.5, 0.8)


# --------------------------------------------------------------------- config


def test_default_config_is_the_point_foot():
    config = make_config()
    assert config.variant == POINTFOOT
    assert len(config.joint_names) == 6


def test_config_tracks_the_family_through_robot_type():
    assert make_config(robot_type="PF_TRON1A").variant == POINTFOOT
    assert len(make_config(robot_type="SF_TRON1A").joint_names) == 8
    assert make_config(robot_type="WF_TRON1A").variant == WHEELFOOT


def test_config_resolves_gains_for_the_family():
    solefoot = make_config(robot_type="SF_TRON1A")
    assert solefoot.kp == list(VARIANT_KP[SOLEFOOT])
    assert solefoot.kd == list(VARIANT_KD[SOLEFOOT])
    wheelfoot = make_config(robot_type="WF_TRON1A")
    assert wheelfoot.kp == list(VARIANT_KP[WHEELFOOT])
    assert wheelfoot.kd == list(VARIANT_KD[WHEELFOOT])


def test_explicit_gains_are_validated_against_the_family_width():
    with pytest.raises(ValueError, match="kp must have 8"):
        make_config(robot_type="SF_TRON1A", kp=[1.0] * 3)
    with pytest.raises(ValueError, match="kd must have 6"):
        make_config(robot_type="PF_TRON1A", kd=[1.0] * 7)


def test_bad_joint_names_fail_at_construction():
    with pytest.raises(ValueError, match="joint_names must have 6"):
        make_config(joint_names=["a", "b"])


def test_duplicate_joint_names_rejected():
    names = list(make_config().joint_names)
    names[1] = names[0]
    with pytest.raises(ValueError, match="duplicates"):
        make_config(joint_names=names)


def test_bad_start_pose_rejected():
    with pytest.raises(ValueError, match="start_pose must have 6"):
        make_config(start_pose=[0.0] * 5)


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


def test_resolve_joint_names_falls_back_to_the_family_default():
    assert resolve_joint_names(None, POINTFOOT) == VARIANT_JOINT_NAMES[POINTFOOT]
    assert resolve_joint_names(None, SOLEFOOT) == VARIANT_JOINT_NAMES[SOLEFOOT]


def test_resolve_joint_vector_fills_neutral_values():
    assert resolve_joint_vector(None, 6, "kp") == (0.0,) * 6
    assert resolve_joint_vector(None, 8, "kp") == (0.0,) * 8


def test_pos_key_appends_the_suffix():
    assert pos_key("hip_L_Joint") == "hip_L_Joint.pos"


def test_joint_vector_follows_the_joint_name_order():
    names = make_config(robot_type="SF_TRON1A").joint_names
    action = {pos_key(name): float(index) for index, name in enumerate(names)}
    np.testing.assert_allclose(joint_vector(action, names), np.arange(8))


def test_joint_vector_reports_missing_keys():
    with pytest.raises(KeyError, match="missing joint keys"):
        joint_vector({pos_key("abad_L_Joint"): 0.0}, make_config().joint_names)


def test_joint_vector_rejects_non_finite_targets():
    action = {pos_key(name): 0.0 for name in make_config().joint_names}
    action[pos_key("hip_L_Joint")] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        joint_vector(action, make_config().joint_names)
