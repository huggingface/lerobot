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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.robots.reachy_mini.configuration_reachy_mini import ReachyMiniConfig
from lerobot.robots.reachy_mini.robot_reachy_mini import ReachyMini

PARAMS = [
    {},  # default config
    {"ip_address": "192.168.1.42"},
]


def _make_reachy_mini_sdk_mock():
    sdk_mock = MagicMock(name="ReachyMiniSDKMock")

    # Mock joint positions
    # head_joints: [body_yaw, stewart_1...6]
    head_joints = [np.deg2rad(10.0), np.deg2rad(5.0), np.deg2rad(-5.0), 0.0, 0.0, 0.0, 0.0]
    # antenna_joints: [right, left]
    antenna_joints = [np.deg2rad(20.0), np.deg2rad(-20.0)]

    sdk_mock.get_current_joint_positions.return_value = (head_joints, antenna_joints)

    # Mock individual joint objects for fallback path
    sdk_mock.body_yaw.present_position = np.deg2rad(10.0)
    sdk_mock.antennas.left.present_position = np.deg2rad(-20.0)
    sdk_mock.antennas.right.present_position = np.deg2rad(20.0)

    sdk_mock._set_joint_positions = MagicMock()
    sdk_mock.__exit__ = MagicMock()

    return sdk_mock


def _make_camera_mock(*args, **kwargs):
    cfg = args[0] if args else kwargs.get("config")
    name = getattr(cfg, "name", kwargs.get("name", "cam"))
    width = getattr(cfg, "width", kwargs.get("width", 640))
    height = getattr(cfg, "height", kwargs.get("height", 480))

    cam = MagicMock(name=f"CameraMock:{name}")
    cam.name = name
    cam.width = width
    cam.height = height
    cam.is_connected = False

    def _connect():
        cam.is_connected = True

    def _disconnect():
        cam.is_connected = False

    cam.connect = MagicMock(side_effect=_connect)
    cam.disconnect = MagicMock(side_effect=_disconnect)
    cam.async_read = MagicMock(side_effect=lambda: np.zeros((height, width, 3), dtype=np.uint8))
    return cam


@pytest.fixture(params=PARAMS, ids=lambda p: "default" if not p else ",".join(p.keys()))
def reachy_mini(request):
    with (
        patch(
            "lerobot.robots.reachy_mini.robot_reachy_mini.ReachyMiniSDK",
            side_effect=lambda *a, **k: _make_reachy_mini_sdk_mock(),
        ),
        patch(
            "lerobot.robots.reachy_mini.robot_reachy_mini.make_cameras_from_configs",
            side_effect=lambda cfgs: {k: _make_camera_mock(v) for k, v in cfgs.items()},
        ),
    ):
        overrides = request.param
        cfg = ReachyMiniConfig(**overrides)
        robot = ReachyMini(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(reachy_mini):
    assert not reachy_mini.is_connected

    reachy_mini.connect()
    assert reachy_mini.is_connected

    robot_mock = reachy_mini.robot
    reachy_mini.disconnect()
    assert not reachy_mini.is_connected
    robot_mock.__exit__.assert_called_once()


def test_get_observation(reachy_mini):
    reachy_mini.connect()
    obs = reachy_mini.get_observation()

    expected_keys = {
        "body_rotation.pos",
        "stewart_1.pos",
        "stewart_2.pos",
        "stewart_3.pos",
        "stewart_4.pos",
        "stewart_5.pos",
        "stewart_6.pos",
        "right_antenna.pos",
        "left_antenna.pos",
        *reachy_mini.cameras.keys(),
    }
    assert set(obs.keys()) == expected_keys

    # Check values from the mock
    assert obs["body_rotation.pos"] == pytest.approx(10.0)
    assert obs["stewart_1.pos"] == pytest.approx(5.0)
    assert obs["stewart_2.pos"] == pytest.approx(-5.0)
    assert obs["right_antenna.pos"] == pytest.approx(20.0)
    assert obs["left_antenna.pos"] == pytest.approx(-20.0)

    cam_key = list(reachy_mini.cameras.keys())[0]
    cam_cfg = reachy_mini.config.cameras[cam_key]
    assert obs[cam_key].shape == (cam_cfg.height, cam_cfg.width, 3)


def test_send_action(reachy_mini):
    reachy_mini.connect()

    action = {
        "body_rotation.pos": -45.0,
        "stewart_1.pos": 10.0,
        "stewart_2.pos": 10.0,
        "stewart_3.pos": 10.0,
        "stewart_4.pos": 10.0,
        "stewart_5.pos": 10.0,
        "stewart_6.pos": 10.0,
        "right_antenna.pos": 60.0,
        "left_antenna.pos": -60.0,
    }

    returned_action = reachy_mini.send_action(action)
    assert returned_action == action

    # Verify that _set_joint_positions was called
    reachy_mini.robot._set_joint_positions.assert_called_once()
    call_args, call_kwargs = reachy_mini.robot._set_joint_positions.call_args
    assert "head_joint_positions" in call_kwargs
    assert "antennas_joint_positions" in call_kwargs

    # Check that values were converted correctly (deg to rad)
    assert call_kwargs["head_joint_positions"][0] == pytest.approx(np.deg2rad(-45.0))
    assert call_kwargs["head_joint_positions"][1] == pytest.approx(np.deg2rad(10.0))
    assert call_kwargs["antennas_joint_positions"][0] == pytest.approx(np.deg2rad(60.0))
    assert call_kwargs["antennas_joint_positions"][1] == pytest.approx(np.deg2rad(-60.0))


def test_send_action_clipping(reachy_mini):
    reachy_mini.connect()

    action = {
        "body_rotation.pos": -200.0,  # Exceeds min
        "stewart_1.pos": 100.0,  # Exceeds max
        "stewart_2.pos": 10.0,
        "stewart_3.pos": 10.0,
        "stewart_4.pos": 10.0,
        "stewart_5.pos": 10.0,
        "stewart_6.pos": 10.0,
        "right_antenna.pos": 100.0,  # Exceeds max
        "left_antenna.pos": -100.0,  # Exceeds min
    }

    reachy_mini.send_action(action)

    reachy_mini.robot._set_joint_positions.assert_called_once()
    _, call_kwargs = reachy_mini.robot._set_joint_positions.call_args

    assert call_kwargs["head_joint_positions"][0] == pytest.approx(np.deg2rad(reachy_mini.config.body_yaw_limits_deg[0]))
    assert call_kwargs["head_joint_positions"][1] == pytest.approx(np.deg2rad(reachy_mini.config.stewart_pos_limits_deg[1]))
    assert call_kwargs["antennas_joint_positions"][0] == pytest.approx(
        np.deg2rad(reachy_mini.config.antennas_pos_limits_deg[1])
    )
    assert call_kwargs["antennas_joint_positions"][1] == pytest.approx(
        np.deg2rad(reachy_mini.config.antennas_pos_limits_deg[0])
    )


def test_observation_fallback(reachy_mini):
    # Test the fallback observation path if get_..._positions methods don't exist
    reachy_mini.connect()
    # Remove the primary methods from the mock
    del reachy_mini.robot.get_current_joint_positions
    del reachy_mini.robot.get_present_antenna_joint_positions

    obs = reachy_mini.get_observation()

    assert obs["body_rotation.pos"] == pytest.approx(10.0)
    assert obs["right_antenna.pos"] == pytest.approx(20.0)
    assert obs["left_antenna.pos"] == pytest.approx(-20.0)
