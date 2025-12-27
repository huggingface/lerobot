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
    sdk_mock.get_current_joint_positions.return_value = [[np.deg2rad(10.0)]]  # body_yaw
    sdk_mock.get_present_antenna_joint_positions.return_value = [
        np.deg2rad(20.0),
        np.deg2rad(-20.0),
    ]  # left, right

    # Mock individual joint objects for fallback path
    sdk_mock.body_yaw.present_position = np.deg2rad(10.0)
    sdk_mock.antennas.left.present_position = np.deg2rad(20.0)
    sdk_mock.antennas.right.present_position = np.deg2rad(-20.0)

    sdk_mock.set_target = MagicMock()
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
        "head.z.pos",
        "body.yaw.pos",
        "antennas.left.pos",
        "antennas.right.pos",
        *reachy_mini.cameras.keys(),
    }
    assert set(obs.keys()) == expected_keys

    # Check values from the mock
    assert obs["body.yaw.pos"] == pytest.approx(10.0)
    assert obs["antennas.left.pos"] == pytest.approx(20.0)
    assert obs["antennas.right.pos"] == pytest.approx(-20.0)
    # head.z.pos is from last action, 0.0 initially
    assert obs["head.z.pos"] == 0.0

    cam_key = list(reachy_mini.cameras.keys())[0]
    cam_cfg = reachy_mini.config.cameras[cam_key]
    assert obs[cam_key].shape == (cam_cfg.height, cam_cfg.width, 3)


def test_send_action(reachy_mini):
    reachy_mini.connect()

    action = {
        "head.z.pos": 30.0,
        "body.yaw.pos": -45.0,
        "antennas.left.pos": 60.0,
        "antennas.right.pos": -60.0,
    }

    returned_action = reachy_mini.send_action(action)
    assert returned_action == action

    # Check that the last commanded head position is stored
    assert reachy_mini.last_head_z_pos == 30.0

    # Verify that set_target was called
    reachy_mini.robot.set_target.assert_called_once()
    call_args, call_kwargs = reachy_mini.robot.set_target.call_args
    assert "head" in call_kwargs
    assert "antennas" in call_kwargs
    assert "body_yaw" in call_kwargs

    # Check that values were converted correctly (deg to rad for angles)
    assert call_kwargs["body_yaw"] == pytest.approx(np.deg2rad(-45.0))
    assert call_kwargs["antennas"][0] == pytest.approx(np.deg2rad(60.0))
    assert call_kwargs["antennas"][1] == pytest.approx(np.deg2rad(-60.0))


def test_send_action_clipping(reachy_mini):
    reachy_mini.connect()

    action = {
        "head.z.pos": 100.0,  # Exceeds max
        "body.yaw.pos": -200.0,  # Exceeds min
        "antennas.left.pos": 100.0,  # Exceeds max
        "antennas.right.pos": -100.0,  # Exceeds min
    }

    reachy_mini.send_action(action)

    assert reachy_mini.last_head_z_pos == reachy_mini.config.head_z_pos_limits_mm[1]

    reachy_mini.robot.set_target.assert_called_once()
    _, call_kwargs = reachy_mini.robot.set_target.call_args

    assert call_kwargs["body_yaw"] == pytest.approx(np.deg2rad(reachy_mini.config.body_yaw_limits_deg[0]))
    assert call_kwargs["antennas"][0] == pytest.approx(
        np.deg2rad(reachy_mini.config.antennas_pos_limits_deg[1])
    )
    assert call_kwargs["antennas"][1] == pytest.approx(
        np.deg2rad(reachy_mini.config.antennas_pos_limits_deg[0])
    )


def test_observation_fallback(reachy_mini):
    # Test the fallback observation path if get_..._positions methods don't exist
    reachy_mini.connect()
    # Remove the primary methods from the mock
    del reachy_mini.robot.get_current_joint_positions
    del reachy_mini.robot.get_present_antenna_joint_positions

    obs = reachy_mini.get_observation()

    assert obs["body.yaw.pos"] == pytest.approx(10.0)
    assert obs["antennas.left.pos"] == pytest.approx(20.0)
    assert obs["antennas.right.pos"] == pytest.approx(-20.0)
