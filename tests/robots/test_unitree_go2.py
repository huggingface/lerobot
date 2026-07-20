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

"""Tests for the Unitree Go2 robot.

The SDK is only imported inside ``UnitreeGo2.connect()``, so everything
here runs without unitree_sdk2py installed: the sport client, state
subscriber and video client are replaced with mocks.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from lerobot.robots.unitree_go2 import UnitreeGo2, UnitreeGo2Config

# ---------------------------------------------------------------------------
# Config (no SDK needed)
# ---------------------------------------------------------------------------


class TestUnitreeGo2Config:
    def test_registered_type_name(self):
        assert UnitreeGo2Config().type == "unitree_go2"

    def test_default_config(self):
        cfg = UnitreeGo2Config()
        assert cfg.domain_id == 0
        assert cfg.use_front_camera is True
        assert cfg.stand_on_connect is True
        assert cfg.cameras == {}

    def test_safety_clamps_are_positive(self):
        cfg = UnitreeGo2Config()
        assert cfg.max_x_vel > 0
        assert cfg.max_y_vel > 0
        assert cfg.max_theta_vel > 0


# ---------------------------------------------------------------------------
# Features (no SDK needed)
# ---------------------------------------------------------------------------


def _make_robot(**cfg_kwargs) -> UnitreeGo2:
    cfg = UnitreeGo2Config(id="test_go2", **cfg_kwargs)
    return UnitreeGo2(cfg)


class TestFeatures:
    def test_action_features(self):
        robot = _make_robot()
        assert robot.action_features == {"x.vel": float, "y.vel": float, "theta.vel": float}

    def test_observation_features_with_front_camera(self):
        robot = _make_robot()
        ft = robot.observation_features
        assert ft["front"] == (720, 1280, 3)
        for key in ("x.pos", "y.pos", "theta.pos", "x.vel", "y.vel", "theta.vel"):
            assert ft[key] is float

    def test_observation_features_without_front_camera(self):
        robot = _make_robot(use_front_camera=False)
        assert "front" not in robot.observation_features

    def test_features_available_before_connect(self):
        robot = _make_robot()
        assert not robot.is_connected
        assert robot.observation_features
        assert robot.action_features

    def test_is_calibrated_always_true(self):
        assert _make_robot().is_calibrated is True


# ---------------------------------------------------------------------------
# I/O with mocked SDK handles
# ---------------------------------------------------------------------------


def _connected_robot(**cfg_kwargs) -> UnitreeGo2:
    """A robot with mocked SDK handles, as if connect() had run."""
    robot = _make_robot(**cfg_kwargs)
    robot._sport = MagicMock()
    robot._video = MagicMock()
    robot._connected = True
    return robot


def _fake_state(x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, yaw_speed=0.0):
    return SimpleNamespace(
        position=[x, y, 0.0],
        velocity=[vx, vy, 0.0],
        yaw_speed=yaw_speed,
        imu_state=SimpleNamespace(rpy=[0.0, 0.0, yaw]),
    )


class TestSendAction:
    def test_action_reaches_sport_move(self):
        robot = _connected_robot()
        sent = robot.send_action({"x.vel": 0.3, "y.vel": -0.1, "theta.vel": 0.5})
        robot._sport.Move.assert_called_once_with(0.3, -0.1, 0.5)
        assert sent == {"x.vel": 0.3, "y.vel": -0.1, "theta.vel": 0.5}

    def test_action_is_clamped(self):
        robot = _connected_robot(max_x_vel=0.5, max_y_vel=0.2, max_theta_vel=1.0)
        sent = robot.send_action({"x.vel": 9.0, "y.vel": -9.0, "theta.vel": -9.0})
        robot._sport.Move.assert_called_once_with(0.5, -0.2, -1.0)
        assert sent == {"x.vel": 0.5, "y.vel": -0.2, "theta.vel": -1.0}

    def test_missing_keys_default_to_zero(self):
        robot = _connected_robot()
        sent = robot.send_action({})
        robot._sport.Move.assert_called_once_with(0.0, 0.0, 0.0)
        assert sent == {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

    def test_raises_when_not_connected(self):
        robot = _make_robot()
        with pytest.raises(ConnectionError):
            robot.send_action({"x.vel": 0.1})


class TestGetObservation:
    def test_odometry_fields(self):
        robot = _connected_robot(use_front_camera=False)
        robot._latest_state = _fake_state(x=1.0, y=2.0, yaw=0.3, vx=0.1, vy=-0.05, yaw_speed=0.2)
        obs = robot.get_observation()
        assert obs["x.pos"] == pytest.approx(1.0)
        assert obs["y.pos"] == pytest.approx(2.0)
        assert obs["theta.pos"] == pytest.approx(0.3)
        assert obs["x.vel"] == pytest.approx(0.1)
        assert obs["y.vel"] == pytest.approx(-0.05)
        assert obs["theta.vel"] == pytest.approx(0.2)

    def test_odometry_zero_before_first_state(self):
        robot = _connected_robot(use_front_camera=False)
        obs = robot.get_observation()
        assert all(obs[k] == 0.0 for k in robot._odom_ft)

    def test_front_camera_decodes_to_configured_shape(self):
        robot = _connected_robot(front_camera_width=64, front_camera_height=48)
        raw = np.full((48, 64, 3), 128, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", raw)
        assert ok
        robot._video.GetImageSample.return_value = (0, jpeg.tobytes())
        obs = robot.get_observation()
        assert obs["front"].shape == (48, 64, 3)
        assert obs["front"].dtype == np.uint8

    def test_front_camera_resizes_native_frames(self):
        robot = _connected_robot(front_camera_width=64, front_camera_height=48)
        native = np.zeros((720, 1280, 3), dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", native)
        assert ok
        robot._video.GetImageSample.return_value = (0, jpeg.tobytes())
        assert robot.get_observation()["front"].shape == (48, 64, 3)

    def test_front_camera_failure_returns_black_frame(self):
        robot = _connected_robot(front_camera_width=64, front_camera_height=48)
        robot._video.GetImageSample.return_value = (1, None)
        frame = robot.get_observation()["front"]
        assert frame.shape == (48, 64, 3)
        assert frame.sum() == 0

    def test_observation_matches_features(self):
        robot = _connected_robot(front_camera_width=64, front_camera_height=48)
        raw = np.zeros((48, 64, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", raw)
        robot._video.GetImageSample.return_value = (0, jpeg.tobytes())
        obs = robot.get_observation()
        assert set(obs.keys()) == set(robot.observation_features.keys())

    def test_raises_when_not_connected(self):
        robot = _make_robot()
        with pytest.raises(ConnectionError):
            robot.get_observation()


class TestDisconnect:
    def test_disconnect_stops_motion(self):
        robot = _connected_robot()
        sport = robot._sport
        robot.disconnect()
        sport.StopMove.assert_called_once()
        assert not robot.is_connected
