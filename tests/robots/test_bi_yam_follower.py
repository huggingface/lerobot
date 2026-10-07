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

import math
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.bi_yam_follower import BiYamFollower, BiYamFollowerConfig
from lerobot.robots.yam_follower import YamFollower, YamFollowerConfigBase, yam_follower as yam_module


def arm_config(port="can0", **kwargs):
    return YamFollowerConfigBase(
        port=port,
        gripper_closed_deg=math.degrees(0.1),
        gripper_open_deg=math.degrees(6.1),
        gravity_compensation=False,
        **kwargs,
    )


@pytest.fixture
def robot(tmp_path, monkeypatch):
    monkeypatch.setattr(yam_module, "require_package", lambda *a, **kw: None)
    return BiYamFollower(
        BiYamFollowerConfig(
            id="test",
            calibration_dir=tmp_path,
            left_arm_config=arm_config("can0"),
            right_arm_config=arm_config("can1"),
        )
    )


def mock_bus():
    bus = MagicMock(spec=yam_module._YamBus)
    bus.enabled = False
    position = np.array([0, 0, 0, 0, 0, 0, 0.1])  # raw radians, gripper closed
    bus.read_states.return_value = yam_module.MotorStates(
        position=position, velocity=np.zeros(7), torque=np.zeros(7)
    )
    return bus


def attach_bus(arm, bus):
    arm.bus = arm.servo.bus = bus
    return bus


def mock_hardware(robot, monkeypatch):
    for arm in robot.arms.values():
        attach_bus(arm, mock_bus())
        monkeypatch.setattr(arm.servo, "start", MagicMock())


def make_writable(robot):
    for arm in robot.arms.values():
        arm.config.read_only = False


def ready(robot):
    make_writable(robot)
    for arm in robot.arms.values():
        arm._connected = True
        arm.servo.active = True
        arm.servo.updated_at = time.monotonic()
    return dict.fromkeys(robot.action_features, 0.0)


def test_bimanual_composes_single_arm_followers(robot):
    assert isinstance(robot.left_arm, YamFollower)
    assert isinstance(robot.right_arm, YamFollower)
    assert robot.left_arm.servo.stop_event is robot.right_arm.servo.stop_event is robot._stop


def test_default_id_gives_each_arm_its_own_calibration_file(tmp_path, monkeypatch):
    monkeypatch.setattr(yam_module, "require_package", lambda *a, **kw: None)
    robot = BiYamFollower(
        BiYamFollowerConfig(
            calibration_dir=tmp_path, left_arm_config=arm_config("can0"), right_arm_config=arm_config("can1")
        )
    )
    assert robot.left_arm.calibration_fpath.name == "bi_yam_follower_left.json"
    assert robot.right_arm.calibration_fpath.name == "bi_yam_follower_right.json"


def test_per_arm_and_top_level_cameras_follow_bimanual_conventions(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from lerobot.cameras.opencv import OpenCVCameraConfig

    monkeypatch.setattr(yam_module, "require_package", lambda *a, **kw: None)

    def camera():
        return OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)

    robot = BiYamFollower(
        BiYamFollowerConfig(
            calibration_dir=tmp_path,
            left_arm_config=arm_config("can0", cameras={"wrist": camera()}),
            right_arm_config=arm_config("can1", cameras={"wrist": camera()}),
            cameras={"top": camera()},
        )
    )
    camera_keys = {k for k, v in robot.observation_features.items() if isinstance(v, tuple)}
    assert camera_keys == {"top", "left_wrist", "right_wrist"}
    with pytest.raises(ValueError, match="collide"):
        BiYamFollower(
            BiYamFollowerConfig(
                calibration_dir=tmp_path,
                left_arm_config=arm_config("can0", cameras={"top": camera()}),
                right_arm_config=arm_config("can1"),
                cameras={"top": camera()},
            )
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_action_rejected_atomically(robot, bad):
    action = ready(robot)
    action["right_joint_1.pos"] = bad
    with pytest.raises(ValueError):
        robot.send_action(action)
    assert all(np.all(arm.servo.target == 0) for arm in robot.arms.values())


def test_quantized_home_clamped_to_joint_limit(robot):
    action = ready(robot)
    action["left_joint_2.pos"] = math.degrees(-0.00019074)
    sent = robot.send_action(action)
    assert sent["left_joint_2.pos"] == 0
    assert robot.left_arm.servo.target[1] == 0


def test_stale_feedback_refuses_both_targets(robot):
    action = ready(robot)
    robot.right_arm.servo.updated_at -= 1
    with pytest.raises(ConnectionError, match="stale"):
        robot.send_action(action)
    assert robot._stop.is_set()
    assert all(np.all(arm.servo.target == 0) for arm in robot.arms.values())


def test_readonly_rejects_actions(robot):
    for arm in robot.arms.values():
        arm._connected = True
    with pytest.raises(RuntimeError, match="read-only"):
        robot.send_action(dict.fromkeys(robot.action_features, 0))


def test_calibration_connect_never_enables_motors(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    make_writable(robot)
    robot.connect(calibrate=False)
    for arm in robot.arms.values():
        arm.bus.set_mit_mode.assert_not_called()
        arm.bus.enable.assert_not_called()
    robot.disconnect()


def test_each_arm_loads_its_model_before_its_can_interface(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    make_writable(robot)
    events = []
    for side, arm in robot.arms.items():
        arm.config.gravity_compensation = True

        def load_model(side=side, arm=arm):
            if arm.gravity_model is None:
                events.append(f"{side}:model")
                arm.gravity_model = object()

        arm._load_control_model = load_model
        arm.bus.open.side_effect = lambda side=side: events.append(f"{side}:connect")
    robot.connect()
    robot.disconnect()
    assert events[:4] == ["left:model", "left:connect", "right:model", "right:connect"]


def test_bad_second_arm_pose_never_enables_first(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    make_writable(robot)
    robot.right_arm.bus.read_states.return_value.position[1] = 1
    with pytest.raises(ValueError, match="initial pose"):
        robot.connect()
    for arm in robot.arms.values():
        arm.bus.enable.assert_not_called()
        arm.bus.close.assert_called_once()


def test_both_arms_configure_before_either_is_enabled(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    make_writable(robot)
    events = []
    for side, arm in robot.arms.items():
        arm.bus.set_mit_mode.side_effect = lambda side=side: events.append(f"{side}:configure")
        arm.bus.enable.side_effect = lambda hold, side=side: events.append(f"{side}:enable")
    robot.connect()
    robot.disconnect()
    assert events == ["left:configure", "right:configure", "left:enable", "right:enable"]


def test_servo_error_stops_and_disables_both_arms(robot):
    for arm in robot.arms.values():
        attach_bus(arm, mock_bus())
    robot.left_arm.bus.read_states.side_effect = ConnectionError("lost")
    robot.left_arm.servo._run()
    robot.right_arm.servo._run()
    assert isinstance(robot.left_arm.servo.failure, ConnectionError)
    assert robot._stop.is_set()
    for arm in robot.arms.values():
        arm.bus.disable.assert_called_once()


def test_disconnect_attempts_both_arms_after_one_fails(robot):
    for arm in robot.arms.values():
        arm._connected = True
    robot.left_arm.disconnect = MagicMock(side_effect=RuntimeError("left failed"))
    robot.right_arm.disconnect = MagicMock()
    with pytest.raises(RuntimeError, match="left failed"):
        robot.disconnect()
    robot.left_arm.disconnect.assert_called_once()
    robot.right_arm.disconnect.assert_called_once()
    for arm in robot.arms.values():
        arm._connected = False


def test_partial_disconnect_can_be_retried(robot):
    for arm in robot.arms.values():
        arm._connected = True
    robot.left_arm.disconnect = MagicMock(side_effect=RuntimeError("servo did not stop"))
    robot.right_arm.disconnect = MagicMock(side_effect=lambda: setattr(robot.right_arm, "_connected", False))
    with pytest.raises(RuntimeError, match="servo did not stop"):
        robot.disconnect()
    assert not robot.is_connected
    robot.left_arm.disconnect.side_effect = lambda: setattr(robot.left_arm, "_connected", False)
    robot.disconnect()
    assert robot.left_arm.disconnect.call_count == 2
    robot.right_arm.disconnect.assert_called_once()


def test_partial_connection_must_be_disconnected_before_reconnecting(robot):
    robot.left_arm._connected = True
    with pytest.raises(RuntimeError, match="Disconnect both"):
        robot.connect()


def test_calibration_delegates_to_both_single_arms(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)
    robot.left_arm.calibrate = MagicMock()
    robot.right_arm.calibrate = MagicMock()
    robot.calibrate()
    robot.left_arm.calibrate.assert_called_once()
    robot.right_arm.calibrate.assert_called_once()
    robot.disconnect()


def test_installed_optional_dependencies_allow_construction(tmp_path):
    pytest.importorskip("motorbridge")
    pytest.importorskip("can")
    pytest.importorskip("placo")
    bot = BiYamFollower(
        BiYamFollowerConfig(
            id="imports",
            calibration_dir=tmp_path,
            left_arm_config=YamFollowerConfigBase(port="can0"),
            right_arm_config=YamFollowerConfigBase(port="can1"),
        )
    )
    assert not bot.is_connected
