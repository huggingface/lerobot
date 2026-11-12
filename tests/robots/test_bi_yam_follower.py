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

from unittest.mock import MagicMock

import pytest

from lerobot.robots.bi_yam_follower import BiYamFollower, BiYamFollowerConfig
from lerobot.robots.yam_follower import (
    YamFollowerConfig,
    yam_follower as yam_module,
)
from tests.mocks.mock_yam import calibrate_gripper, make_ready, mock_hardware as mock_arm_hardware


def arm_config(port="can0", **kwargs):
    return YamFollowerConfig(port=port, gravity_compensation=False, **kwargs)


@pytest.fixture
def robot(tmp_path, monkeypatch):
    monkeypatch.setattr(yam_module, "require_package", lambda *a, **kw: None)
    robot = BiYamFollower(
        BiYamFollowerConfig(
            id="test",
            calibration_dir=tmp_path,
            left_arm_config=arm_config("can0"),
            right_arm_config=arm_config("can1"),
        )
    )
    for arm in robot.arms.values():
        calibrate_gripper(arm)
    return robot


def mock_hardware(robot, monkeypatch):
    for arm in robot.arms.values():
        mock_arm_hardware(arm, monkeypatch)


def answer_gripper_prompts(robot, monkeypatch, answers=()):
    """Answer calibration prompts with `answers`, then put each arm's gripper where the prompt asks."""
    pending = list(answers)
    prompts = []

    def prompt(text):
        prompts.append(text)
        side = "left" if "test_left" in text else "right"
        robot.arms[side].bus.read_states.return_value.position[6] = 6.1 if "open" in text else 0.1
        return pending.pop(0) if pending else ""

    monkeypatch.setattr("builtins.input", prompt)
    monkeypatch.setattr(yam_module.time, "sleep", lambda _: None)
    return prompts


def ready(robot):
    for arm in robot.arms.values():
        make_ready(arm)
    return dict.fromkeys(robot.action_features, 0.0)


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


def test_stale_feedback_on_either_arm_stops_both(robot):
    action = ready(robot)
    robot.right_arm.controller._updated_at -= 1
    with pytest.raises(ConnectionError, match="stale"):
        robot.send_action(action)
    with pytest.raises(ConnectionError, match="stopped"):
        robot.left_arm.controller.latest()


def test_inactive_arm_refuses_its_action(robot):
    action = ready(robot)
    robot.right_arm.controller._thread = None
    with pytest.raises(RuntimeError, match="active controller"):
        robot.send_action(action)


def test_connect_keeps_both_arms_off_until_both_are_calibrated(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.right_arm.calibration = {}
    robot.connect(calibrate=False)
    for arm in robot.arms.values():
        arm.bus.enable_torque_at.assert_not_called()
    robot.disconnect()


def test_connect_calibrates_both_arms_before_enabling_torque(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    calibrated_at_enable = []
    for arm in robot.arms.values():
        arm.calibration = {}
        arm.bus.enable_torque_at.side_effect = lambda hold: calibrated_at_enable.append(robot.is_calibrated)
    answer_gripper_prompts(robot, monkeypatch)
    robot.connect()
    assert calibrated_at_enable == [True, True]
    for arm in robot.arms.values():
        arm.bus.connect.assert_called_once()
    robot.disconnect()


def test_recalibrating_one_arm_keeps_the_other_running(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.connect(calibrate=False)  # what lerobot-calibrate does before calibrate()
    # Re-measure the left gripper, keep the right one's calibration.
    prompts = answer_gripper_prompts(robot, monkeypatch, answers=["c"])
    robot.calibrate()
    assert "Support the arm" in prompts[1] and "test_left" in prompts[1]
    assert not robot.left_arm.controller.active
    robot.left_arm.bus.disable_torque.assert_called_once()
    robot.right_arm.controller.latest()  # still running, with torque on
    robot.right_arm.bus.disable_torque.assert_not_called()
    robot.disconnect()


def test_second_arm_refuses_torque_after_a_fault_on_the_first(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    left = robot.left_arm.controller
    start_left = left.start

    def start_then_fault():
        start_left()
        robot._fault.set()  # what the left control loop does when it faults

    monkeypatch.setattr(left, "start", start_then_fault)
    with pytest.raises(ConnectionError, match="can't start"):
        robot.connect()
    robot.right_arm.bus.enable_torque_at.assert_not_called()
    assert not robot.is_connected


def test_bad_second_arm_position_never_enables_first(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    robot.right_arm.bus.read_states.return_value.position[1] = 1
    with pytest.raises(ValueError, match="initial_position_deg"):
        robot.connect()
    for arm in robot.arms.values():
        arm.bus.enable_torque_at.assert_not_called()
        arm.bus.disconnect.assert_called_once()


def test_both_arms_configure_before_either_is_enabled(robot, monkeypatch):
    mock_hardware(robot, monkeypatch)
    events = []
    for side, arm in robot.arms.items():
        arm.bus.set_mit_mode.side_effect = lambda side=side: events.append(f"{side}:configure")
        arm.bus.enable_torque_at.side_effect = lambda hold, side=side: events.append(f"{side}:enable")
    robot.connect()
    robot.disconnect()
    assert events == ["left:configure", "right:configure", "left:enable", "right:enable"]


def test_partial_disconnect_can_be_retried(robot):
    for arm in robot.arms.values():
        arm._connected = True
    robot.left_arm.disconnect = MagicMock(side_effect=RuntimeError("controller did not stop"))
    robot.right_arm.disconnect = MagicMock(side_effect=lambda: setattr(robot.right_arm, "_connected", False))
    with pytest.raises(RuntimeError, match="controller did not stop"):
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


def test_installed_optional_dependencies_allow_construction(tmp_path):
    pytest.importorskip("motorbridge")
    pytest.importorskip("can")
    pytest.importorskip("placo")
    bot = BiYamFollower(
        BiYamFollowerConfig(
            id="imports",
            calibration_dir=tmp_path,
            left_arm_config=YamFollowerConfig(port="can0"),
            right_arm_config=YamFollowerConfig(port="can1"),
        )
    )
    assert not bot.is_connected
