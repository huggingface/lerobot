# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.bi_yam_follower import (
    BiYamFollower,
    BiYamFollowerConfig,
    bi_yam_follower as robot_module,
    yam_arm as arm_module,
)
from lerobot.robots.bi_yam_follower.config_bi_yam_follower import MOTOR_NAMES, YamArmConfig
from lerobot.robots.bi_yam_follower.yam_arm import (
    GravityCompensation,
    YamArm,
    decode_positions,
    encode_positions,
)


def arm_config(port="can0", **kwargs):
    return YamArmConfig(
        port=port, gripper_closed_rad=0.1, gripper_open_rad=6.1, gravity_compensation=False, **kwargs
    )


@pytest.fixture
def robot(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_module, "require_package", lambda *a, **kw: None)
    return BiYamFollower(
        BiYamFollowerConfig(
            id="test",
            calibration_dir=tmp_path,
            left_arm=arm_config("can0"),
            right_arm=arm_config("can1"),
        )
    )


@pytest.mark.parametrize("opened", [6.1, -5.9])
def test_units_order_and_gripper_polarity(opened):
    cfg = arm_config()
    cfg.gripper_open_rad = opened
    raw = [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, (0.1 + opened) / 2]
    cfg.joint_signs[0] = -1
    cfg.joint_offsets_rad[0] = 0.2
    states = {name: SimpleNamespace(pos=value) for name, value in zip(MOTOR_NAMES, raw, strict=True)}
    decoded = decode_positions(cfg, states)
    np.testing.assert_allclose(decoded, [0.1, 0.2, 0.3, -0.4, 0.5, -0.6, 0.5])
    np.testing.assert_allclose(encode_positions(cfg, decoded), raw)


def test_motorbridge_mapping_and_mit_radians(monkeypatch):
    controller = MagicMock()
    controller.add_damiao_motor.side_effect = lambda *args: MagicMock()
    factory = MagicMock(return_value=controller)
    monkeypatch.setattr(arm_module, "Controller", factory, raising=False)
    monkeypatch.setattr(arm_module, "Mode", SimpleNamespace(MIT=1), raising=False)
    monkeypatch.setattr(arm_module, "can", SimpleNamespace(Bus=MagicMock()), raising=False)
    arm = YamArm(arm_config())
    arm.connect()
    factory.assert_called_once_with(channel="can0")
    assert [call.args for call in controller.add_damiao_motor.call_args_list] == [
        (i + 1, i + 17, "4340" if i < 3 else "4310") for i in range(7)
    ]
    arm.position = np.array([0.2, 0.5, 0.3, 0, 0, 0, 1.0])
    arm.configure()
    arm.enable()
    arm.motors["joint_0"].send_mit.assert_called_once_with(0.2, 0, 0, 0, 0)
    arm.motors["gripper"].send_mit.assert_called_once_with(6.1, 0, 0, 0, 0)
    arm.close()
    controller.close.assert_called_once()


def frame(i, age=0, status=1):
    return SimpleNamespace(
        is_error_frame=False,
        is_remote_frame=False,
        is_extended_id=False,
        dlc=8,
        arbitration_id=i + 16,
        data=bytes([status << 4 | i]) + bytes(7),
        timestamp=time.time() - age,
    )


def monitored_arm(frames):
    arm = YamArm(arm_config())
    arm.bus = MagicMock()
    queued = deque(frames)
    arm.monitor = SimpleNamespace(recv=lambda timeout: queued.popleft() if queued else None)
    arm.motors = {name: MagicMock() for name in MOTOR_NAMES}
    for motor in arm.motors.values():
        motor.get_state.return_value = SimpleNamespace(pos=0.1, status_code=1)
    return arm


def test_cached_motorbridge_states_do_not_hide_missing_motor():
    arm = monitored_arm([frame(i) for i in range(1, 7)])
    with pytest.raises(ConnectionError, match="stale"):
        arm.read(0.2)


def test_queued_old_packets_do_not_count_as_fresh():
    arm = monitored_arm([frame(i, age=1) for i in range(1, 8)])
    with pytest.raises(ConnectionError, match="stale"):
        arm.read(0.2)


def test_fault_in_any_feedback_packet_fails():
    arm = monitored_arm([frame(i, status=13 if i == 3 else 1) for i in range(1, 8)])
    with pytest.raises(ConnectionError, match="fault 0xd"):
        arm.read(0.2)


def test_fresh_feedback_accepts_stationary_motors():
    arm = monitored_arm([frame(i) for i in range(1, 8)])
    assert len(arm.read(0.2)) == 7


def test_slew_gripper_torque_and_gravity_feedforward():
    arm = YamArm(arm_config(max_gripper_speed_s=2))
    arm.position = np.array([0, 0.5, 0.5, 0, 0, 0, 0.5])
    arm.command = arm.position.copy()
    arm.target = arm.position + 0.1
    arm.gravity = SimpleNamespace(torque=lambda p: np.ones(6))
    packet = arm.command_packet(arm.position, 0.01)
    assert packet["joint_0"] == pytest.approx((0.003, 0, 80, 5, 1))
    assert packet["joint_1"][-1] == pytest.approx(1.1)
    # Raw gripper target is capped to 0.5 Nm / 20 Nm/rad = 0.025 rad ahead.
    assert packet["gripper"] == pytest.approx((3.125, 0, 20, 0.5, 0))


def test_gravity_matches_potential_energy_gradient():
    pytest.importorskip("mujoco")
    model = GravityCompensation()
    pose = np.array([0.2, 1.0, 1.1, -0.5, 0.3, -0.2, 0.5])
    torque = model.torque(pose)
    expected = []
    for i in range(6):
        energies = []
        for delta in (-1e-5, 1e-5):
            shifted = pose.copy()
            shifted[i] += delta
            model.torque(shifted)
            energies.append(
                float(-np.sum(model.model.body_mass[:, None] * model.data.xipos * model.model.opt.gravity))
            )
        expected.append((energies[1] - energies[0]) / 2e-5)
    np.testing.assert_allclose(torque, expected, atol=1e-6)


def ready(robot):
    robot._connected = True
    robot.config.read_only = False
    for arm in robot.arms.values():
        arm.updated_at = time.monotonic()
    return dict.fromkeys(robot.action_features, 0.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 5.0])
def test_action_rejected_atomically(robot, bad):
    action = ready(robot)
    action["right_joint_0.pos"] = bad
    with pytest.raises(ValueError):
        robot.send_action(action)
    assert all(np.all(arm.target == 0) for arm in robot.arms.values())


def test_quantized_home_clamped_to_joint_limit(robot):
    action = ready(robot)
    action["left_joint_1.pos"] = -0.00019074
    sent = robot.send_action(action)
    assert sent["left_joint_1.pos"] == 0
    assert robot.arms["left"].target[1] == 0


def test_stale_feedback_refuses_action(robot):
    action = ready(robot)
    robot.arms["right"].updated_at -= 1
    with pytest.raises(ConnectionError, match="stale"):
        robot.send_action(action)
    assert robot._stop.is_set()


def test_readonly_rejects_actions(robot):
    robot._connected = True
    with pytest.raises(RuntimeError, match="read-only"):
        robot.send_action(dict.fromkeys(robot.action_features, 0))


def mock_arm_io(robot, monkeypatch):
    for arm in robot.arms.values():
        monkeypatch.setattr(arm, "connect", MagicMock())
        monkeypatch.setattr(
            arm,
            "read",
            MagicMock(
                return_value={
                    name: SimpleNamespace(pos=0.1 if name == "gripper" else 0, status_code=0)
                    for name in MOTOR_NAMES
                }
            ),
        )
        monkeypatch.setattr(arm, "close", MagicMock())
        monkeypatch.setattr(arm, "configure", MagicMock())
        monkeypatch.setattr(arm, "enable", MagicMock())


def test_calibration_connect_never_enables_motors(robot, monkeypatch):
    mock_arm_io(robot, monkeypatch)
    robot.config.read_only = False
    robot.connect(calibrate=False)
    for arm in robot.arms.values():
        arm.enable.assert_not_called()
        arm.configure.assert_not_called()
    robot.disconnect()


def test_bad_second_arm_pose_never_enables_first(robot, monkeypatch):
    mock_arm_io(robot, monkeypatch)
    robot.config.read_only = False
    robot.arms["right"].read.return_value["joint_1"].pos = 1
    with pytest.raises(ValueError, match="initial pose"):
        robot.connect()
    for arm in robot.arms.values():
        arm.enable.assert_not_called()
        arm.close.assert_called_once()


def test_servo_error_disables_both_arms(robot, monkeypatch):
    mock_arm_io(robot, monkeypatch)
    for arm in robot.arms.values():
        arm.enabled = True
        arm.bus = MagicMock()
    robot.arms["left"].read.side_effect = ConnectionError("lost")
    robot._run()
    assert isinstance(robot._failure, ConnectionError)
    assert robot._stop.is_set()
    for arm in robot.arms.values():
        arm.bus.disable_all.assert_called_once()


def test_calibration_preserves_factory_zeros_and_polarity(robot, monkeypatch):
    mock_arm_io(robot, monkeypatch)
    robot.connect(calibrate=False)

    def prompt(text):
        opening = "open" in text
        robot.arms["left"].read.return_value["gripper"].pos = -5.9 if opening else 0.1
        robot.arms["right"].read.return_value["gripper"].pos = 6.1 if opening else 0.1

    monkeypatch.setattr("builtins.input", prompt)
    robot.calibrate()
    assert robot.calibration["left_gripper"].drive_mode == 1
    assert robot.calibration["right_gripper"].drive_mode == 0
    assert robot.config.left_arm.gripper_open_rad == pytest.approx(-5.9, abs=0.0002)
    assert robot.calibration_fpath.is_file()
    for arm in robot.arms.values():
        arm.enable.assert_not_called()
        arm.configure.assert_not_called()
    robot.disconnect()


def test_installed_optional_dependencies_allow_construction(tmp_path):
    pytest.importorskip("motorbridge")
    pytest.importorskip("can")
    bot = BiYamFollower(BiYamFollowerConfig(id="imports", calibration_dir=tmp_path))
    assert not bot.is_connected


def test_servo_waits_for_async_reply_instead_of_rejecting_empty_cache(robot):
    # A healthy reply appears only after the initial nonblocking receive. The old
    # servo failed immediately even though the response met the existing deadline.
    for side, cfg in (("left", robot.config.left_arm), ("right", robot.config.right_arm)):
        arm = monitored_arm([])
        arm.config = cfg
        replies = deque([None, *[frame(i) for i in range(1, 8)], None])

        def receive(timeout, replies=replies, side=side):
            value = replies.popleft() if replies else None
            if not replies and side == "right":
                robot._stop.set()
            return value

        arm.monitor.recv = receive
        robot.arms[side] = arm
    robot._run()
    assert robot._failure is None
    assert all(arm.updated_at > 0 for arm in robot.arms.values())


@pytest.mark.parametrize("enabled,preexisting", [(True, False), (True, True), (False, False)])
def test_gc_freeze_is_shared_and_preserves_callers_state(monkeypatch, enabled, preexisting):
    gc_mock = MagicMock()
    gc_mock.isenabled.return_value = enabled
    gc_mock.get_freeze_count.return_value = int(preexisting)
    monkeypatch.setattr(robot_module, "gc", gc_mock)
    guard = robot_module._ControlGC
    assert guard._users == 0
    guard.acquire()
    guard.acquire()
    guard.release()
    gc_mock.unfreeze.assert_not_called()
    guard.release()
    assert gc_mock.collect.call_count == int(enabled)
    assert gc_mock.freeze.call_count == int(enabled)
    assert gc_mock.unfreeze.call_count == int(enabled and not preexisting)
    assert guard._users == 0


def test_connect_failure_releases_gc_freeze(robot, monkeypatch):
    mock_arm_io(robot, monkeypatch)
    robot.arms["right"].connect.side_effect = ConnectionError("no adapter")
    guard = MagicMock()
    monkeypatch.setattr(robot_module, "_ControlGC", guard)
    with pytest.raises(ConnectionError, match="no adapter"):
        robot.connect()
    guard.acquire.assert_called_once()
    guard.release.assert_called_once()
    assert not robot._gc_acquired
