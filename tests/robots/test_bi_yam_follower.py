"""Unit/frame and lifecycle checks; no real robot is connected by this suite."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.bi_yam_follower import (
    BiYamFollower,
    BiYamFollowerConfig,
    YamArmConfig,
    bi_yam_follower as module,
)
from lerobot.robots.bi_yam_follower.bi_yam_follower import _Arm, _Gravity, decode_positions, encode_positions
from lerobot.robots.bi_yam_follower.config_bi_yam_follower import MOTOR_NAMES, YAM_FEATURE_NAMES

pytest.importorskip("can")


def arm_config(port="can0", **kwargs):
    return YamArmConfig(
        port=port, gripper_closed_rad=0.1, gripper_open_rad=6.6, gravity_compensation=False, **kwargs
    )


def states(raw_radians):
    return {
        name: {
            "position": math.degrees(float(value)),
            "velocity": 0.0,
            "torque": 0.0,
            "temp_mos": 25.0,
            "temp_rotor": 25.0,
        }
        for name, value in zip(MOTOR_NAMES, raw_radians, strict=True)
    }


@pytest.mark.parametrize("closed,opened", [(0.1, 6.6), (6.6, 0.1)])
def test_radians_and_continuous_gripper(closed, opened):
    cfg = YamArmConfig(port="can0", gripper_closed_rad=closed, gripper_open_rad=opened)
    raw = np.array([0.2, 1.0, 2.0, -0.5, 0.3, -0.2, closed + 0.25 * (opened - closed)])
    result = decode_positions(cfg, states(raw))
    np.testing.assert_allclose(result, [0.2, 1.0, 2.0, -0.5, 0.3, -0.2, 0.25])
    np.testing.assert_allclose(encode_positions(cfg, result), np.degrees(raw))


def test_joint_frame_transform_is_inverse():
    cfg = arm_config(joint_signs=[-1, 1, 1, 1, 1, 1], joint_offsets_rad=[0.3, 0, 0, 0, 0, 0])
    raw = np.array([0.2, 1, 2, -0.5, 0.3, -0.2, 3.35])
    decoded = decode_positions(cfg, states(raw))
    assert decoded[0] == pytest.approx(0.1)
    assert decoded[-1] == pytest.approx(0.5)
    np.testing.assert_allclose(encode_positions(cfg, decoded), np.degrees(raw))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gripper_closed_rad": 0.0, "gripper_open_rad": 0.0},
        {"joint_signs": [0.0] * 6},
        {"kp": [float("nan")] * 6},
        {"gripper_torque_limit": 2.0},
        {"initial_position_rad": [8.0] * 6},
    ],
)
def test_bad_calibration_or_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        YamArmConfig(port="can0", **kwargs)


def test_no_nominal_gripper_calibration():
    with pytest.raises(ValueError, match="Measure"):
        decode_positions(YamArmConfig(port="can0"), states(np.zeros(7)))


class Bus:
    def __init__(self, raw=None, fail=False):
        self.is_connected = False
        self.raw = np.array([0, 0, 0, 0, 0, 0, 0.1]) if raw is None else raw
        self.fail = fail
        self.calls = []

    def connect(self, handshake=True):
        self.calls.append(("connect", handshake))
        self.is_connected = True

    def sync_read_all_states(self, strict=False):
        assert strict
        if self.fail:
            raise ConnectionError("missing motor")
        return states(self.raw)

    def sync_write_mit(self, packet):
        self.calls.append(("write", packet))

    def enable_torque(self):
        self.calls.append(("enable",))

    def disable_torque(self):
        self.calls.append(("disable",))

    def disconnect(self, disable_torque=True):
        self.calls.append(("disconnect", disable_torque))
        self.is_connected = False


def robot(monkeypatch, tmp_path, *, read_only=True, buses=None):
    buses = buses or {"left": Bus(), "right": Bus()}
    monkeypatch.setattr(module, "make_yam_bus", lambda cfg: buses[cfg.port])
    cfg = BiYamFollowerConfig(
        left_arm=arm_config("left"),
        right_arm=arm_config("right"),
        calibration_dir=tmp_path,
        read_only=read_only,
    )
    return BiYamFollower(cfg), buses


def test_read_only_never_enables_or_commands(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path)
    bot.connect()
    try:
        assert list(bot.action_features) == list(YAM_FEATURE_NAMES)
        assert list(bot.get_observation()) == list(YAM_FEATURE_NAMES)
        with pytest.raises(RuntimeError, match="read_only"):
            bot.send_action(dict.fromkeys(YAM_FEATURE_NAMES, 0.0))
    finally:
        bot.disconnect()
    for bus in buses.values():
        assert bus.calls == [("connect", False), ("disconnect", False)]


def test_second_arm_failure_never_energizes_first(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False, buses={"left": Bus(), "right": Bus(fail=True)})
    with pytest.raises(ConnectionError):
        bot.connect()
    assert all(not b.is_connected for b in buses.values())
    assert all(all(c[0] not in ("enable", "write") for c in b.calls) for b in buses.values())


def test_initial_pose_is_checked_without_homing(monkeypatch, tmp_path):
    raw = np.array([1.0, 0, 0, 0, 0, 0, 0.1])
    bot, buses = robot(monkeypatch, tmp_path, read_only=False, buses={"left": Bus(raw), "right": Bus()})
    with pytest.raises(ValueError, match="initial pose"):
        bot.connect()
    assert not any(c[0] in ("write", "enable") for b in buses.values() for c in b.calls)


def test_bimanual_action_validation_is_atomic(monkeypatch, tmp_path):
    bot, _ = robot(monkeypatch, tmp_path, read_only=False)
    bot.connect()
    try:
        action = dict.fromkeys(YAM_FEATURE_NAMES, 0.0)
        action["left_joint_0.pos"] = 0.2
        action["right_joint_0.pos"] = float("nan")
        with pytest.raises(ValueError):
            bot.send_action(action)
        assert bot.arms["left"].target[0] == 0
        with pytest.raises(ValueError, match="complete"):
            bot.send_action({"x": 0.1, "y": 0.2})
    finally:
        bot.disconnect()


def test_feedback_fault_stops_both_workers(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.connect()
    buses["right"].fail = True
    assert bot._stop.wait(1)
    with pytest.raises(ConnectionError):
        bot.get_observation()
    bot.disconnect()
    assert all(("disable",) in b.calls for b in buses.values())


def test_joint_slew_and_gripper_proportional_torque_bound():
    arm = _Arm(arm_config())
    arm.position = np.zeros(7)
    arm.command = np.zeros(7)
    arm.target = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    packet = arm.command_packet(arm.position, 0.02)
    assert math.radians(packet["joint_0"][2]) == pytest.approx(0.006)
    raw_error = math.radians(packet["gripper"][2]) - 0.1
    assert abs(raw_error * packet["gripper"][0]) <= 0.5 + 1e-8


def test_shared_can_rejected():
    with pytest.raises(ValueError, match="distinct"):
        BiYamFollowerConfig(left_arm=arm_config(), right_arm=arm_config())


def test_cartesian_policy_rejected_before_connect(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path)
    policy = SimpleNamespace(
        type="molmoact2",
        output_features={"action": SimpleNamespace(shape=(14,))},
        dataset_feature_names={},
        control_mode="delta end-effector pose",
    )
    with pytest.raises(ValueError, match="end-effector"):
        bot.validate_policy_config(policy)
    assert all(not b.calls for b in buses.values())


def test_gravity_matches_potential_energy_gradient():
    pytest.importorskip("mujoco")
    model = _Gravity()
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


def test_initial_gripper_check_precedes_enabling_either_arm(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.config.right_arm.initial_gripper_position = 1.0
    with pytest.raises(ValueError, match="gripper.*initial"):
        bot.connect()
    assert not any(c[0] in ("write", "enable") for b in buses.values() for c in b.calls)


def test_startup_packet_uses_measured_pose_and_zero_gains(monkeypatch, tmp_path):
    raw = np.array([0.1, 0.05, 0.1, -0.1, 0.1, 0.1, 0.1])
    bot, buses = robot(monkeypatch, tmp_path, read_only=False, buses={"left": Bus(raw), "right": Bus(raw)})
    bot.connect()
    bot.disconnect()
    for bus in buses.values():
        first = next(call[1] for call in bus.calls if call[0] == "write")
        for i, name in enumerate(MOTOR_NAMES):
            assert first[name] == pytest.approx((0, 0, math.degrees(raw[i]), 0, 0))
        assert bus.calls.index(("enable",)) > next(i for i, c in enumerate(bus.calls) if c[0] == "write")


def test_expired_target_holds_measured_position(monkeypatch, tmp_path):
    bot, _ = robot(monkeypatch, tmp_path, read_only=False)
    arm = bot.arms["left"]
    arm.enabled = True
    arm.command[:] = 0.1
    arm.target[:] = 0.2
    arm.commanded_at = -10
    sent = []

    def stop_after_write(packet):
        sent.append(packet)
        bot._stop.set()

    monkeypatch.setattr(arm.bus, "sync_write_mit", stop_after_write)
    bot._run(arm)
    np.testing.assert_allclose(arm.target, arm.position)
    np.testing.assert_allclose(arm.command, arm.position)
    assert len(sent) == 1
    assert ("disable",) in arm.bus.calls


def test_adapter_serial_mismatch_never_opens_can(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path)

    def mismatch(config):
        raise ValueError("adapter serial mismatch")

    monkeypatch.setattr(module, "verify_adapter", mismatch)
    with pytest.raises(ValueError, match="adapter serial"):
        bot.connect()
    assert all(not bus.calls for bus in buses.values())


def test_standard_calibration_cli_never_enables_and_reloads_endpoints(monkeypatch, tmp_path):
    from lerobot.scripts import lerobot_calibrate

    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    for arm in bot.arms.values():
        arm.config.gripper_closed_rad = arm.config.gripper_open_rad = None
    monkeypatch.setattr(lerobot_calibrate, "make_robot_from_config", lambda _: bot)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    ends = iter(((0.1, -0.02), (-5.1, 5.8)))

    def position_grippers(_):
        left, right = next(ends)
        buses["left"].raw[-1], buses["right"].raw[-1] = left, right
        return ""

    monkeypatch.setattr("builtins.input", position_grippers)
    lerobot_calibrate.calibrate.__wrapped__(lerobot_calibrate.CalibrateConfig(robot=bot.config))
    assert bot.is_calibrated and not bot.is_connected
    assert bot.calibration_fpath.is_file()
    assert not any(call[0] in ("write", "enable", "disable") for bus in buses.values() for call in bus.calls)
    for arm in bot.arms.values():
        arm.config.gripper_closed_rad = arm.config.gripper_open_rad = None
    reloaded = BiYamFollower(bot.config)
    for side, expected in (("left", (0.1, -5.1)), ("right", (-0.02, 5.8))):
        cfg = reloaded.arms[side].config
        assert (cfg.gripper_closed_rad, cfg.gripper_open_rad) == pytest.approx(expected, abs=0.0002)
        assert decode_positions(cfg, states(buses[side].raw))[-1] == pytest.approx(1, abs=0.0001)


def test_calibration_session_rejects_actions_even_if_previously_calibrated(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.connect(calibrate=False)
    try:
        with pytest.raises(RuntimeError, match="calibration"):
            bot.send_action(dict.fromkeys(YAM_FEATURE_NAMES, 0.0))
        with pytest.raises(RuntimeError, match="Reconnect"):
            bot.get_observation()
    finally:
        bot.disconnect()
    assert not any(call[0] in ("write", "enable", "disable") for bus in buses.values() for call in bus.calls)


def test_failed_calibration_preserves_previous_file_and_endpoints(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path)
    bot.calibration_fpath.write_text("previous calibration")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    # No actual travel on either gripper: a cancelled/incorrect manual procedure.
    monkeypatch.setattr("builtins.input", lambda _: "")
    bot.connect(calibrate=False)
    try:
        with pytest.raises(ValueError, match="stroke"):
            bot.calibrate()
        assert bot.calibration_fpath.read_text() == "previous calibration"
        assert bot.config.left_arm.gripper_closed_rad == 0.1
        assert bot.config.left_arm.gripper_open_rad == 6.6
    finally:
        bot.disconnect()
    assert not any(call[0] in ("write", "enable", "disable") for bus in buses.values() for call in bus.calls)


def test_delayed_feedback_never_produces_motor_command(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    arm = bot.arms["left"]
    arm.enabled = True
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    def delayed_read(strict=False):
        now[0] = bot.config.feedback_timeout_s + 0.01
        return states(buses["left"].raw)

    monkeypatch.setattr(arm.bus, "sync_read_all_states", delayed_read)
    bot._run(arm)
    assert bot._stop.is_set()
    assert "freshness deadline" in str(bot._failure)
    assert not any(call[0] == "write" for call in arm.bus.calls)
    assert ("disable",) in arm.bus.calls


@pytest.mark.parametrize("second_error", [None, TimeoutError("second timeout")])
def test_camera_timeout_reopens_once_before_any_motor_connection(monkeypatch, tmp_path, second_error):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    calls = []

    class Camera:
        def __init__(self, error=None):
            self.error = error
            self.is_connected = False

        def connect(self):
            assert not any(bus.calls for bus in buses.values())
            calls.append(("connect", self))
            self.is_connected = True
            if self.error:
                raise self.error

        def disconnect(self):
            calls.append(("disconnect", self))
            self.is_connected = False

    first = Camera(TimeoutError("no first frame"))
    replacement = Camera(second_error)
    bot.cameras = {"left": first}
    bot.config.cameras = {"left": SimpleNamespace()}
    monkeypatch.setattr(module, "make_cameras_from_configs", lambda cfg: {"left": replacement})
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    if second_error:
        with pytest.raises(TimeoutError, match="second timeout"):
            bot.connect()
        assert not any(bus.calls for bus in buses.values())
        assert calls == [
            ("connect", first),
            ("disconnect", first),
            ("connect", replacement),
            ("disconnect", replacement),
        ]
    else:
        bot.connect()
        try:
            assert bot.cameras["left"] is replacement
            assert calls == [("connect", first), ("disconnect", first), ("connect", replacement)]
        finally:
            bot.disconnect()
        assert not replacement.is_connected


def test_camera_configuration_error_is_not_retried(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)

    def fail():
        raise ValueError("unsupported resolution")

    bot.cameras = {"left": SimpleNamespace(connect=fail, is_connected=False)}

    def unexpected_retry(config):
        pytest.fail("Configuration failures must not retry")

    monkeypatch.setattr(module, "make_cameras_from_configs", unexpected_retry)
    with pytest.raises(ValueError, match="unsupported resolution"):
        bot.connect()
    assert not any(bus.calls for bus in buses.values())
