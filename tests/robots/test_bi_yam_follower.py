"""Unit/frame and lifecycle checks; no real robot is connected by this suite."""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

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


@pytest.fixture(autouse=True)
def control_gc(monkeypatch):
    """Exercise ownership without changing the test runner's global GC state."""
    fake = MagicMock()
    fake.isenabled.return_value = True
    fake.get_freeze_count.return_value = 0
    monkeypatch.setattr(module, "gc", fake)
    yield fake
    assert module._ControlGC._users == 0
    assert not module._ControlGC._owns_freeze


def test_control_gc_shared_ownership(control_gc):
    module._ControlGC.acquire()
    module._ControlGC.acquire()
    control_gc.collect.assert_called_once()
    control_gc.freeze.assert_called_once()
    module._ControlGC.release()
    control_gc.unfreeze.assert_not_called()
    module._ControlGC.release()
    control_gc.unfreeze.assert_called_once()


@pytest.mark.parametrize("enabled,frozen", [(False, 0), (True, 10)])
def test_control_gc_preserves_caller_state(control_gc, enabled, frozen):
    control_gc.isenabled.return_value = enabled
    control_gc.get_freeze_count.return_value = frozen
    module._ControlGC.acquire()
    module._ControlGC.release()
    if enabled:
        # A small preexisting startup freeze does not cover the loaded model.
        control_gc.collect.assert_called_once()
        control_gc.freeze.assert_called_once()
    else:
        control_gc.collect.assert_not_called()
        control_gc.freeze.assert_not_called()
    control_gc.unfreeze.assert_not_called()


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


@pytest.mark.parametrize(
    "outcome,timeout_s",
    [
        ("arrive", 7.0),
        ("arrive_slowly", 60.0),
        ("stuck", 7.0),
        ("stuck", 60.0),
        ("fault", 7.0),
        ("stale", 7.0),
    ],
)
def test_return_waits_for_measured_arrival(monkeypatch, tmp_path, outcome, timeout_s):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.config.return_timeout_s = timeout_s
    bot._connected = True
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    for arm in bot.arms.values():
        arm.position[0] = 1.5  # Five seconds at the configured 0.3 rad/s.
        arm.position[6] = 0.0
    target = dict.fromkeys(YAM_FEATURE_NAMES, 0.0)
    target["left_gripper.pos"] = target["right_gripper.pos"] = 1.0

    def tick(dt):
        clock[0] += dt
        for arm in bot.arms.values():
            # The final target must be refreshed throughout settling.
            assert clock[0] - arm.commanded_at < bot.config.command_timeout_s
            if outcome != "stale":
                arm.updated_at = clock[0]
            if outcome in ("arrive", "arrive_slowly"):
                # Slow measured progress can take longer than distance / target speed.
                joint_speed = arm.config.max_joint_speed_rad_s * (0.1 if outcome == "arrive_slowly" else 1)
                arm.position[:6] += np.clip(
                    arm.target[:6] - arm.position[:6],
                    -joint_speed * dt,
                    joint_speed * dt,
                )
                arm.position[6] += np.clip(
                    arm.target[6] - arm.position[6],
                    -arm.config.max_gripper_speed_s * dt,
                    arm.config.max_gripper_speed_s * dt,
                )
        if outcome == "fault":
            bot._failure = ConnectionError("motor fault")

    monkeypatch.setattr(bot._stop, "wait", tick)
    try:
        if outcome in ("arrive", "arrive_slowly"):
            bot.wait_until_reached(target)
            assert (45.0 if outcome == "arrive_slowly" else 5.0) < clock[0] < timeout_s
            for arm in bot.arms.values():
                np.testing.assert_allclose(arm.position, [0, 0, 0, 0, 0, 0, 1], atol=0.03)
        else:
            error = TimeoutError if outcome == "stuck" else ConnectionError
            with pytest.raises(error):
                bot.wait_until_reached(target)
            if outcome == "stuck":
                assert timeout_s <= clock[0] < timeout_s + 0.1
            else:
                assert clock[0] < 0.2  # Fault handling is not delayed by the return timeout.
            for arm in bot.arms.values():
                np.testing.assert_array_equal(arm.target, arm.position)
                np.testing.assert_array_equal(arm.command, arm.position)
        # This unit test only simulates feedback; no bus write is performed.
        assert all(not bus.calls for bus in buses.values())
    finally:
        bot._connected = False


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("nan"), float("inf")])
def test_return_timeout_must_be_finite_and_positive(timeout_s):
    with pytest.raises(ValueError, match="return_timeout_s"):
        BiYamFollowerConfig(return_timeout_s=timeout_s)


def test_return_read_only_forbids_target_updates(monkeypatch, tmp_path):
    bot, _ = robot(monkeypatch, tmp_path)
    bot._connected = True
    try:
        with pytest.raises(RuntimeError, match="read_only"):
            bot.wait_until_reached(dict.fromkeys(YAM_FEATURE_NAMES, 0.0))
    finally:
        bot._connected = False


def test_read_only_never_enables_or_commands(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path)
    bot.connect()
    try:
        assert list(bot.action_features) == list(YAM_FEATURE_NAMES)
        assert list(bot.get_observation()) == list(YAM_FEATURE_NAMES)
        with pytest.raises(RuntimeError, match="read_only"):
            bot.send_action(dict.fromkeys(YAM_FEATURE_NAMES, 0.0))
        with pytest.raises(RuntimeError, match="forbids torque enable"):
            bot.start_control()
    finally:
        bot.disconnect()
    for bus in buses.values():
        assert bus.calls == [("connect", False), ("disconnect", False)]


def test_deferred_control_never_energizes_until_start(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.config.defer_torque_enable = True
    bot.connect()
    try:
        module.time.sleep(0.03)
        assert not bot.is_control_enabled
        assert all(bus.calls == [("connect", False)] for bus in buses.values())
        with pytest.raises(RuntimeError, match="use /start"):
            bot.send_action(dict.fromkeys(YAM_FEATURE_NAMES, 0.0))
        bot.start_control()
        assert bot.is_control_enabled
        for bus in buses.values():
            assert bus.calls[1][0] == "write"
            assert all(packet[:2] == (0.0, 0.0) for packet in bus.calls[1][1].values())
            assert bus.calls[2] == ("enable",)
        bot.start_control()
        assert all(bus.calls.count(("enable",)) == 1 for bus in buses.values())
    finally:
        bot.disconnect()


def test_deferred_start_rechecks_both_poses(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.config.defer_torque_enable = True
    bot.connect()
    try:
        buses["right"].raw[0] = 1.0
        deadline = module.time.monotonic() + 1.0
        while bot.get_observation()["right_joint_0.pos"] < 0.9:
            assert module.time.monotonic() < deadline
            module.time.sleep(0.005)
        with pytest.raises(ValueError, match="initial pose"):
            bot.start_control()
        assert all(bus.calls == [("connect", False)] for bus in buses.values())
    finally:
        bot.disconnect()


def test_deferred_enable_failure_disables_both_arms(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.config.defer_torque_enable = True

    def fail_enable():
        raise ConnectionError("enable failed")

    monkeypatch.setattr(buses["right"], "enable_torque", fail_enable)
    bot.connect()
    try:
        with pytest.raises(ConnectionError):
            bot.start_control()
    finally:
        bot.disconnect()
    assert all(not arm.enabled for arm in bot.arms.values())
    assert ("disable",) in buses["right"].calls


def test_second_arm_failure_never_energizes_first(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False, buses={"left": Bus(), "right": Bus(fail=True)})
    with pytest.raises(ConnectionError):
        bot.connect()
    assert all(not b.is_connected for b in buses.values())
    assert all(all(c[0] not in ("enable", "write") for c in b.calls) for b in buses.values())
    assert not bot._gc_acquired
    module.gc.unfreeze.assert_called_once()


def test_stale_feedback_reports_channel_and_age(monkeypatch, tmp_path):
    bot, _ = robot(monkeypatch, tmp_path)
    monkeypatch.setattr(module.time, "monotonic", lambda: 0.121)
    with pytest.raises(ConnectionError, match=r"left: 121.0 ms.*right: 121.0 ms.*deadline 100.0 ms"):
        bot._check_feedback()
    assert bot._stop.is_set()


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


def stop_with_timeout(bot):
    bot._stop.set()
    for arm in bot.arms.values():
        arm.thread.join(timeout=2)
        assert not arm.thread.is_alive()
    bot._failure = module._FeedbackTimeoutError("injected scheduling stall")


def test_return_recovers_timeout_with_fresh_feedback_without_cameras(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.config.recover_on_feedback_timeout = True
    bot.connect()
    try:
        target = bot.get_observation()
        stop_with_timeout(bot)
        assert bot.has_started_control and not bot.is_control_enabled
        # Motor feedback changes while torque is off. Reactivation must seed the
        # newly measured pose, never the old policy target or stale snapshot.
        for bus in buses.values():
            bus.raw[0] = 0.1
            bus.calls.clear()
        bot.cameras["failed"] = MagicMock()
        bot.cameras["failed"].async_read.side_effect = TimeoutError("camera unavailable")
        monkeypatch.setattr(bot, "wait_until_reached", MagicMock())
        assert bot.return_to_position(target)
        assert bot.is_control_enabled
        bot.wait_until_reached.assert_called_once_with(target)
        bot.cameras["failed"].async_read.assert_not_called()
        for bus in buses.values():
            assert bus.calls[0][0] == "write"
            assert bus.calls[0][1]["joint_0"][2] == pytest.approx(math.degrees(0.1))
            assert bus.calls[1] == ("enable",)
    finally:
        bot.disconnect()


@pytest.mark.parametrize("failure", ["missing", "fault", "invalid", "disabled", "disable_failed"])
def test_return_recovery_never_enables_on_unhealthy_feedback(monkeypatch, tmp_path, failure):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.config.recover_on_feedback_timeout = failure != "disabled"
    bot.connect()
    try:
        target = bot.get_observation()
        stop_with_timeout(bot)
        for bus in buses.values():
            bus.calls.clear()
        if failure == "missing":
            buses["right"].fail = True
        elif failure == "fault":
            bot._failure = ConnectionError("motor fault status 0xD")
        elif failure == "invalid":
            buses["right"].raw[0] = float("nan")
        elif failure == "disable_failed":
            bot._failure = RuntimeError("disable failed")
        with pytest.raises((ConnectionError, ValueError)):
            bot.return_to_position(target)
        assert all(not any(c[0] == "enable" for c in b.calls) for b in buses.values())
    finally:
        bot.disconnect()


def test_return_does_not_enable_before_first_start(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.config.defer_torque_enable = True
    bot.config.recover_on_feedback_timeout = True
    bot.connect()
    try:
        target = bot.get_observation()
        assert not bot.has_started_control
        with pytest.raises(RuntimeError, match="previously activated"):
            bot.return_to_position(target)
        assert all(not any(c[0] == "enable" for c in b.calls) for b in buses.values())
        # Default initial_gripper_position=None accepts closed or partly open.
        bot.start_control()
        assert bot.has_started_control
    finally:
        bot.disconnect()


def test_optional_initial_pose_seeds_current_joints_and_grippers(monkeypatch, tmp_path):
    import time

    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot.config.defer_torque_enable = True
    for side, arm in bot.arms.items():
        arm.config.initial_position_rad = None
        buses[side].raw = np.array([0.4, 0.8, 0.7, -0.3, 0.2, 0.1, 3.35])
    bot.connect()
    try:
        assert not any(c[0] in ("enable", "write") for b in buses.values() for c in b.calls)
        # Supported arms can change pose during the initial torque-off wait.
        for bus in buses.values():
            bus.raw[0] = 0.6
        deadline = time.monotonic() + 1
        while any(abs(a.position[0] - 0.6) > 1e-6 for a in bot.arms.values()):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        bot.start_control()
        for bus in buses.values():
            packet = next(c[1] for c in bus.calls if c[0] == "write")
            assert packet["joint_0"][2] == pytest.approx(math.degrees(0.6))
            assert packet["gripper"][2] == pytest.approx(math.degrees(3.35))
    finally:
        bot.disconnect()


@pytest.mark.parametrize("bad_index,bad_value", [(0, 8.0), (6, 9.0), (2, float("nan"))])
def test_optional_initial_pose_keeps_feedback_limits(monkeypatch, tmp_path, bad_index, bad_value):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    for arm in bot.arms.values():
        arm.config.initial_position_rad = None
    buses["right"].raw[bad_index] = bad_value
    with pytest.raises(ConnectionError if math.isnan(bad_value) else ValueError):
        bot.connect()
    assert not any(c[0] in ("enable", "write") for b in buses.values() for c in b.calls)


def test_initial_pose_null_config_decode():
    import draccus

    cfg = draccus.decode(YamArmConfig, {"port": "can0", "initial_position_rad": None})
    assert cfg.initial_position_rad is None
    for invalid in ([0.0] * 5, [float("nan")] * 6):
        with pytest.raises(ValueError):
            YamArmConfig(port="can0", initial_position_rad=invalid)


def test_expired_commands_latch_hold_pose_until_next_command(monkeypatch, tmp_path):
    bot, buses = robot(monkeypatch, tmp_path, read_only=False)
    bot._connected = True
    arm = bot.arms["left"]
    arm.enabled = True
    arm.target[0] = 0.5  # An old target must be abandoned on expiry.
    clock = [2.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    packets = []

    def write(packet):
        packets.append(packet)

    def wait(_):
        if len(packets) == 4:
            bot._stop.set()
            return
        clock[0] += 0.005
        buses["left"].raw[0] = [0.0, 0.1, 0.12, 0.14][len(packets)]
        if len(packets) == 2:
            # A new command re-arms the timeout, so its later expiry captures
            # the new pose once instead of retaining the previous hold target.
            for other in bot.arms.values():
                other.updated_at = clock[0]
            action = dict.fromkeys(YAM_FEATURE_NAMES, 0.0)
            action["left_joint_0.pos"] = 0.2
            bot.send_action(action)
            clock[0] += 2.0

    monkeypatch.setattr(arm.bus, "sync_write_mit", write)
    monkeypatch.setattr(bot._stop, "wait", wait)
    bot._run(arm)
    assert bot._failure is None
    goals = [math.radians(p["joint_0"][2]) for p in packets]
    assert goals == pytest.approx([0.0, 0.0, 0.12, 0.12])
