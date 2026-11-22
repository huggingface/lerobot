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

"""Exercise the robot contract without a robot or the optional Menlo SDK."""

import time
from dataclasses import dataclass
from math import radians
from types import SimpleNamespace
from unittest.mock import Mock

import draccus
import numpy as np
import pytest

from lerobot.cameras import DepthCamera
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.robots.asimov import Asimov, AsimovConfig, asimov as module
from lerobot.robots.asimov.asimov import ARM_ACTIONS, ASIMOV_JOINTS, BASE_ACTIONS
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


@dataclass
class Velocity:
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0


@pytest.fixture
def setup_robot(monkeypatch, tmp_path):
    state = SimpleNamespace(
        joints=tuple(
            # Ankles at zero: a pose inside the firmware's ankle limits, which trajectory() enforces.
            SimpleNamespace(name=name, pos=0.0 if "Ankle" in name else i / 100, vel=0.0)
            for i, name in enumerate(ASIMOV_JOINTS)
        ),
        gravity=(0.0, 0.0, -1.0),
        gyro=(0.1, 0.2, 0.3),
        quat=(1.0, 0.0, 0.0, 0.0),
        age_s=0.01,
        mode=2,
        faulted=False,
    )
    client = Mock(connected=True)
    client.get_state.return_value = state
    client.info.joint_names = ASIMOV_JOINTS
    sdk = SimpleNamespace(
        Robot=Mock(return_value=client),
        ConnectionConfig=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)),
        UdpConfig=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)),
        Limits=Velocity,
        Velocity=Velocity,
        Mode=SimpleNamespace(DAMP=0, STAND=1, MOVE=2, FAULT_DAMP=5),
    )

    def set_velocity(vx, vy, vyaw, *, duration, wait, timeout):
        assert duration > 0 and wait is False and timeout == 0.0
        limits = sdk.Robot.call_args.kwargs["limits"]
        command = Velocity(
            *(
                max(-limit, min(limit, value))
                for value, limit in zip((vx, vy, vyaw), (limits.vx, limits.vy, limits.vyaw), strict=True)
            )
        )
        return SimpleNamespace(command=command)

    client.set_velocity.side_effect = set_velocity
    monkeypatch.setattr(module, "menlo", sdk)
    monkeypatch.setattr(module, "require_package", Mock())
    cameras = {}
    monkeypatch.setattr(module, "make_cameras_from_configs", lambda configs: cameras)
    robots = []

    def make(**kwargs):
        config = AsimovConfig(**{"host": "192.0.2.10", "calibration_dir": tmp_path, **kwargs})
        robot = make_robot_from_config(config)
        robots.append(robot)
        return robot

    yield SimpleNamespace(make=make, client=client, sdk=sdk, state=state, cameras=cameras)
    for robot in robots:
        # Tests explicitly exercise failures; remove those before fixture teardown.
        client.close.side_effect = None
        for camera in cameras.values():
            camera.disconnect.side_effect = None
        robot.disconnect()


def test_features_and_factory_before_connection(setup_robot):
    env = setup_robot
    robot = env.make()
    assert isinstance(robot, Asimov)
    assert robot.config.type == "asimov"
    assert set(robot.action_features) == set(BASE_ACTIONS)
    assert len(robot.observation_features) == 37
    assert "x.vel" not in robot.observation_features  # Firmware does not report odometry.
    assert robot.is_calibrated and not robot.is_connected
    env.sdk.Robot.assert_not_called()


def test_connection_observes_without_changing_mode(setup_robot):
    env = setup_robot
    env.state.mode = 0  # DAMP can be observed, but not driven.
    robot = env.make()
    robot.connect()
    assert robot.is_connected
    env.client.connect.assert_called_once_with("udp", timeout=5.0, persist=False)
    env.sdk.UdpConfig.assert_called_once_with(
        host="192.0.2.10",
        command_port=8850,
        state_bind=("0.0.0.0", 8851),
        state_source="192.0.2.10",
        version_port=3000,
        version_timeout=5.0,
    )
    env.client.stand.assert_not_called()
    env.client.damp.assert_not_called()
    env.client.set_velocity.assert_not_called()
    with pytest.raises(DeviceAlreadyConnectedError):
        robot.connect()


def test_observation_contains_measured_state_and_camera(setup_robot):
    env = setup_robot
    camera = Mock(is_connected=False)
    camera.connect.side_effect = lambda: setattr(camera, "is_connected", True)
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    camera.read_latest.return_value = image
    env.cameras["front"] = camera
    robot = env.make(cameras={"front": SimpleNamespace(width=48, height=32, fps=30)})
    robot.connect()
    observation = robot.get_observation()
    assert set(observation) == set(robot.observation_features)
    assert observation["Neck_Pitch.pos"] == pytest.approx(0.24)
    assert observation["imu.gravity.z"] == -1.0
    assert observation["imu.quat.w"] == 1.0
    assert observation["mode"] == 2.0
    assert observation["front"] is image
    assert robot.observation_features["front"] == image.shape


def test_actions_convert_units_clip_and_return_sent_values(setup_robot):
    env = setup_robot
    robot = env.make(max_angular_velocity=40.0)
    robot.connect()
    action = {"x.vel": 0.9, "y.vel": -0.8, "theta.vel": 90.0}
    sent = robot.send_action(action)
    assert sent == pytest.approx({"x.vel": 0.3, "y.vel": -0.2, "theta.vel": 40.0})
    env.client.set_velocity.assert_called_once_with(
        vx=0.9,
        vy=-0.8,
        vyaw=radians(90),
        duration=0.25,
        wait=False,
        timeout=0.0,
    )
    assert action["theta.vel"] == 90.0
    assert robot.send_action({"x.vel": -0.1}) == {"x.vel": -0.1, "y.vel": 0.0, "theta.vel": 0.0}
    assert robot.send_action({}) == dict.fromkeys(BASE_ACTIONS, 0.0)


def test_depth_only_camera_matches_features(setup_robot):
    env = setup_robot
    camera = Mock(
        spec=DepthCamera,
        is_connected=False,
        use_rgb=False,
        use_depth=True,
        connect=Mock(),
        disconnect=Mock(),
        read_latest=Mock(),
    )
    camera.connect.side_effect = lambda: setattr(camera, "is_connected", True)
    depth = np.zeros((32, 48, 1), dtype=np.float32)
    camera.read_latest_depth.return_value = depth
    env.cameras["front"] = camera
    robot = env.make(
        cameras={
            "front": SimpleNamespace(width=48, height=32, fps=30, use_rgb=False, use_depth=True),
        }
    )
    robot.connect()
    observation = robot.get_observation()
    assert set(observation) == set(robot.observation_features)
    assert observation["front_depth"] is depth
    assert "front" not in observation
    camera.read_latest.assert_not_called()


def test_camera_cannot_overwrite_proprioception(setup_robot):
    with pytest.raises(ValueError, match="collides"):
        setup_robot.make(cameras={"mode": SimpleNamespace(width=48, height=32, fps=30)})


@pytest.mark.parametrize("action", [{"theta.vel": float("nan")}, {"x.vel": float("inf")}, {"joint.pos": 0.0}])
def test_invalid_actions_never_reach_sdk(setup_robot, action):
    env = setup_robot
    robot = env.make()
    robot.connect()
    with pytest.raises(ValueError):
        robot.send_action(action)
    env.client.set_velocity.assert_not_called()


@pytest.mark.parametrize("mode,faulted", [(0, False), (1, False), (5, True), (2, True), (-1, False)])
def test_motion_requires_move_and_no_faults(setup_robot, mode, faulted):
    env = setup_robot
    robot = env.make()
    robot.connect()
    env.state.mode, env.state.faulted = mode, faulted
    with pytest.raises(RuntimeError, match="MOVE"):
        robot.send_action({"x.vel": 0.1})
    env.client.set_velocity.assert_not_called()


@pytest.mark.parametrize("age", [0.51, float("inf"), float("nan")])
def test_stale_state_rejected_for_observation_and_action(setup_robot, age):
    env = setup_robot
    robot = env.make()
    robot.connect()
    env.state.age_s = age
    with pytest.raises(RuntimeError, match="stale"):
        robot.get_observation()
    with pytest.raises(RuntimeError, match="stale"):
        robot.send_action({"x.vel": 0.1})
    env.client.set_velocity.assert_not_called()


@pytest.mark.parametrize(
    "field,value", [("gyro", None), ("quat", (1.0,)), ("gravity", (0.0, 0.0, float("nan"))), ("joints", ())]
)
def test_incomplete_telemetry_is_not_fabricated(setup_robot, field, value):
    env = setup_robot
    robot = env.make()
    robot.connect()
    setattr(env.state, field, value)
    with pytest.raises(ValueError):
        robot.get_observation()


def test_disconnect_after_link_loss_still_releases_resources(setup_robot):
    env = setup_robot
    robot = env.make()
    robot.connect()
    robot.send_action({"x.vel": 0.1})
    env.client.connected = False
    assert robot.is_connected  # Cleanup is still owed.
    with pytest.raises(DeviceNotConnectedError):
        robot.get_observation()
    with pytest.raises(DeviceNotConnectedError):
        robot.send_action({})
    robot.disconnect()
    robot.disconnect()
    env.client.close.assert_called_once()
    env.client.damp.assert_not_called()
    env.client.stand.assert_not_called()
    assert not robot.is_connected


def test_stop_keeps_balance_and_does_not_override_operator_mode(setup_robot):
    env = setup_robot
    robot = env.make()
    robot.connect()
    robot.stop()
    env.client.balance.assert_not_called()
    robot.send_action({"x.vel": 0.1})
    robot.stop()
    env.client.balance.assert_called_once_with(wait=False, timeout=0.0)
    robot.send_action({"x.vel": 0.1})
    env.state.mode = 1
    robot.stop()
    assert env.client.balance.call_count == 1


@pytest.mark.parametrize("failure", ["sdk", "body", "state", "camera"])
def test_connect_failure_closes_sdk_and_connected_cameras(setup_robot, failure):
    env = setup_robot
    good = Mock(is_connected=False)
    good.connect.side_effect = lambda: setattr(good, "is_connected", True)
    good.disconnect.side_effect = lambda: setattr(good, "is_connected", False)
    bad = Mock(is_connected=False)
    env.cameras.update(first=good, second=bad)
    if failure == "sdk":
        env.client.connect.side_effect = RuntimeError("connect failed")
    elif failure == "body":
        env.client.info.joint_names = ("other",)
    elif failure == "state":
        env.state.gyro = None
    else:
        bad.connect.side_effect = RuntimeError("camera failed")
    robot = env.make()
    with pytest.raises((RuntimeError, ValueError)):
        robot.connect()
    env.client.close.assert_called_once()
    if failure == "camera":
        good.disconnect.assert_called_once()
    assert not robot.is_connected


def test_cleanup_continues_after_camera_failure_and_can_be_retried(setup_robot):
    env = setup_robot
    first, second = Mock(is_connected=True), Mock(is_connected=True)
    robot = env.make()
    robot.connect()
    env.cameras.update(first=first, second=second)
    first.disconnect.side_effect = RuntimeError("disconnect failed")
    second.disconnect.side_effect = lambda: setattr(second, "is_connected", False)
    with pytest.raises(RuntimeError, match="disconnect failed"):
        robot.disconnect()
    env.client.close.assert_called_once()
    second.disconnect.assert_called_once()
    assert robot.is_connected
    first.disconnect.side_effect = lambda: setattr(first, "is_connected", False)
    robot.disconnect()
    assert not robot.is_connected


def make_arms_robot(env, **kwargs):
    robot = env.make(control_mode="arms", robot_supported=True, **kwargs)
    robot.connect()
    return robot


def test_arm_mode_features(setup_robot):
    robot = setup_robot.make(control_mode="arms", robot_supported=True)
    assert tuple(robot.action_features) == ARM_ACTIONS
    assert len(ARM_ACTIONS) == 10
    assert set(ARM_ACTIONS) <= set(robot.observation_features)


def test_arm_targets_ramp_from_measured_pose_and_hold_other_joints(setup_robot):
    env = setup_robot
    robot = make_arms_robot(env, max_arm_step_rad=0.1)
    measured = [joint.pos for joint in env.state.joints]
    elbow = ASIMOV_JOINTS.index("L_Elbow")

    sent = robot.send_action({"L_Elbow.pos": measured[elbow] + 1.0})
    (positions,), kwargs = env.client.trajectory.call_args
    assert kwargs == {"timeout": 0.0}
    expected = list(measured)
    expected[elbow] += 0.1
    assert positions == pytest.approx(expected)
    assert sent["L_Elbow.pos"] == pytest.approx(measured[elbow] + 0.1)
    assert set(sent) == set(ARM_ACTIONS)

    # Later commands step from the previous target, not from new measurements.
    env.state.joints = tuple(SimpleNamespace(name=j.name, pos=0.0) for j in env.state.joints)
    robot.send_action({"L_Elbow.pos": measured[elbow] + 1.0})
    (positions,), _ = env.client.trajectory.call_args
    assert positions[elbow] == pytest.approx(measured[elbow] + 0.2)
    assert positions[:12] == pytest.approx(measured[:12])  # Legs held.
    env.client.set_velocity.assert_not_called()


def test_arm_control_requires_stand_or_move_and_stops_after_damp(setup_robot):
    env = setup_robot
    robot = make_arms_robot(env)
    env.state.mode = 0
    with pytest.raises(RuntimeError, match="STAND or MOVE"):
        robot.send_action({"L_Elbow.pos": 0.0})
    env.client.trajectory.assert_not_called()

    env.state.mode = 1
    robot.send_action({"L_Elbow.pos": 0.0})
    env.state.mode = 0
    with pytest.raises(RuntimeError, match="DAMP"):
        robot.send_action({"L_Elbow.pos": 0.0})
    with pytest.raises(RuntimeError, match="STAND or MOVE"):
        robot.send_action({"L_Elbow.pos": 0.0})
    assert env.client.trajectory.call_count == 1


@pytest.mark.parametrize("action", [{"x.vel": 0.1}, {"L_Elbow.pos": float("nan")}])
def test_invalid_arm_actions_never_reach_sdk(setup_robot, action):
    robot = make_arms_robot(setup_robot)
    with pytest.raises(ValueError):
        robot.send_action(action)
    setup_robot.client.trajectory.assert_not_called()


def test_arm_stop_never_starts_walking_policy(setup_robot):
    env = setup_robot
    robot = make_arms_robot(env)
    robot.send_action({"L_Elbow.pos": 0.0})
    robot.stop()
    env.client.balance.assert_not_called()
    env.client.set_velocity.assert_not_called()
    # The next command restarts from the measured pose.
    robot.send_action({})
    (positions,), _ = env.client.trajectory.call_args
    assert positions == pytest.approx([joint.pos for joint in env.state.joints])


def test_disconnected_calls_fail(setup_robot):
    robot = setup_robot.make()
    with pytest.raises(DeviceNotConnectedError):
        robot.get_observation()
    with pytest.raises(DeviceNotConnectedError):
        robot.send_action({})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"host": ""},
        {"state_port": 0},
        {"command_port": 65536},
        {"version_port": 1.5},
        {"command_timeout_s": 0},
        {"connect_timeout_s": -1},
        {"max_state_age_s": float("nan")},
        {"max_linear_velocity": -1},
        {"max_lateral_velocity": 0.5},
        {"max_angular_velocity": 90},
        {"control_mode": "torque"},
        {"control_mode": "arms"},  # An unsupported robot would fall with the walking policy off.
        {"control_mode": "arms", "robot_supported": True, "max_arm_step_rad": 0},
        {"control_mode": "whole_body"},  # Not yet validated off the gantry.
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        AsimovConfig(**{"host": "192.0.2.10", **kwargs})


def test_simulation_config_targets_the_hub_sim():
    # On the sim, whole-body control needs no gantry.
    config = AsimovConfig(control_mode="whole_body", is_simulation=True)
    assert config.host == "127.0.0.1"
    assert config.sim_env.hub_path == "lerobot/asimov-mujoco"
    assert config.sim_env.trust_remote_code
    assert config.sim_env.onscreen and not config.sim_env.publish_images
    cameras = AsimovConfig(control_mode="whole_body", is_simulation=True, sim_publish_images=True)
    assert not cameras.sim_env.onscreen and cameras.sim_env.publish_images


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # The sim has no onboard walking policy.
        {"control_mode": "whole_body", "host": "192.0.2.10"},
        {"control_mode": "whole_body", "input_timeout_s": 0},
        {"control_mode": "whole_body", "sim_publish_images": True, "sim_onscreen": True},
    ],
)
def test_invalid_simulation_config(kwargs):
    with pytest.raises(ValueError):
        AsimovConfig(is_simulation=True, **kwargs)


def test_simulation_launches_the_sim_and_connects_to_it(setup_robot, monkeypatch):
    pytest.importorskip("menlo")
    import lerobot.envs
    from lerobot.controllers.asimov import locomotion

    monkeypatch.setattr(locomotion, "AsimovLocomotionController", FakeController)
    sim = Mock()
    make_env = Mock(return_value={"hub_env": {0: SimpleNamespace(envs=[sim])}})
    monkeypatch.setattr(lerobot.envs, "make_env", make_env)
    env = setup_robot
    env.state.mode = 1
    robot = env.make(host="", control_mode="whole_body", is_simulation=True)
    robot.connect()
    make_env.assert_called_once_with(robot.config.sim_env)
    sim.reset.assert_called_once()
    sim.simulator.start_as_thread.assert_called_once()
    assert env.sdk.UdpConfig.call_args.kwargs["host"] == "127.0.0.1"
    robot.disconnect()
    sim.close.assert_called_once()
    assert not robot.is_connected


class FakeController:
    control_dt = 0.001
    leg_defaults = np.full(12, 0.05)
    default_pose = np.concatenate([leg_defaults, np.full(len(ASIMOV_JOINTS) - 12, 0.2)])
    kp = np.ones(len(ASIMOV_JOINTS))
    kd = np.ones(len(ASIMOV_JOINTS))

    def __init__(self, repo_id):
        self.commands = []

    def reset(self):
        pass

    def run_step(self, q, dq, gyro, gravity, command, upper_body):
        self.commands.append(np.array(command))
        return upper_body


@pytest.fixture
def whole_body(setup_robot, monkeypatch):
    pytest.importorskip("menlo")
    from lerobot.controllers.asimov import locomotion

    monkeypatch.setattr(locomotion, "AsimovLocomotionController", FakeController)
    env = setup_robot
    env.state.mode = 1
    robot = env.make(control_mode="whole_body", robot_supported=True, controller_ease_in_s=0.005)
    robot.connect()
    deadline = time.monotonic() + 2.0
    while not robot._controller.commands and time.monotonic() < deadline:
        time.sleep(0.001)
    return robot


def test_ankle_conversion_matches_sdk_wire_format():
    pytest.importorskip("menlo")
    from menlo.asimov.robots import trajectory_wire_positions

    joints = np.zeros(len(ASIMOV_JOINTS))
    joints[[4, 5, 10, 11]] = (0.2, -0.05, -0.3, 0.08)
    motors = module.joints_to_motors(joints)
    assert trajectory_wire_positions(tuple(motors)) == pytest.approx(joints)
    assert module.motors_to_joints(motors) == pytest.approx(joints)


def test_whole_body_streams_policy_targets_with_policy_gains(whole_body, setup_robot):
    assert set(whole_body.action_features) == {*BASE_ACTIONS, *ARM_ACTIONS}
    (positions,), kwargs = setup_robot.client.trajectory.call_args
    assert kwargs["kp"] == pytest.approx(FakeController.kp) and kwargs["timeout"] == 0.0
    assert module.motors_to_joints(positions)[12:] == pytest.approx(FakeController.default_pose[12:])
    setup_robot.client.set_velocity.assert_not_called()


def test_whole_body_actions_only_update_controller_input(whole_body, setup_robot):
    elbow = ASIMOV_JOINTS.index("L_Elbow")
    sent = whole_body.send_action({"x.vel": 9.0, "theta.vel": -10.0, "L_Elbow.pos": 1.0})
    assert sent["x.vel"] == pytest.approx(whole_body.config.max_linear_velocity)
    assert sent["y.vel"] == 0.0 and sent["theta.vel"] == pytest.approx(-10.0)
    assert sent["L_Elbow.pos"] == pytest.approx(0.2 + whole_body.config.max_arm_step_rad)
    assert whole_body._upper_targets[elbow] == pytest.approx(sent["L_Elbow.pos"])
    controller = whole_body._controller
    n = len(controller.commands)
    deadline = time.monotonic() + 2.0
    while len(controller.commands) <= n + 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert controller.commands[-1] == pytest.approx([0.3, 0.0, radians(-10.0)])
    whole_body.stop()
    time.sleep(0.01)
    assert controller.commands[-1] == pytest.approx([0.0, 0.0, 0.0])


def test_whole_body_surfaces_damp_from_send_action(whole_body, setup_robot):
    setup_robot.state.mode = 0
    whole_body._controller_thread.join(timeout=2.0)
    with pytest.raises(RuntimeError, match="controller stopped"):
        whole_body.send_action({"x.vel": 0.1})


def test_cli_config_roundtrip():
    @dataclass
    class Config:
        robot: RobotConfig

    config = draccus.parse(Config, args=["--robot.type=asimov", "--robot.host=192.0.2.10"])
    assert isinstance(config.robot, AsimovConfig)
    assert draccus.decode(Config, draccus.encode(config)) == config


def test_missing_sdk_has_actionable_install_error(monkeypatch, tmp_path):
    from lerobot.utils import import_utils

    monkeypatch.setitem(import_utils._require_package_cache, "menlo", False)
    with pytest.raises(ImportError, match="asimov"):
        Asimov(AsimovConfig(host="192.0.2.10", calibration_dir=tmp_path))


# ---------------------------------------------------------------- arms: running a learned arm policy

STANCE = {
    "L_Hip_Pitch": -0.5,
    "L_Knee": 0.55,
    "L_Ankle_Pitch": -0.21,
    "R_Hip_Pitch": 0.5,
    "R_Knee": -0.55,
    "R_Ankle_Pitch": 0.21,
}
FREE_STANDING = {
    "control_mode": "arms",
    "free_standing": True,
    "exit_mode": "balance",
    "start_pose": STANCE,
    "start_pose_s": 0.02,
    "exit_glide_s": 0.02,
    "arm_loop_hz": 200.0,
}


@pytest.fixture
def fast_handover(monkeypatch):
    monkeypatch.setattr(module, "HANDOVER_CHECK_S", 0.0)


def wait_for(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.002)
    return condition()


def tilt(state, degrees_):
    state.gravity = (np.sin(np.radians(degrees_)), 0.0, -np.cos(np.radians(degrees_)))


def test_setpoint_loop_keeps_targets_alive_between_actions(setup_robot):
    env = setup_robot
    robot = make_arms_robot(env, arm_loop_hz=100.0)
    sent = robot.send_action({"L_Elbow.pos": 0.0})
    calls = env.client.trajectory.call_count
    # A caller that pauses (a reset phase, a prompt) must not let Asimov Edge DAMP.
    assert wait_for(lambda: env.client.trajectory.call_count >= calls + 3)
    (positions,), _ = env.client.trajectory.call_args
    assert positions[ASIMOV_JOINTS.index("L_Elbow")] == pytest.approx(sent["L_Elbow.pos"])
    robot.stop()
    calls = env.client.trajectory.call_count
    time.sleep(0.05)
    assert env.client.trajectory.call_count == calls  # Released: no setpoint after stop().
    env.client.balance.assert_not_called()


def test_setpoint_loop_reports_damp_to_the_next_action(setup_robot):
    env = setup_robot
    robot = make_arms_robot(env, arm_loop_hz=200.0)
    robot.send_action({})
    env.state.mode = 0
    assert wait_for(lambda: robot._joint_targets is None)
    with pytest.raises(RuntimeError, match="DAMP"):
        robot.send_action({})
    env.client.stand.assert_not_called()


def test_gains_are_sent_with_every_setpoint(setup_robot):
    env = setup_robot
    kp, kd = [float(i + 1) for i in range(25)], [0.5] * 25
    robot = make_arms_robot(env, kp=kp, kd=kd)
    robot.send_action({"R_Elbow.pos": 0.0})
    _, kwargs = env.client.trajectory.call_args
    assert kwargs == {"timeout": 0.0, "kp": kp, "kd": kd}


def test_start_pose_is_eased_into_on_connect(setup_robot):
    pytest.importorskip("menlo")
    env = setup_robot
    robot = env.make(
        control_mode="arms",
        robot_supported=True,
        start_pose={**STANCE, "Neck_Pitch": 0.3},
        start_pose_s=0.05,
        arm_loop_hz=100.0,
    )
    robot.connect()
    assert env.client.trajectory.call_count >= 5  # Eased, not snapped.
    (first,), _ = env.client.trajectory.call_args_list[0]
    (last,), _ = env.client.trajectory.call_args_list[-1]
    joints = module.motors_to_joints(last)
    for name in ("L_Hip_Pitch", "L_Knee", "R_Hip_Pitch", "R_Knee", "Neck_Pitch"):
        assert joints[ASIMOV_JOINTS.index(name)] == pytest.approx({**STANCE, "Neck_Pitch": 0.3}[name])
    assert joints[ASIMOV_JOINTS.index("L_Ankle_A")] == pytest.approx(-0.21)  # Ankle pitch, joint space.
    assert joints[ASIMOV_JOINTS.index("L_Ankle_B")] == pytest.approx(0.0)  # Roll kept as measured.
    assert abs(first[0] - (-0.5)) > abs(last[0] - (-0.5))
    assert robot.get_observation()["L_Elbow.pos"] == pytest.approx(0.15)
    robot.disconnect()
    env.client.balance.assert_not_called()  # exit_mode defaults to release.


def test_balance_exit_glides_upper_body_back_then_stands_and_balances(setup_robot, fast_handover):
    pytest.importorskip("menlo")
    env = setup_robot
    robot = env.make(**FREE_STANDING)
    robot.connect()
    start = module.motors_to_joints([joint.pos for joint in env.state.joints])
    robot.send_action({"R_Elbow.pos": 1.0})
    env.client.reset_mock()
    robot.stop()
    names = [call[0] for call in env.client.mock_calls if call[0] in ("trajectory", "stand", "balance")]
    assert names[-2:] == ["stand", "balance"] and "trajectory" in names[:-2]
    (last,), _ = env.client.trajectory.call_args
    joints = module.motors_to_joints(last)
    assert joints[12:] == pytest.approx(start[12:])  # Arms, waist and neck back where they started.
    assert joints[ASIMOV_JOINTS.index("L_Hip_Pitch")] == pytest.approx(-0.5)  # Legs keep the stance.
    env.client.balance.assert_called_once_with(timeout=5.0)


def test_handover_retries_when_the_robot_does_not_stay_in_move(setup_robot, fast_handover):
    pytest.importorskip("menlo")
    env = setup_robot
    robot = env.make(**FREE_STANDING)
    robot.connect()
    env.client.balance.side_effect = lambda **_: setattr(env.state, "mode", 1)  # Stays in STAND.
    robot.disconnect()
    assert env.client.balance.call_count == 2 and env.client.stand.call_count == 2
    env.client.close.assert_called_once()


def test_lean_watchdog_hands_over_without_glide_and_refuses_further_actions(setup_robot, fast_handover):
    pytest.importorskip("menlo")
    env = setup_robot
    robot = env.make(**FREE_STANDING, max_extra_lean_deg=5.0)
    robot.connect()
    tilt(env.state, 4.0)
    robot.send_action({})  # Within the margin.
    calls = env.client.trajectory.call_count
    tilt(env.state, 7.0)
    assert wait_for(lambda: env.client.balance.called)
    assert env.client.trajectory.call_count == calls  # No glide while tipping.
    env.client.stand.assert_called_once()
    for _ in range(2):  # Handed to the walking policy: never taken back silently.
        with pytest.raises(RuntimeError, match="leans"):
            robot.send_action({})
    assert env.client.trajectory.call_count == calls


def test_no_handover_from_damp(setup_robot, fast_handover):
    pytest.importorskip("menlo")
    env = setup_robot
    robot = env.make(**FREE_STANDING)
    robot.connect()
    env.state.mode = 0
    assert wait_for(lambda: robot._joint_targets is None)
    env.client.stand.assert_not_called()
    env.client.balance.assert_not_called()


def test_joint_velocities_are_observed_on_request(setup_robot):
    env = setup_robot
    robot = env.make(observe_velocities=True)
    assert len(robot.observation_features) == 37 + 25
    robot.connect()
    observation = robot.get_observation()
    assert set(observation) == set(robot.observation_features)
    assert observation["R_Elbow.vel"] == 0.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"control_mode": "arms", "free_standing": True, "start_pose": STANCE}, "exit_mode=balance"),
        ({"control_mode": "arms", "free_standing": True, "exit_mode": "balance"}, "stance"),
        ({**FREE_STANDING, "max_extra_lean_deg": 0.0}, "watchdog"),
        ({"control_mode": "arms", "robot_supported": True, "kp": [1.0] * 25}, "both"),
        ({"control_mode": "arms", "robot_supported": True, "exit_mode": "damp"}, "exit_mode"),
        ({"control_mode": "arms", "robot_supported": True, "arm_loop_hz": 1.0}, "arm_loop_hz"),
        ({"start_pose": STANCE}, "only apply"),
        ({"exit_mode": "balance"}, "only apply"),
        ({"head_camera": True, "head_camera_width": 0}, "head_camera_width"),
        ({"head_camera": True, "head_camera_latency_s": -0.1}, "latency"),
    ],
)
def test_invalid_arm_policy_and_head_camera_config(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AsimovConfig(**{"host": "192.0.2.10", **kwargs})


def test_head_camera_is_not_simulated():
    with pytest.raises(ValueError, match="head camera"):
        AsimovConfig(control_mode="whole_body", is_simulation=True, head_camera=True)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"start_pose": {"L_Ankle_A": 0.1}}, "Unknown start_pose"),
        ({"kp": [1.0] * 24, "kd": [1.0] * 24}, "25"),
        ({"kp": [0.0] * 25, "kd": [1.0] * 25}, "positive"),
    ],
)
def test_invalid_arm_options_checked_against_the_joint_list(setup_robot, kwargs, match):
    with pytest.raises(ValueError, match=match):
        setup_robot.make(control_mode="arms", robot_supported=True, **kwargs)


# ---------------------------------------------------------------- head camera


def raw_frame(image, encoding="rgb8", age=0.02):
    return SimpleNamespace(encoding=encoding, to_numpy=lambda: image, age_s=age)


@pytest.fixture
def head_camera(setup_robot, monkeypatch):
    env = setup_robot
    monkeypatch.setenv("MENLO_CREDENTIAL", "test-credential")
    env.sdk.ManagerConfig = Mock(side_effect=lambda **kw: SimpleNamespace(**kw))
    image = np.zeros((72, 128, 3), dtype=np.uint8)
    image[..., 0] = 200  # Red in RGB.
    env.frame = raw_frame(image)
    env.client.camera.latest.side_effect = lambda: env.frame
    return env


def test_head_camera_connects_hybrid_and_observes_frame_age_and_count(head_camera):
    env = head_camera
    robot = env.make(head_camera=True, head_camera_width=64, head_camera_height=36, head_camera_latency_s=0.1)
    assert robot.observation_features["head"] == (36, 64, 3)
    robot.connect()
    env.client.connect.assert_called_once_with("hybrid", timeout=5.0, media_timeout=5.0, persist=False)
    livekit = env.sdk.ConnectionConfig.call_args.kwargs["livekit"]
    assert livekit.url == "http://192.0.2.10" and livekit.credential == "test-credential"
    env.client.camera.photo.assert_called_once_with(timeout=5.0)

    first = robot.get_observation()
    assert set(first) == set(robot.observation_features)
    assert first["head"].shape == (36, 64, 3) and first["head"][0, 0, 0] == 200
    assert first["head.age_s"] == pytest.approx(0.12)
    assert first["head.frame"] == 1.0
    assert robot.get_observation()["head.frame"] == 1.0  # Same frame: same count.
    bgr = np.zeros((36, 64, 3), dtype=np.uint8)
    bgr[..., 2] = 50  # Red in BGR.
    env.frame = raw_frame(bgr, encoding="bgr8", age=0.0)
    second = robot.get_observation()
    assert second["head.frame"] == 2.0 and second["head"][0, 0, 0] == 50


def test_head_camera_uses_manager_url(head_camera):
    robot = head_camera.make(head_camera=True, manager_url="http://asimov.local:8080")
    robot.connect()
    assert head_camera.sdk.ConnectionConfig.call_args.kwargs["livekit"].url == "http://asimov.local:8080"


def test_head_camera_needs_credential_from_environment(head_camera, monkeypatch):
    monkeypatch.delenv("MENLO_CREDENTIAL")
    robot = head_camera.make(head_camera=True)
    with pytest.raises(RuntimeError, match="MENLO_CREDENTIAL"):
        robot.connect()
    assert not robot.is_connected


def test_head_camera_collides_with_a_camera_named_head(setup_robot):
    with pytest.raises(ValueError, match="collides"):
        setup_robot.make(head_camera=True, cameras={"head": SimpleNamespace(width=48, height=32, fps=30)})


def test_jpeg_and_gray_frames_decode_to_rgb():
    import cv2

    rgb = np.zeros((8, 16, 3), dtype=np.uint8)
    rgb[..., 1] = 255
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    decoded = module.frame_to_rgb(SimpleNamespace(encoding="jpeg", data=jpeg.tobytes()), 16, 8)
    assert decoded.shape == (8, 16, 3) and decoded[4, 8, 1] > 240 and decoded[4, 8, 0] < 15
    gray = module.frame_to_rgb(raw_frame(np.full((8, 16), 7, dtype=np.uint8), encoding="gray8"), 16, 8)
    assert gray.shape == (8, 16, 3) and (gray == 7).all()
    with pytest.raises(ValueError, match="h264"):
        module.frame_to_rgb(SimpleNamespace(encoding="h264"), 16, 8)


def test_lean_from_gravity():
    assert module.lean_deg((0.0, 0.0, -1.0)) == pytest.approx(0.0)
    assert module.lean_deg((0.20, -0.01, -0.98)) == pytest.approx(11.5, abs=0.1)
