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
from lerobot.robots.asimov.asimov import ASIMOV_JOINTS, BASE_ACTIONS
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError


@dataclass
class Velocity:
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0


@pytest.fixture
def setup_robot(monkeypatch, tmp_path):
    state = SimpleNamespace(
        joints=tuple(SimpleNamespace(name=name, pos=i / 100) for i, name in enumerate(ASIMOV_JOINTS)),
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
        Mode=SimpleNamespace(MOVE=2),
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
        config = AsimovConfig(host="192.0.2.10", calibration_dir=tmp_path, **kwargs)
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
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        AsimovConfig(**{"host": "192.0.2.10", **kwargs})


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
