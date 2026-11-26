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

import numpy as np
import pytest

from lerobot.robots.metal_follower.config_metal_follower import METAL_FEATURE_NAMES
from lerobot.robots.yam_follower import mit_control
from lerobot.teleoperators.bi_metal_leader import BiMetalLeader, BiMetalLeaderConfig
from lerobot.teleoperators.metal_leader import MetalLeader, MetalLeaderConfig, MetalLeaderTeleopConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config
from tests.mocks.mock_metal import make_ready, mock_hardware, without_hardware_packages


@pytest.fixture(autouse=True)
def no_hardware_packages(monkeypatch):
    without_hardware_packages(monkeypatch)


@pytest.fixture
def leader(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.robots.metal_follower.metal_follower.hf_hub_download", lambda *a, **kw: "x")
    monkeypatch.setattr(
        "lerobot.robots.metal_follower.metal_follower.GravityCompensation", lambda *a, **kw: None
    )
    leader = MetalLeader(MetalLeaderTeleopConfig(id="test", calibration_dir=tmp_path, port="can1"))
    yield leader
    leader.arm._connected = False


def test_factory_builds_the_leaders(tmp_path):
    single = MetalLeaderTeleopConfig(port="can1", calibration_dir=tmp_path)
    assert isinstance(make_teleoperator_from_config(single), MetalLeader)
    pair = BiMetalLeaderConfig(
        calibration_dir=tmp_path,
        left_arm_config=MetalLeaderConfig(port="can2"),
        right_arm_config=MetalLeaderConfig(port="can3"),
    )
    assert isinstance(make_teleoperator_from_config(pair), BiMetalLeader)


def test_leader_floats_with_a_free_gripper(leader):
    params = leader.arm.controller._params
    assert params.idle_mode == "float"
    np.testing.assert_allclose(params.float_kd, np.full(6, 0.1))
    assert leader.arm.controller._gripper_params.kp == 0.0
    assert leader.arm.config.gravity_compensation


def test_actions_are_the_follower_action_keys_in_its_units(leader):
    assert tuple(leader.action_features) == METAL_FEATURE_NAMES
    make_ready(leader.arm)
    leader.arm.controller.state = mit_control.JointState(
        position=np.array([0.1, -0.2, 0.3, 0, 0, 0, 0.5]), velocity=np.zeros(7), torque=np.zeros(7)
    )
    action = leader.get_action()
    assert action["shoulder_pan.pos"] == pytest.approx(math.degrees(0.1))
    assert action["gripper.pos"] == pytest.approx(50.0)


def test_float_mode_commands_no_stiffness_and_gravity(leader):
    """In float mode, every arm joint gets kp 0 and the gravity torque; the gripper gets kp 0."""
    controller = leader.arm.controller
    state = mit_control.JointState(position=np.zeros(7), velocity=np.zeros(7), torque=np.zeros(7))
    _, commands = mit_control._float_step(
        controller._params, controller._gripper_params, state, np.zeros(7), np.full(6, 0.4)
    )
    assert all(command.kp == 0.0 for command in commands)
    assert [command.torque for command in commands] == pytest.approx(
        np.full(6, 0.4) * [1.2, 1.15, 1.1, 1.15, 1, 1]
    )
    gripper = mit_control._GripperForceLimiter(controller._gripper_params).step(0.5, 0.0, 0.0, 0.5, 0.0)
    assert (gripper.kp, gripper.kd) == (0.0, 0.1)


def test_connect_loads_gravity_before_torque_and_disconnect_releases(leader, monkeypatch):
    bus = mock_hardware(leader.arm, monkeypatch)
    leader.connect()
    assert leader.is_connected and leader.arm.controller.active
    bus.enable.assert_called_once()
    leader.disconnect()
    assert not leader.is_connected
    bus.disconnect.assert_called_once()


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.setattr("lerobot.robots.metal_follower.metal_follower.hf_hub_download", lambda *a, **kw: "x")
    monkeypatch.setattr(
        "lerobot.robots.metal_follower.metal_follower.GravityCompensation", lambda *a, **kw: None
    )
    pair = BiMetalLeader(
        BiMetalLeaderConfig(
            id="pair",
            calibration_dir=tmp_path,
            left_arm_config=MetalLeaderConfig(port="can2"),
            right_arm_config=MetalLeaderConfig(port="can3"),
        )
    )
    yield pair
    for arm in pair.arms.arms.values():
        arm._connected = False


def test_bimanual_leader_prefixes_actions_and_shares_one_fault(pair):
    left, right = pair.arms.left_arm, pair.arms.right_arm
    assert left.id == "pair_left" and left.controller._params.idle_mode == "float"
    assert right.controller._gripper_params.kp == 0.0
    assert left.controller.fault_event is right.controller.fault_event
    assert len(pair.action_features) == 14 and "right_gripper.pos" in pair.action_features


def test_bimanual_leader_enables_neither_arm_until_both_are_folded(pair, monkeypatch):
    left = mock_hardware(pair.arms.left_arm, monkeypatch)
    right = mock_hardware(pair.arms.right_arm, monkeypatch)
    right.read_states.return_value.position[2] = 1.0
    with pytest.raises(ValueError, match="fold it"):
        pair.connect()
    left.enable.assert_not_called()
    right.enable.assert_not_called()
    left.disconnect.assert_called_once()
    assert not pair.is_connected


def test_bimanual_leader_disconnect_attempts_both_arms(pair, monkeypatch):
    for arm in pair.arms.arms.values():
        arm._connected = True
    pair.arms.left_arm.disconnect = lambda: (_ for _ in ()).throw(RuntimeError("left failed"))
    right_calls = []
    pair.arms.right_arm.disconnect = lambda: (
        right_calls.append(1) or setattr(pair.arms.right_arm, "_connected", False)
    )
    with pytest.raises(RuntimeError, match="left failed"):
        pair.disconnect()
    assert right_calls == [1]


def test_loaded_leader_gripper_moves_freely(leader):
    """Contact torque on a gripper with no position gain never engages force regulation (it divides by kp)."""
    gripper_params = leader.arm.controller._gripper_params
    make_ready(leader.arm)
    leader.arm.controller._set_initial_state(
        mit_control.JointState(position=np.zeros(7), velocity=np.zeros(7), torque=np.zeros(7))
    )
    limiter = leader.arm.controller._gripper
    for step in range(50):
        command = limiter.step(0.4, 0.0 if step % 2 else 0.5, 0.6, 0.4, now=step * 0.01)
        assert (command.kp, command.kd, command.torque) == (0.0, gripper_params.kd, 0.0)
