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

"""Tests for the navigation base controller.

Hardware-free and SDK-free: the frame math is pure, the stub is
kinematic, and the robot-backed controller is exercised through a fake
Robot that records actions and serves canned odometry.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from lerobot.navigation.base_controller import (
    BaseController,
    RobotBaseController,
    RobotBaseControllerConfig,
    SafeBaseController,
    StubBaseController,
    odometry_to_world_pose,
    world_velocity_to_body,
)

# ----- world_velocity_to_body ---------------------------------------------


def test_forward_maps_to_body_x():
    """heading=0, world +z (forward) → (vx>0, 0, 0)."""
    vx_f, vy_l, vyaw = world_velocity_to_body(0.0, 0.3, 0.0, heading_rad=0.0)
    assert vx_f == pytest.approx(0.3)
    assert vy_l == pytest.approx(0.0)
    assert vyaw == pytest.approx(0.0)


def test_world_right_maps_to_negative_left():
    """heading=0, world +x is the robot's RIGHT → negative y.vel."""
    vx_f, vy_l, _ = world_velocity_to_body(0.3, 0.0, 0.0, heading_rad=0.0)
    assert vx_f == pytest.approx(0.0)
    assert vy_l == pytest.approx(-0.3)


def test_world_x_is_forward_after_quarter_turn():
    vx_f, vy_l, _ = world_velocity_to_body(0.3, 0.0, 0.0, heading_rad=math.pi / 2)
    assert vx_f == pytest.approx(0.3)
    assert vy_l == pytest.approx(0.0, abs=1e-9)


def test_yaw_rate_sign_flips():
    _, _, vyaw = world_velocity_to_body(0.0, 0.0, 0.5, heading_rad=0.0)
    assert vyaw == pytest.approx(-0.5)


def test_velocity_magnitude_preserved_under_rotation():
    vx_f, vy_l, _ = world_velocity_to_body(0.3, 0.4, 0.0, heading_rad=1.234)
    assert math.hypot(vx_f, vy_l) == pytest.approx(0.5)


# ----- odometry_to_world_pose ---------------------------------------------


def test_odometry_at_origin_is_identity():
    pose, heading = odometry_to_world_pose(1.0, 2.0, 0.3, origin=(1.0, 2.0, 0.3))
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-12)
    assert heading == pytest.approx(0.0)


def test_odometry_forward_maps_to_world_z():
    pose, heading = odometry_to_world_pose(1.0, 0.0, 0.0, origin=(0.0, 0.0, 0.0))
    assert pose[0, 3] == pytest.approx(0.0)
    assert pose[2, 3] == pytest.approx(1.0)
    assert heading == pytest.approx(0.0)


def test_odometry_left_maps_to_world_negative_x():
    pose, _ = odometry_to_world_pose(0.0, 1.0, 0.0, origin=(0.0, 0.0, 0.0))
    assert pose[0, 3] == pytest.approx(-1.0)
    assert pose[2, 3] == pytest.approx(0.0)


def test_odometry_yaw_sign_flip():
    pose, heading = odometry_to_world_pose(0.0, 0.0, 0.5, origin=(0.0, 0.0, 0.0))
    assert heading == pytest.approx(-0.5)
    fwd = pose[:3, 2]
    np.testing.assert_allclose(fwd, [math.sin(-0.5), 0.0, math.cos(-0.5)], atol=1e-12)


def test_odometry_origin_yaw_is_derotated():
    """Motion along the boot-time heading is always world +z, whatever
    direction the robot faced when odometry started."""
    origin = (0.0, 0.0, math.pi / 2)
    pose, heading = odometry_to_world_pose(0.0, 1.0, math.pi / 2, origin=origin)
    assert pose[0, 3] == pytest.approx(0.0, abs=1e-12)
    assert pose[2, 3] == pytest.approx(1.0)
    assert heading == pytest.approx(0.0)


# ----- StubBaseController --------------------------------------------------


def test_stub_is_basecontroller():
    assert isinstance(StubBaseController(), BaseController)


def test_stub_integrates_forward():
    c = StubBaseController()
    c.move(0.0, 0.2, dt=1.0)
    assert c.position()[2] == pytest.approx(0.2)


def test_stub_clamps_velocity():
    c = StubBaseController(max_lin_speed=0.1)
    c.move(5.0, 0.0, dt=1.0)
    assert c.position()[0] == pytest.approx(0.1)


# ----- RobotBaseController -------------------------------------------------


class FakeRobot:
    """Minimal Robot stand-in: records actions, serves canned odometry."""

    def __init__(self) -> None:
        self.actions: list[dict] = []
        self.obs: dict = {}

    def send_action(self, action: dict) -> dict:
        self.actions.append(action)
        return action

    def get_observation(self) -> dict:
        return self.obs


def _robot_controller(**cfg_kwargs) -> tuple[RobotBaseController, FakeRobot]:
    robot = FakeRobot()
    cfg = RobotBaseControllerConfig(**cfg_kwargs)
    return RobotBaseController(robot, cfg), robot


def _odom(x=0.0, y=0.0, yaw=0.0) -> dict:
    return {"x.pos": x, "y.pos": y, "theta.pos": yaw}


def test_robot_controller_is_basecontroller():
    ctl, _ = _robot_controller()
    assert isinstance(ctl, BaseController)


def test_forward_command_reaches_send_action():
    ctl, robot = _robot_controller()
    ctl.feed_observation(_odom())  # heading 0
    ctl.move(vx=0.0, vz=0.3, dt=0.05)
    assert robot.actions[-1] == {
        "x.vel": pytest.approx(0.3),
        "y.vel": pytest.approx(0.0),
        "theta.vel": pytest.approx(0.0),
    }


def test_command_uses_odometry_heading():
    """After the robot turns to heading +π/2, a world +x command comes out
    as pure body-forward. First sample fixes the origin."""
    ctl, robot = _robot_controller()
    ctl.feed_observation(_odom())  # origin, heading 0
    ctl.feed_observation(_odom(yaw=-math.pi / 2))  # turned; heading +π/2
    ctl.move(vx=0.3, vz=0.0, dt=0.05)
    assert robot.actions[-1]["x.vel"] == pytest.approx(0.3)
    assert robot.actions[-1]["y.vel"] == pytest.approx(0.0, abs=1e-9)


def test_command_is_clamped_before_send():
    ctl, robot = _robot_controller(max_lin_speed=0.1)
    ctl.feed_observation(_odom())
    ctl.move(vx=0.0, vz=9.0, dt=0.05)
    assert robot.actions[-1]["x.vel"] == pytest.approx(0.1)


def test_pose_comes_from_odometry_not_integration():
    ctl, _ = _robot_controller()
    ctl.feed_observation(_odom())
    ctl.move(0.0, 0.3, dt=1.0)  # would integrate 0.3 m open-loop
    ctl.feed_observation(_odom(x=0.05))  # ...but odometry says 5 cm forward
    assert ctl.position()[2] == pytest.approx(0.05)


def test_origin_is_first_odometry_sample():
    ctl, _ = _robot_controller()
    ctl.feed_observation(_odom(x=3.0, y=-1.0, yaw=0.7))
    np.testing.assert_allclose(ctl.pose(), np.eye(4), atol=1e-12)


def test_open_loop_fallback_without_odometry():
    """No odometry fed → integrate open-loop like the stub."""
    ctl, _ = _robot_controller()
    ctl.move(0.0, 0.2, dt=1.0)
    assert ctl.position()[2] == pytest.approx(0.2)


def test_stop_sends_zero_velocity():
    ctl, robot = _robot_controller()
    ctl.stop()
    assert robot.actions[-1] == {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
    assert ctl.is_stopped


def test_robot_controller_matches_stub_open_loop():
    """Open-loop pose integration matches StubBaseController for the same
    command sequence — sim runs must transfer to the real base."""
    ctl, _ = _robot_controller(max_lin_speed=1.0)
    stub = StubBaseController()
    for vx, vz, yaw in [(0.2, 0.0, 0.0), (0.0, 0.3, 0.5), (0.1, 0.1, -0.2)]:
        ctl.move(vx, vz, yaw, dt=0.5)
        stub.move(vx, vz, yaw, dt=0.5)
    np.testing.assert_allclose(ctl.pose(), stub.pose(), atol=1e-9)


# ----- SafeBaseController --------------------------------------------------


class FakeGrid:
    """Occupancy stand-in with a single obstacle cell."""

    def __init__(self, obstacle_cell=(5, 6), cell_size=0.1, origin_x=-0.5, origin_z=-0.5):
        self.obstacle_cell = obstacle_cell
        self.cell_size = cell_size
        self.origin_x = origin_x
        self.origin_z = origin_z

    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        ix = int((x - self.origin_x) / self.cell_size)
        iz = int((z - self.origin_z) / self.cell_size)
        return iz, ix

    def is_obstacle(self, iz: int, ix: int) -> bool:
        return (iz, ix) == self.obstacle_cell


def test_safe_passes_normal_moves():
    inner = StubBaseController()
    safe = SafeBaseController(inner=inner)
    safe.feed_watchdog()
    safe.move(0.0, 0.1, dt=1.0)
    assert inner.position()[2] == pytest.approx(0.1)


def test_safe_clamps_speed():
    inner = StubBaseController(max_lin_speed=100.0)
    safe = SafeBaseController(inner=inner, max_lin_speed=0.5)
    safe.feed_watchdog()
    safe.move(10.0, 0.0, dt=1.0)
    assert inner.position()[0] == pytest.approx(0.5)


def test_safe_watchdog_latches_on_stale_keyframes():
    inner = StubBaseController()
    safe = SafeBaseController(inner=inner, watchdog_timeout_s=0.05)
    safe.feed_watchdog()
    time.sleep(0.1)
    safe.move(0.0, 0.1, dt=1.0)
    assert safe.e_stop_latched
    safe.move(0.0, 10.0, dt=1.0)  # refused
    assert inner.position()[2] == pytest.approx(0.0, abs=1e-6)


def test_safe_reset_watchdog_re_enables_motion():
    inner = StubBaseController()
    safe = SafeBaseController(inner=inner, watchdog_timeout_s=0.05)
    safe.feed_watchdog()
    time.sleep(0.1)
    safe.move(0.0, 0.1)
    assert safe.e_stop_latched
    safe.reset_watchdog()
    safe.move(0.0, 0.1, dt=1.0)
    assert inner.position()[2] == pytest.approx(0.1)


def test_safe_refuses_move_into_obstacle():
    inner = StubBaseController()
    grid = FakeGrid(obstacle_cell=(5, 6))
    safe = SafeBaseController(inner=inner, occupancy_provider=lambda: grid)
    safe.feed_watchdog()
    # +0.15 m in x from origin lands mid-column ix=6 (origin_x=-0.5, cell=0.1).
    safe.move(vx=0.15, vz=0.0, dt=1.0)
    assert safe.e_stop_latched
    assert inner.position()[0] == pytest.approx(0.0, abs=1e-6)


def test_safe_allows_move_into_free_cell():
    inner = StubBaseController()
    grid = FakeGrid(obstacle_cell=(99, 99))
    safe = SafeBaseController(inner=inner, occupancy_provider=lambda: grid)
    safe.feed_watchdog()
    safe.move(vx=0.1, vz=0.0, dt=1.0)
    assert inner.position()[0] == pytest.approx(0.1)


def test_safe_allows_when_no_map_yet():
    inner = StubBaseController()
    safe = SafeBaseController(inner=inner, occupancy_provider=lambda: None)
    safe.feed_watchdog()
    safe.move(vx=0.1, vz=0.0, dt=1.0)
    assert inner.position()[0] == pytest.approx(0.1)
