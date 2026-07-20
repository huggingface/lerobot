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

"""Base controller for spatial-memory navigation.

The navigation/skills layer commands motion in a single **world frame**
(OpenCV convention: x right, y down, z forward — the base lives in the XZ
plane, y is gravity) and reads back an SE(3) pose. :class:`BaseController`
is that seam. Three implementations:

  - :class:`StubBaseController` — kinematic integrator, no hardware; sim +
    unit tests.
  - :class:`RobotBaseController` — drives any LeRobot :class:`Robot` whose
    action space is body-frame velocities ``x.vel`` (forward, m/s),
    ``y.vel`` (left, m/s), ``theta.vel`` (CCW yaw, rad/s) and whose
    observation carries planar odometry ``x.pos``/``y.pos``/``theta.pos``
    (REP-103: x forward, y left, yaw CCW). The Unitree Go2 satisfies this
    out of the box; so would a LeKiwi base.
  - :class:`SafeBaseController` — wraps any of the above with velocity
    clamping, an optional occupancy gate, a keyframe watchdog and an
    e-stop latch.

All frame conversions between the world frame and a robot's body/odometry
frame live in :func:`world_velocity_to_body` and
:func:`odometry_to_world_pose`; nothing else needs to know the mapping.
"""

from __future__ import annotations

import logging
import math
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from lerobot.robots import Robot

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# BaseController protocol
# --------------------------------------------------------------------- #


@runtime_checkable
class BaseController(Protocol):
    """Mobile-base interface used by the navigation/skills layer.

    Velocities are in **world** frame XZ (m/s); ``yaw_rate`` is rad/s
    about the world's −Y axis (turning around the up vector). ``pose``
    is 4×4 SE(3) camera-to-world (OpenCV).
    """

    @abstractmethod
    def move(self, vx: float, vz: float, yaw_rate: float = 0.0, dt: float = 0.05) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def pose(self) -> np.ndarray: ...

    @abstractmethod
    def position(self) -> tuple[float, float, float]: ...


# --------------------------------------------------------------------- #
# Frame math (pure functions)
# --------------------------------------------------------------------- #


def world_velocity_to_body(
    vx_world: float,
    vz_world: float,
    yaw_rate_rad_s: float,
    heading_rad: float,
) -> tuple[float, float, float]:
    """World-frame velocity → body-frame ``(x.vel, y.vel, theta.vel)``.

    Returns ``(vx_forward, vy_left, vyaw)`` in m/s, m/s, rad/s — the
    action a REP-103 base expects. At heading ``h`` the body axes in the
    world XZ plane are forward = (sin h, cos h), left = (−cos h, sin h)
    (left = up × forward, up = −y). The navigation world's positive yaw
    is clockwise about the up vector; a REP-103 base's ``theta.vel`` is
    counter-clockwise, hence the sign flip.
    """
    s, c = math.sin(heading_rad), math.cos(heading_rad)
    vx_fwd = vx_world * s + vz_world * c
    vy_left = -vx_world * c + vz_world * s
    return vx_fwd, vy_left, -yaw_rate_rad_s


def odometry_to_world_pose(
    x_fwd: float,
    y_left: float,
    yaw: float,
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    """Planar odometry ``(x_fwd, y_left, yaw)`` → world pose + heading.

    ``origin`` is the ``(x_fwd, y_left, yaw)`` sample captured when the
    controller first saw odometry, so the run starts at identity
    regardless of where the robot's odometry origin sits. The result is
    the OpenCV world convention, planarized: height Y is 0 and only yaw
    survives of the orientation — pitch/roll gait wobble is the camera's
    concern, not the base's.

    Odometry frame is REP-103 (x forward, y left, yaw CCW about z-up).
    Mapping to OpenCV world: ``x_world = −y_odom``, ``z_world = x_odom``,
    ``heading = −yaw``.
    """
    ox, oy, oyaw = origin
    dx, dy = x_fwd - ox, y_left - oy
    c0, s0 = math.cos(-oyaw), math.sin(-oyaw)
    x_rel = c0 * dx - s0 * dy
    y_rel = s0 * dx + c0 * dy
    yaw_rel = yaw - oyaw

    x_world, z_world = -y_rel, x_rel
    heading = -yaw_rel

    ch, sh = math.cos(heading), math.sin(heading)
    pose = np.eye(4, dtype=np.float64)
    pose[0, 0], pose[0, 2] = ch, sh
    pose[2, 0], pose[2, 2] = -sh, ch
    pose[0, 3], pose[2, 3] = x_world, z_world
    return pose, heading


def _heading_pose(x: float, z: float, heading: float) -> np.ndarray:
    """Build a planar world pose from position + heading."""
    c, s = math.cos(heading), math.sin(heading)
    pose = np.eye(4, dtype=np.float64)
    pose[0, 0], pose[0, 2] = c, s
    pose[2, 0], pose[2, 2] = -s, c
    pose[0, 3], pose[2, 3] = x, z
    return pose


# --------------------------------------------------------------------- #
# Stub controller (kinematic, no hardware)
# --------------------------------------------------------------------- #


@dataclass
class StubBaseController:
    """Kinematic stub: integrates each ``move()`` into pose exactly.

    No latency, slip or dynamics — for sim and skill-layer unit tests.
    """

    initial_pose: np.ndarray | None = None
    max_lin_speed: float = 1.0
    max_yaw_rate: float = 1.0

    def __post_init__(self) -> None:
        self._pose = (
            np.asarray(self.initial_pose, dtype=np.float64).copy()
            if self.initial_pose is not None
            else np.eye(4, dtype=np.float64)
        )
        if self._pose.shape != (4, 4):
            raise ValueError(f"initial_pose must be (4, 4); got {self._pose.shape}")
        self._heading = 0.0
        self._stopped = False

    def move(self, vx: float, vz: float, yaw_rate: float = 0.0, dt: float = 0.05) -> None:
        vx = float(np.clip(vx, -self.max_lin_speed, self.max_lin_speed))
        vz = float(np.clip(vz, -self.max_lin_speed, self.max_lin_speed))
        yaw_rate = float(np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))
        if dt <= 0:
            return
        self._pose[0, 3] += vx * dt
        self._pose[2, 3] += vz * dt
        if yaw_rate != 0.0:
            self._heading += yaw_rate * dt
            self._pose = _heading_pose(self._pose[0, 3], self._pose[2, 3], self._heading)
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def pose(self) -> np.ndarray:
        return self._pose.copy()

    def position(self) -> tuple[float, float, float]:
        p = self._pose[:3, 3]
        return float(p[0]), float(p[1]), float(p[2])

    @property
    def is_stopped(self) -> bool:
        return self._stopped


# --------------------------------------------------------------------- #
# Robot-backed controller
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class RobotBaseControllerConfig:
    """Behaviour knobs for :class:`RobotBaseController`."""

    max_lin_speed: float = 0.6
    """Hard cap on per-axis world linear velocity (m/s)."""

    max_yaw_rate: float = 1.2
    """Hard cap on yaw rate (rad/s)."""

    pose_from_odometry: bool = True
    """Report pose from the robot's odometry (closed-loop). When False,
    integrate pose open-loop from commanded velocities."""


class RobotBaseController(BaseController):
    """:class:`BaseController` over any LeRobot :class:`Robot`.

    The robot must accept body-velocity actions ``x.vel`` (forward),
    ``y.vel`` (left), ``theta.vel`` (CCW yaw) and — for closed-loop pose
    — report odometry ``x.pos``/``y.pos``/``theta.pos`` in its
    observation. This is the standard REP-103 mobile-base contract, which
    ``UnitreeGo2`` implements.

    Pose is refreshed from observations the navigation loop already
    fetches: call :meth:`feed_observation` each keyframe rather than
    having the controller poll the robot (which would trigger an extra
    camera read). Absent any fed observation, pose falls back to
    open-loop integration so sim/dry-run behaves like the stub.
    """

    def __init__(self, robot: Robot, cfg: RobotBaseControllerConfig | None = None) -> None:
        self.robot = robot
        self.cfg = cfg or RobotBaseControllerConfig()
        self._pose = np.eye(4, dtype=np.float64)
        self._heading = 0.0
        self._stopped = False
        self._odom_origin: tuple[float, float, float] | None = None
        self._have_odom = False

    # ----- odometry feed --------------------------------------------------

    def feed_observation(self, obs: dict) -> None:
        """Update pose from an observation the nav loop already fetched."""
        if not self.cfg.pose_from_odometry:
            return
        if not {"x.pos", "y.pos", "theta.pos"} <= obs.keys():
            return
        sample = (float(obs["x.pos"]), float(obs["y.pos"]), float(obs["theta.pos"]))
        if self._odom_origin is None:
            self._odom_origin = sample
        self._pose, self._heading = odometry_to_world_pose(*sample, self._odom_origin)
        self._have_odom = True

    # ----- BaseController API --------------------------------------------

    def move(self, vx: float, vz: float, yaw_rate: float = 0.0, dt: float = 0.05) -> None:
        vx = float(np.clip(vx, -self.cfg.max_lin_speed, self.cfg.max_lin_speed))
        vz = float(np.clip(vz, -self.cfg.max_lin_speed, self.cfg.max_lin_speed))
        yaw_rate = float(np.clip(yaw_rate, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate))
        if dt <= 0:
            return

        vx_fwd, vy_left, vyaw = world_velocity_to_body(vx, vz, yaw_rate, self._heading)
        self.robot.send_action({"x.vel": vx_fwd, "y.vel": vy_left, "theta.vel": vyaw})

        # Open-loop pose only when we have no odometry to trust.
        if not (self.cfg.pose_from_odometry and self._have_odom):
            self._pose[0, 3] += vx * dt
            self._pose[2, 3] += vz * dt
            if yaw_rate != 0.0:
                self._heading += yaw_rate * dt
            self._pose = _heading_pose(self._pose[0, 3], self._pose[2, 3], self._heading)
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True
        try:
            self.robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        except Exception:
            logger.exception("stop(): failed to send zero-velocity action")

    def pose(self) -> np.ndarray:
        return self._pose.copy()

    def position(self) -> tuple[float, float, float]:
        p = self._pose[:3, 3]
        return float(p[0]), float(p[1]), float(p[2])

    @property
    def is_stopped(self) -> bool:
        return self._stopped


# --------------------------------------------------------------------- #
# Safety wrapper
# --------------------------------------------------------------------- #


@dataclass
class SafeBaseController(BaseController):
    """Wrap any :class:`BaseController` with safety layers:

    - **velocity clamp** on every ``move()``;
    - **occupancy gate**: when ``occupancy_provider`` is set, predict
      the next position and refuse (latch e-stop) if it lands in an
      obstacle cell. The provider returns an object exposing
      ``world_to_cell(x, z) -> (iz, ix)`` and an ``is_obstacle(iz, ix)
      -> bool`` predicate; ``None`` means "no map yet, allow";
    - **watchdog**: if no keyframe has been fed in
      ``watchdog_timeout_s`` (caller ticks :meth:`feed_watchdog` per
      map update), ``move()`` latches stop until :meth:`reset_watchdog`.
    """

    inner: BaseController
    max_lin_speed: float = 0.6
    max_yaw_rate: float = 1.2
    occupancy_provider: object = None  # callable[[], grid | None] when set
    watchdog_timeout_s: float = 2.0
    e_stop_latched: bool = False
    _last_keyframe_walltime: float = field(default_factory=time.monotonic, init=False)

    def feed_watchdog(self) -> None:
        self._last_keyframe_walltime = time.monotonic()

    def reset_watchdog(self) -> None:
        self.e_stop_latched = False
        self._last_keyframe_walltime = time.monotonic()

    def latch_estop(self, reason: str = "external") -> None:
        logger.warning("SafeBaseController e-stop latched: %s", reason)
        self.e_stop_latched = True
        self.inner.stop()

    def move(self, vx: float, vz: float, yaw_rate: float = 0.0, dt: float = 0.05) -> None:
        if self.e_stop_latched:
            return
        if (time.monotonic() - self._last_keyframe_walltime) > self.watchdog_timeout_s:
            self.latch_estop(f"watchdog: no keyframe in last {self.watchdog_timeout_s:.2f}s")
            return

        vx = float(np.clip(vx, -self.max_lin_speed, self.max_lin_speed))
        vz = float(np.clip(vz, -self.max_lin_speed, self.max_lin_speed))
        yaw_rate = float(np.clip(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate))

        if self.occupancy_provider is not None:
            try:
                grid = self.occupancy_provider()
            except Exception:
                logger.exception("occupancy_provider raised; refusing move")
                return
            if grid is not None and self._would_enter_obstacle(grid, vx, vz, dt):
                self.latch_estop("about to enter obstacle cell")
                return

        self.inner.move(vx, vz, yaw_rate, dt)

    def stop(self) -> None:
        self.inner.stop()

    def pose(self) -> np.ndarray:
        return self.inner.pose()

    def position(self) -> tuple[float, float, float]:
        return self.inner.position()

    def _would_enter_obstacle(self, grid, vx: float, vz: float, dt: float) -> bool:
        pos = self.inner.position()
        next_x = pos[0] + vx * dt
        next_z = pos[2] + vz * dt
        iz, ix = grid.world_to_cell(next_x, next_z)
        return bool(grid.is_obstacle(iz, ix))
