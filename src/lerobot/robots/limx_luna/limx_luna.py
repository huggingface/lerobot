#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""LeRobot driver for the LimX Dynamics Luna humanoid robot (``HU_L04``).

The connection rides on the vendor low-level SDK, which reaches the robot's own control process
over IP.  The SDK is imported lazily inside [`LimxLuna.connect`], so a missing optional
dependency surfaces as a one-line ``pip install 'lerobot[limx_luna]'`` hint at connect time
rather than an ImportError at import time.

Two properties of that SDK shape this driver.  It picks the joint table from the ``ROBOT_TYPE``
environment variable, which it reads when the vendor package is first imported -- so the driver
exports it before importing when ``robot_type`` is set, and a process cannot switch robot builds
once the package has been loaded.  And it addresses joints positionally: ``RobotState`` carries
``q``/``dq``/``tau`` as bare lists, so the driver always sends ``motor_names`` alongside a command
and keeps its own name table (see [`limx_luna.joints`][lerobot.robots.limx_luna.joints]).

Luna's ankles and waist are driven through parallel linkages, so the vendor controller has two
joint spaces: a "motor" space, where the ankle and waist slots hold linkage motor positions, and
a "serial" joint space, where they are the ankle and waist axes themselves and the controller
solves the linkages.  This driver works in the serial joint space -- every command sets
``parallel_solve_required`` -- so observations and actions are joint angles, consistent with the
URDF.  Data recorded in motor space (for example by the vendor's closed-loop motion-tracking
policies) is not directly interchangeable with it.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.import_utils import require_package

from ..robot import Robot
from .config_limx_luna import LimxLunaConfig
from .joints import LUNA_DIM, joint_vector, pos_key

logger = logging.getLogger(__name__)


class LimxLuna(Robot):
    """Drive a LimX Luna robot through the vendor low-level SDK.

    Joint targets are positions in radians.  While the robot is connected a background thread
    republishes the latest target at ``publish_rate`` so the controller keeps receiving commands
    between [`send_action`][lerobot.robots.limx_luna.limx_luna.LimxLuna.send_action]
    calls; nothing is published until either ``start_pose`` or a first action provides a target.

    **Attributes**:
        - **config_class** (`type[LimxLunaConfig]`) -- The configuration class this robot reads.
        - **name** (`str`) -- The robot type used on the command line, `"limx_luna"`.
    """

    config_class = LimxLunaConfig
    name = "limx_luna"

    def __init__(self, config: LimxLunaConfig):
        super().__init__(config)
        self.config = config
        # The config already rejects missing gains; re-check so a config mutated after
        # construction cannot reach the robot with no gains.
        if config.kp is None or config.kd is None:
            raise ValueError("limx_luna needs explicit kp and kd gains")
        self._kp: list[float] = list(config.kp)
        self._kd: list[float] = list(config.kd)
        self._robot: Any | None = None
        self._datatypes: Any | None = None
        #: Latest joint positions from the SDK, snapshotted in the callback (see `_on_state`).
        self._latest_state: tuple[float, ...] | None = None
        #: Vendor motor names for the command name field, read best-effort at connect time.
        self._sdk_motor_names: list[str] | None = None
        self._state_received_at: float = 0.0
        self._state_lock = threading.Lock()
        #: Serialises SDK publishes: `send_action` and the publish loop run on different threads.
        self._publish_lock = threading.Lock()
        #: Latest joint target, or `None` while the robot should be left alone.
        self._target: np.ndarray | None = (
            np.asarray(config.start_pose, dtype=np.float64) if config.start_pose is not None else None
        )
        self._publisher: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ setup

    @property
    def is_connected(self) -> bool:
        """Whether the vendor SDK handle is currently held."""
        return self._robot is not None

    @property
    def is_calibrated(self) -> bool:
        """Bring-up and zeroing happen in the robot controller, not here."""
        return True

    def calibrate(self) -> None:
        """No-op: calibration is owned by the robot controller."""

    def configure(self) -> None:
        """No-op: the vendor controller owns its own runtime configuration.

        Kept as an explicit hook so a deployment can add a fault check or a mode switch without
        having to touch the rest of the driver.
        """

    def connect(self, calibrate: bool = True) -> None:
        """Open the connection to the robot-side controller.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Accepted for interface compatibility.  The robot controller zeroes itself, so this
                only calls [`LimxLuna.calibrate`], which does nothing.

        Raises:
            ConnectionError: If the SDK does not report a successful initialisation.
            ImportError: If the optional `limxsdk` dependency is not installed.
        """
        if self.is_connected:
            return

        robot, datatypes = self._make_robot()
        if not robot.init(self.config.robot_ip):
            raise ConnectionError(f"LimX Luna SDK failed to initialise against {self.config.robot_ip}")

        self._robot = robot
        self._datatypes = datatypes
        self._sdk_motor_names = self._query_motor_names(robot)

        # Each connection gets its own stop event and its own handle, and the publish thread
        # closes over both.  A thread left behind by a previous connection (one that was still
        # inside a blocking `publishRobotCmd` when `disconnect` gave up joining it) therefore
        # sees its own event already set and its own handle no longer current, so it can never
        # publish to the new connection.  Clearing a shared event instead -- as this used to do
        # -- let such a thread resume against the freshly opened handle.
        stop = threading.Event()
        self._stop = stop
        self._publisher = threading.Thread(
            target=self._publish_loop,
            args=(stop, robot, datatypes),
            name="limx-luna-publish",
            daemon=True,
        )
        self._publisher.start()

        if calibrate:
            # Nothing to do, but keep the hook so subclasses can override.
            self.calibrate()

    def _make_robot(self) -> tuple[Any, Any]:
        """Import the vendor SDK and build its robot handle for this configuration.

        Kept as a separate method so tests -- and hardware-free dry runs -- can substitute a stub
        instead of importing the vendor package.
        """
        if self.config.robot_type is not None:
            # The vendor package reads this when it is first imported, so it has to be set before
            # the import below rather than at connect time on a loaded module.
            os.environ["ROBOT_TYPE"] = self.config.robot_type

        require_package("limxsdk", extra="limx_luna", import_name="limxsdk")

        from limxsdk import datatypes
        from limxsdk.robot import Robot as VendorRobot, RobotType

        logger.info("Connecting to LimX Luna at %s", self.config.robot_ip)
        robot = VendorRobot(RobotType.Humanoid)
        robot.subscribeRobotState(self._on_state)
        return robot, datatypes

    def _query_motor_names(self, robot: Any) -> list[str] | None:
        """Read the vendor's real motor names for the command name field, best-effort.

        The SDK indexes joints positionally, so the ``q`` vector works without names.  But the
        vendor's own example fills ``RobotCmd.motor_names`` from ``getMotorNames()``, which
        suggests the controller *may* route commands by name rather than by index.  This reads
        those names and returns them only when they look usable (exactly `LUNA_DIM`,
        non-empty, no duplicates); otherwise it returns `None` and the caller falls back to the
        descriptive table.  A failure here must never block connecting.

        Returns:
            `list[str] | None`: The vendor motor names, or `None` if they are unavailable or
                not usable.
        """
        try:
            names = robot.getMotorNames()
        except Exception:
            return None
        if (
            isinstance(names, (list, tuple))
            and len(names) == LUNA_DIM
            and all(isinstance(name, str) and name.strip() for name in names)
            and len(set(names)) == LUNA_DIM
        ):
            return [name.strip() for name in names]
        logger.warning(
            "LimX Luna SDK returned no usable motor names; sending the descriptive "
            "table in the command name field instead"
        )
        return None

    def disconnect(self) -> None:
        """Stop republishing and drop the SDK handle.

        The vendor SDK exposes no teardown call, so releasing the handle is all this can do; the
        controller notices a silent publisher on its own.

        Note:
            The publish thread is joined with a bounded timeout: a wedged publisher must not hang
            process shutdown, and the thread is a daemon so it cannot keep the interpreter alive.
        """
        self._stop.set()
        publisher, self._publisher = self._publisher, None
        if publisher is not None and publisher.is_alive():
            publisher.join(timeout=2.0)

        self._robot = None
        self._datatypes = None
        self._sdk_motor_names = None
        # Drop the last command so a later reconnect does not immediately re-send it.
        self._target = (
            np.asarray(self.config.start_pose, dtype=np.float64)
            if self.config.start_pose is not None
            else None
        )
        with self._state_lock:
            self._latest_state = None

    # ------------------------------------------------------------------- loop

    def _on_state(self, state: Any) -> None:
        """Record the newest state sample pushed by the SDK.

        The SDK only promises a freshly-allocated state object for its newer Tron2 channels,
        so on this channel it may hand out a reused object that it mutates in place.  The
        positions are therefore copied out here -- a tuple of floats has no identity worth
        sharing -- and only that immutable snapshot is kept.
        """
        positions = tuple(state.q)
        with self._state_lock:
            self._latest_state = positions
            self._state_received_at = time.monotonic()

    def _publish_loop(self, stop: threading.Event, robot: Any, datatypes: Any) -> None:
        """Republish the latest joint target at `publish_rate` until this connection ends.

        The target is read inside the publish lock rather than before it.  Reading it outside
        let a stale snapshot win: the loop would take a copy of target A, then `send_action`
        would install and send B and return, and only then would the loop publish A -- so the
        robot received B followed by the older A.  Holding the lock across "read the target and
        publish it" means the two callers are serialised over the whole operation, not just over
        the SDK call.
        """
        period = 1.0 / self.config.publish_rate
        while not stop.is_set():
            with self._publish_lock:
                # Checked under the lock as well: `disconnect` may have set it while this
                # thread was waiting, and this thread must not touch the SDK after that.
                if stop.is_set() or self._robot is not robot:
                    return
                target = self._target
                if target is not None:
                    try:
                        self._send(robot, datatypes, target)
                    except Exception:  # a transient publish failure must not kill the loop
                        logger.exception("Failed to publish a LimX Luna joint command")
            stop.wait(period)

    def _send(self, robot: Any, datatypes: Any, target: np.ndarray) -> None:
        """Send one joint-position command built from the configured gains and names.

        Callers hold `_publish_lock`; this never takes it itself, so `send_action` can install a
        new target and publish it as one atomic step.
        """
        command = datatypes.RobotCmd()
        command.stamp = time.time_ns()
        # mode 0 is the SDK's "torque-position hybrid" mode (see datatypes.h); with zero
        # feedforward torque and zero desired velocity it reduces to pure PD position tracking,
        # the same command shape the vendor's humanoid controllers send (mode 0, tau 0, dq 0).
        command.mode = [0] * LUNA_DIM
        command.q = target.tolist()
        command.dq = [0.0] * LUNA_DIM
        command.tau = [0.0] * LUNA_DIM
        command.Kp = list(self._kp)
        command.Kd = list(self._kd)
        # Prefer the vendor's own motor names when the SDK returned them (the controller may
        # route by name); fall back to the descriptive table otherwise.
        command.motor_names = list(self._sdk_motor_names or self.config.joint_names)
        # Luna drives its ankles and waist through parallel linkages.  Targets here are in joint
        # space, so ask the controller to solve the linkages rather than treat the ankle and waist
        # values as motor positions (the vendor's "serial" joint space; see the module docstring).
        command.parallel_solve_required = [True] * LUNA_DIM
        robot.publishRobotCmd(command)

    # --------------------------------------------------------------- features

    def _scalar_features(self) -> dict[str, type]:
        """Per-joint scalar features, shared by the observation and action space."""
        return {pos_key(name): float for name in self.config.joint_names}

    @property
    def observation_features(self) -> dict[str, type]:
        """Per-joint scalar features the robot reports.

        Narrower than the `dict[str, type | tuple]` the camera-bearing robots return: this
        integration exposes no image features, so every value is a scalar type.
        """
        return self._scalar_features()

    @property
    def action_features(self) -> dict[str, type]:
        """Per-joint scalar features the robot accepts."""
        return self._scalar_features()

    # ------------------------------------------------------------------- i/o

    def get_observation(self) -> RobotObservation:
        """Read one state sample and map it onto LeRobot's feature keys.

        Returns:
            `RobotObservation`: Joint positions in radians, keyed by ``"<joint>.pos"``.

        Raises:
            RuntimeError: If the robot is not connected, no fresh state arrives within
                `observation_timeout`, or the sample carries the wrong number of joints.
        """
        # Fail fast on a missing connection.  Waiting out the timeout would turn a clear
        # programming error into a misleading "the robot is unreachable" message.
        self._require_connected()
        positions = self._wait_for_state()
        if len(positions) != LUNA_DIM:
            raise RuntimeError(f"expected {LUNA_DIM} joint positions from the SDK, got {len(positions)}")
        return {
            pos_key(name): float(value)
            for name, value in zip(self.config.joint_names, positions, strict=True)
        }

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send a joint command to the robot.

        Args:
            action (`RobotAction`):
                Target joint positions in radians, keyed by ``"<joint>.pos"``.

        Returns:
            `RobotAction`: The setpoint actually commanded, keyed by `action_features`.  As in
            the other LeRobot robots (see [`Robot.send_action`]), this is the command, not a
            measured echo; measured positions come from [`LimxLuna.get_observation`].  Any
            keys the robot does not command are dropped, so recorded datasets match
            `action_features`.

        Raises:
            KeyError: If a joint key is missing from the action.
            ValueError: If a joint target is not finite.
            RuntimeError: If the robot is not connected.
        """
        # Check the connection first: "not connected" is a more useful failure than
        # "the action is missing a key" when the caller has not opened the robot yet.
        robot, datatypes = self._require_connected()
        target = joint_vector(action, self.config.joint_names)
        # Install the target and publish it as one step under the publish lock.  Splitting these
        # let the publish loop send the previous target after this call had already returned,
        # which is how an older setpoint could land on the robot after a newer one.
        with self._publish_lock:
            self._target = target
            self._send(robot, datatypes, target)
        return {
            pos_key(name): float(value) for name, value in zip(self.config.joint_names, target, strict=True)
        }

    # ---------------------------------------------------------------- helpers

    def _wait_for_state(self) -> tuple[float, ...]:
        """Block until a state sample arrives that is fresh enough to use.

        Returns:
            `tuple[float, ...]`: The joint positions of the newest sample, in SDK vector order.

        Raises:
            RuntimeError: If no acceptable sample arrives within `observation_timeout`.
        """
        deadline = time.monotonic() + self.config.observation_timeout
        while True:
            with self._state_lock:
                state = self._latest_state
                received_at = self._state_received_at
            if state is not None:
                age = time.monotonic() - received_at
                if self.config.state_max_age is None or age <= self.config.state_max_age:
                    return state
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"no fresh state sample within {self.config.observation_timeout}s - "
                    "is the robot reachable at the configured address?"
                )
            time.sleep(0.005)

    def _require_connected(self) -> tuple[Any, Any]:
        """Return the live SDK handle and its ``datatypes`` module.

        The two are set by [`LimxLuna.connect`] and cleared by
        [`LimxLuna.disconnect`] together, so one check covers both.

        Returns:
            `tuple[Any, Any]`: The vendor robot handle and the ``limxsdk.datatypes``
                module it was built with.

        Raises:
            RuntimeError: If the robot is not connected.
        """
        if self._robot is None or self._datatypes is None:
            raise RuntimeError(
                "LimX Luna robot is not connected - call connect() first "
                "(LeRobot does this for you via the Robot context manager)"
            )
        return self._robot, self._datatypes


__all__ = ["LimxLuna"]
