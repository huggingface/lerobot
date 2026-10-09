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

"""LeRobot driver for the LimX TRON1 legged robots.

The connection rides on the vendor low-level SDK, which reaches the robot's own control process
over IP.  The SDK is imported lazily inside [`LimxTron1.connect`], so a missing optional
dependency surfaces as a one-line ``pip install 'lerobot[limx_tron1]'`` hint at connect time
rather than an ImportError at import time.

Two properties of that SDK shape this driver.  It picks the joint table from the ``ROBOT_TYPE``
environment variable, which it reads when the vendor package is first imported -- so the driver
exports it before importing when ``robot_type`` is set, and a process cannot switch robot builds
once the package has been loaded.  And it addresses joints positionally: ``RobotState`` carries
``q``/``dq``/``tau`` as bare lists, and the legged builds route commands by index, so no
``motor_names`` field is sent (the vendor's own controller leaves it empty); the driver keeps its
own name table only for LeRobot's feature keys (see
[`limx_tron1.joints`][lerobot.robots.limx_tron1.joints]).
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
from .config_limx_tron1 import LimxTron1Config
from .joints import joint_vector, pos_key

logger = logging.getLogger(__name__)


class LimxTron1(Robot):
    """Drive a LimX TRON1 legged robot through the vendor low-level SDK.

    Joint targets are positions in radians.  While the robot is connected a background thread
    republishes the latest target at ``publish_rate`` so the controller keeps receiving commands
    between [`send_action`][lerobot.robots.limx_tron1.limx_tron1.LimxTron1.send_action] calls;
    nothing is published until either ``start_pose`` or a first action provides a target.

    **Attributes**:
        - **config_class** (`type[LimxTron1Config]`) -- The configuration class this robot reads.
        - **name** (`str`) -- The robot type used on the command line, `"limx_tron1"`.
    """

    config_class = LimxTron1Config
    name = "limx_tron1"

    def __init__(self, config: LimxTron1Config):
        super().__init__(config)
        self.config = config
        self._robot: Any | None = None
        self._datatypes: Any | None = None
        #: Latest joint positions from the SDK, snapshotted in the callback (see `_on_state`).
        self._latest_state: tuple[float, ...] | None = None
        self._state_received_at: float = 0.0
        self._state_lock = threading.Lock()
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
                only calls [`LimxTron1.calibrate`], which does nothing.

        Raises:
            ConnectionError: If the SDK does not report a successful initialisation.
            ImportError: If the optional `limxsdk` dependency is not installed.
        """
        if self.is_connected:
            return

        robot, datatypes = self._make_robot()
        if not robot.init(self.config.robot_ip):
            raise ConnectionError(f"LimX TRON1 SDK failed to initialise against {self.config.robot_ip}")

        self._robot = robot
        self._datatypes = datatypes

        self._stop.clear()
        self._publisher = threading.Thread(target=self._publish_loop, name="limx-tron1-publish", daemon=True)
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

        require_package("limxsdk", extra="limx_tron1", import_name="limxsdk")

        from limxsdk import datatypes
        from limxsdk.robot import Robot as VendorRobot, RobotType

        logger.info("Connecting to LimX TRON1 at %s", self.config.robot_ip)
        robot = VendorRobot(RobotType.PointFoot)
        robot.subscribeRobotState(self._on_state)
        return robot, datatypes

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

    def _publish_loop(self) -> None:
        """Republish the latest joint target at `publish_rate` until asked to stop."""
        period = 1.0 / self.config.publish_rate
        while not self._stop.is_set():
            target = self._target
            if target is not None:
                try:
                    self._publish(target)
                except Exception:  # a transient publish failure must not kill the loop
                    logger.exception("Failed to publish a LimX TRON1 joint command")
            self._stop.wait(period)

    def _publish(self, target: np.ndarray) -> None:
        """Send one joint-position command built from the configured gains."""
        robot, datatypes = self._require_connected()
        dim = len(self.config.joint_names)

        command = datatypes.RobotCmd()
        command.stamp = time.time_ns()
        # mode 0 is the SDK's "torque-position hybrid" mode (see datatypes.h); with zero
        # feedforward torque and zero desired velocity it reduces to pure PD position tracking.
        # This is exactly what the vendor's own TRON1 controller ships on the real robot
        # (tron1-rl-deploy-python: mode 0, tau 0, dq 0), so it is the vendor-validated choice
        # rather than an assumption.
        command.mode = [0] * dim
        command.q = target.tolist()
        command.dq = [0.0] * dim
        command.tau = [0.0] * dim
        command.Kp = list(self.config.kp)
        command.Kd = list(self.config.kd)
        # No `motor_names` and no `parallel_solve_required`: the legged builds route commands by
        # index and have no parallel-linkage conversion, and the vendor's own controller leaves
        # both fields empty.
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
        dim = len(self.config.joint_names)
        if len(positions) != dim:
            raise RuntimeError(f"expected {dim} joint positions from the SDK, got {len(positions)}")
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
            `RobotAction`: The action that was sent.  The controller interpolates towards the
            setpoint rather than echoing it, so the setpoint is the closest honest answer to
            "what was sent".

        Raises:
            KeyError: If a joint key is missing from the action.
            ValueError: If a joint target is not finite.
            RuntimeError: If the robot is not connected.
        """
        # Check the connection first: "not connected" is a more useful failure than
        # "the action is missing a key" when the caller has not opened the robot yet.
        self._require_connected()
        target = joint_vector(action, self.config.joint_names)
        # Hand the target to the publish loop first, so a failure to send it once does not leave
        # the loop republishing the previous pose.
        self._target = target
        self._publish(target)
        return action

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

        The two are set by [`LimxTron1.connect`] and cleared by
        [`LimxTron1.disconnect`] together, so one check covers both.

        Returns:
            `tuple[Any, Any]`: The vendor robot handle and the ``limxsdk.datatypes``
                module it was built with.

        Raises:
            RuntimeError: If the robot is not connected.
        """
        if self._robot is None or self._datatypes is None:
            raise RuntimeError(
                "LimX TRON1 robot is not connected - call connect() first "
                "(LeRobot does this for you via the Robot context manager)"
            )
        return self._robot, self._datatypes


__all__ = ["LimxTron1"]
