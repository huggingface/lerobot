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

"""A robot whose joints a controller drives from its own thread."""

import logging
import threading
import time
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .controller import RobotController

logger = logging.getLogger(__name__)

# How often the controller thread logs the rate it achieves.
_RATE_LOG_PERIOD_S = 5.0


@runtime_checkable
class LowLevelControl(Protocol):
    """What a robot exposes for a controller to drive its joints."""

    config: RobotConfig
    control_lock: Any
    """Reentrant lock deciding who drives the joints over a span of time: one controller tick, or a
    whole reset sweep."""

    @property
    def passthrough_features(self) -> dict[str, type]:
        """The joint targets callers can still send while a controller runs (the G1's arms)."""
        ...

    def low_level_state(self) -> Any:
        """The latest low-level state the controller reads, or None before the first one."""
        ...

    def write_joint_targets(self, targets: dict[str, float], kp: Any = None, kd: Any = None) -> None:
        """Write absolute joint targets, with these PD gains instead of the robot's when given."""
        ...

    def set_gains(self, kp: Any, kd: Any) -> None:
        """Set the PD gains the robot holds its joints with, from connect on."""
        ...

    def reset(self, control_dt: float | None = None, default_positions: Any = None) -> None:
        """Move to `default_positions`, the robot's default pose when None, holding `control_lock`."""
        ...


class ControlledRobot(Robot):
    """A robot whose joints a controller drives from its own thread, at the controller's rate.

    `make_robot_from_config` builds one when the robot's config names a controller
    (`--robot.controller=...`), so teleoperation, recording and rollout see a robot whose actions are the
    controller's commands. Every `control_dt`, the thread hands the controller the robot's latest low-level
    state and the last command, and writes the joint targets it returns.

    `send_action` updates the command; the robot's passthrough joints in the action (the G1's arms under a
    locomotion controller) still go to the robot directly. Name, id, calibration and any attribute this
    class does not define (cameras, embodiment, ...) are the robot's.

    Args:
        robot (`Robot`):
            The robot to drive. It must implement `LowLevelControl`.
        controller (`RobotController`):
            The controller driving it.
    """

    config_class = RobotConfig

    def __init__(self, robot: Robot, controller: RobotController):
        if not isinstance(robot, LowLevelControl):
            raise TypeError(f"{robot.name} does not expose the low-level control a controller needs")
        self.name = robot.name
        super().__init__(robot.config)
        self.robot = robot
        self.controller = controller
        self.config = robot.config
        self.supports_position_hold = getattr(controller, "supports_position_hold", False)
        # The last value of each command key; the thread reads a snapshot every tick.
        self.command: dict[str, Any] = {}
        self._command_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __str__(self) -> str:
        return f"{self.robot} with {type(self.controller).__name__}"

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes this instance lacks. `robot` itself is checked first, since
        # __getattr__ also runs while __init__ has not set it yet.
        robot = self.__dict__.get("robot")
        if robot is None or name.startswith("__"):
            raise AttributeError(name)
        return getattr(robot, name)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        robot_ft = self.robot.observation_features
        controller_ft = getattr(self.controller, "observation_ft", None)
        if controller_ft is None:
            return robot_ft
        # The controller's state replaces the robot's proprioception; camera frames (shape tuples) stay.
        cameras = {key: value for key, value in robot_ft.items() if isinstance(value, tuple)}
        return {**controller_ft, **cameras}

    @cached_property
    def action_features(self) -> dict[str, type]:
        commands = dict(getattr(self.controller, "action_ft", {}))
        if getattr(self.controller, "passthrough", False):
            return {**self.robot.passthrough_features, **commands}
        return commands

    @property
    def is_connected(self) -> bool:
        return self.robot.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.robot.is_calibrated

    def calibrate(self) -> None:
        self.robot.calibrate()

    def configure(self) -> None:
        self.robot.configure()

    def connect(self, calibrate: bool = True) -> None:
        """Connect the robot with the controller's gains, ease into its home pose, then start the thread."""
        kp = getattr(self.controller, "kp", None)
        if kp is not None:
            self.robot.set_gains(kp, getattr(self.controller, "kd", None))
        self.robot.connect(calibrate)
        if getattr(self.controller, "default_angles", None) is not None:
            self.reset()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-controller", daemon=True)
        self._thread.start()
        logger.info(f"Controller thread started ({1.0 / self.controller.control_dt:.0f} Hz)")

    def _loop(self) -> None:
        control_dt = self.controller.control_dt
        kp, kd = getattr(self.controller, "kp", None), getattr(self.controller, "kd", None)
        ticks, since = 0, time.perf_counter()
        while not self._stop.is_set():
            start = time.perf_counter()
            state = self.robot.low_level_state()
            if state is not None:
                with self._command_lock:
                    command = dict(self.command)
                # One tick is one turn of control, so a reset sweep cannot interleave its own targets.
                with self.robot.control_lock:
                    targets = self.controller.run_step(command, state)
                    self.robot.write_joint_targets(targets, kp=kp, kd=kd)
                ticks += 1
                if start - since >= _RATE_LOG_PERIOD_S:
                    logger.info(
                        f"Controller rate: {ticks / (start - since):.1f} Hz (target {1.0 / control_dt:.1f} Hz)"
                    )
                    ticks, since = 0, start
            time.sleep(max(0.0, control_dt - (time.perf_counter() - start)))

    def get_observation(self) -> RobotObservation:
        obs = self.robot.get_observation()
        if obs and hasattr(self.controller, "observation_state"):
            obs.update(self.controller.observation_state())
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        """Update the command, and send the robot's passthrough joints in `action` to it directly.

        Every key is kept in the command; each controller reads the keys it understands. Without passthrough
        joints nothing is written, which would only repeat the thread's last targets.
        """
        with self._command_lock:
            for key, value in action.items():
                if isinstance(key, str) and value is not None:
                    self.command[key] = value
        passthrough = {key: value for key, value in action.items() if key in self.robot.passthrough_features}
        if not passthrough:
            return action
        return {**action, **self.robot.send_action(passthrough)}

    def reset(self, control_dt: float | None = None, default_positions: Any = None) -> None:
        """Move to the controller's home pose (or `default_positions`), then clear the controller's state.

        The controller is reset while the thread is still held off, so it cannot half-refill the buffers
        being cleared.
        """
        if default_positions is None:
            default_positions = getattr(self.controller, "default_angles", None)
        with self.robot.control_lock:
            self.robot.reset(control_dt=control_dt, default_positions=default_positions)
            self.controller.reset()

    def disconnect(self) -> None:
        """Stop the controller thread, then disconnect the robot.

        The thread stops first: a tick still in flight would re-stiffen joints the robot is making passive.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Controller thread did not stop cleanly")
            self._thread = None
        self.robot.disconnect()
