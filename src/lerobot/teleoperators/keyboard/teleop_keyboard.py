#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from queue import Queue
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardTeleopConfig,
)

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info("Could not import pynput keyboard backend: %s", e)


class KeyboardTeleop(Teleoperator):
    """Teleoperator that reads raw keyboard key states via `pynput` for manual control.

    [`~teleoperators.Teleoperator.get_action`] reports every key currently held down. Requires an
    interactive desktop session capable of capturing global key events — an X11 session (Linux), a
    Windows desktop, or macOS with Accessibility / Input Monitoring permission granted. On Wayland or a
    headless machine, [`~teleoperators.Teleoperator.connect`] logs a warning and the teleoperator produces
    no actions.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        """Instantiate the teleoperator.

        Args:
            config (`KeyboardTeleopConfig`):
                Configuration for this keyboard teleoperator.

        Raises:
            ImportError: If `pynput` is not installed.
        """
        require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        """See [`~teleoperators.Teleoperator.action_features`].

        Returns:
            `dict`: Motor count and names taken from `self.arm`.
        """
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        """See [`~teleoperators.Teleoperator.feedback_features`]. `KeyboardTeleop` accepts no feedback."""
        return {}

    @property
    def is_connected(self) -> bool:
        """See [`~teleoperators.Teleoperator.is_connected`]."""
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        """See [`~teleoperators.Teleoperator.is_calibrated`]. Keyboard input does not require calibration."""
        pass

    @check_if_already_connected
    def connect(self) -> None:
        """See [`~teleoperators.Teleoperator.connect`].

        Starts a `pynput` keyboard listener if the current session can capture key events; otherwise logs
        a warning and leaves the teleoperator producing no actions.
        """
        if PYNPUT_AVAILABLE and pynput_can_capture():
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.warning(
                "Keyboard teleoperation is unavailable in this environment. pynput can only "
                "capture key events on an X11 session (Linux), a Windows desktop, or macOS with "
                "Accessibility / Input Monitoring granted - not on Wayland or headless machines. "
                "This keyboard teleoperator will produce no actions; use an X11 session, a "
                "gamepad, or a leader-arm teleoperator instead."
            )
            self.listener = None

    def calibrate(self) -> None:
        """See [`~teleoperators.Teleoperator.calibrate`]. No-op: keyboard input does not require calibration."""
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        """See [`~teleoperators.Teleoperator.configure`]. No-op: keyboard input needs no configuration."""
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Read the keys currently held down.

        Returns:
            `dict[str, Any]`: One entry per key character currently pressed, each mapped to `None`. An
            empty dict means no key is currently held.

        Raises:
            DeviceNotConnectedError: If [`~teleoperators.Teleoperator.connect`] has not been called.
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """See [`~teleoperators.Teleoperator.send_feedback`]. No-op: `KeyboardTeleop` accepts no feedback."""
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        """See [`~teleoperators.Teleoperator.disconnect`]. Stops the keyboard listener, if one is running."""
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """Keyboard teleoperator for end-effector (Cartesian delta) control.

    Arrow keys and shift map to `delta_x`/`delta_y`/`delta_z`; `ctrl_l`/`ctrl_r` map to the gripper.
    Designed for use with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        """Instantiate the teleoperator.

        Args:
            config (`KeyboardEndEffectorTeleopConfig`):
                Configuration for this keyboard end-effector teleoperator.
        """
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        """See [`~teleoperators.Teleoperator.action_features`].

        Returns:
            `dict`: A 3-element (or 4-element if `config.use_gripper` is `True`) `float32` vector named
            `delta_x`, `delta_y`, `delta_z`, and optionally `gripper`.
        """
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Translate held-down keys into an end-effector Cartesian delta.

        Arrow keys drive `delta_x`/`delta_y`; `shift`/`shift_r` drive `delta_z`. `ctrl_r` opens the
        gripper and `ctrl_l` closes it (only present when `config.use_gripper` is `True`); any other
        pressed key is queued for [`~teleoperators.keyboard.KeyboardEndEffectorTeleop.get_teleop_events`]
        instead of affecting the action.

        Returns:
            `dict[str, Any]`: `delta_x`, `delta_y`, `delta_z`, and, if enabled, `gripper`.

        Raises:
            DeviceNotConnectedError: If [`~teleoperators.Teleoperator.connect`] has not been called.
        """
        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == keyboard.Key.up:
                delta_y = -int(val)
            elif key == keyboard.Key.down:
                delta_y = int(val)
            elif key == keyboard.Key.left:
                delta_x = int(val)
            elif key == keyboard.Key.right:
                delta_x = -int(val)
            elif key == keyboard.Key.shift:
                delta_z = -int(val)
            elif key == keyboard.Key.shift_r:
                delta_z = int(val)
            elif key == keyboard.Key.ctrl_r:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == keyboard.Key.ctrl_l:
                gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """Read auxiliary keyboard events used to drive episode control during recording.

        Any of the movement/gripper keys held down counts as an active intervention. `s`, `r`, and `q`
        are read once as one-shot signals for success, rerecord, and quit respectively; reading this
        method clears the currently tracked key state.

        Returns:
            `dict[TeleopEvents, bool]`: Values for the [`~teleoperators.TeleopEvents`] keys
            `IS_INTERVENTION`, `TERMINATE_EPISODE`, `SUCCESS`, and `RERECORD_EPISODE`.
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right,
            keyboard.Key.shift,
            keyboard.Key.shift_r,
            keyboard.Key.ctrl_r,
            keyboard.Key.ctrl_l,
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        self.current_pressed.clear()

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class KeyboardRoverTeleop(KeyboardTeleop):
    """Keyboard teleoperator for mobile robots such as EarthRover Mini Plus.

    Provides WASD-style driving controls: `w`/`s` drive forward/backward, `a`/`d` turn (with a forward
    motion assist), `q`/`e` rotate in place, `x` is an emergency stop, and `+`/`-` adjust speed. `ESC`
    disconnects the teleoperator.

    **Attributes**:
        - **current_linear_speed** (`float`) -- Current linear velocity magnitude, adjustable at runtime
          with `+`/`-`.
        - **current_angular_speed** (`float`) -- Current angular velocity magnitude, adjustable at
          runtime with `+`/`-`.

    Example:
        ```python
        >>> from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig
        >>> teleop = KeyboardRoverTeleop(KeyboardRoverTeleopConfig(linear_speed=1.0))  # doctest: +SKIP
        >>> teleop.connect()  # doctest: +SKIP
        >>> teleop.get_action()  # doctest: +SKIP
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        """Instantiate the teleoperator.

        Args:
            config (`KeyboardRoverTeleopConfig`):
                Configuration for this keyboard rover teleoperator.
        """
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """See [`~teleoperators.Teleoperator.action_features`].

        Returns:
            `dict`: `linear_velocity` and `angular_velocity`, each mapped to `float`.
        """
        return {
            "linear_velocity": float,
            "angular_velocity": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """See [`~teleoperators.Teleoperator.is_calibrated`]. Rover teleop does not require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue without clearing held keys."""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # Only remove key if it's being released
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Translate held-down WASD-style keys into linear and angular rover velocities.

        `w`/`s` set the linear velocity; `a`/`d` turn while adding a forward-motion assist
        (`config.turn_assist_ratio`) when not already moving; `q`/`e` rotate in place; `x` stops both
        axes. `+`/`-` adjust `current_linear_speed` and `current_angular_speed` in place, clamped to
        `config.min_linear_speed` / `config.min_angular_speed`.

        Returns:
            `dict[str, float]`: `linear_velocity` and `angular_velocity`.

        Raises:
            DeviceNotConnectedError: If [`~teleoperators.Teleoperator.connect`] has not been called.
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        # Check which keys are currently pressed (not released)
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        # Linear movement (W/S) - these take priority
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # Turning (A/D/Q/E)
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # Rotate in place
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # Rotate in place

        # Stop (X) - overrides everything
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # Speed adjustment
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
