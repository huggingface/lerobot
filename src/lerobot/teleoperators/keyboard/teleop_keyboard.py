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
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardJointTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardTeleopConfig,
)

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
            logging.info("No DISPLAY set. Skipping pynput import.")
            PYNPUT_AVAILABLE = False
        else:
            from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
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
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        key_value = self._normalize_key(key)
        if key_value is not None:
            self.event_queue.put((key_value, True))

    def _on_release(self, key):
        key_value = self._normalize_key(key)
        if key_value is not None:
            self.event_queue.put((key_value, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _normalize_key(self, key):
        if key is None:
            return None
        if hasattr(key, "char") and key.char is not None:
            return key.char.lower()
        return key

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        names = {
            "enabled": 0,
            "target_x": 1,
            "target_y": 2,
            "target_z": 3,
            "target_wx": 4,
            "target_wy": 5,
            "target_wz": 6,
            "gripper_vel": 7,
        }
        return {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()

        enabled = True
        if self.config.require_deadman:
            enabled = bool(self.current_pressed.get(keyboard.Key.space, False))

        # Base-frame translation controls.
        x_plus = int(self.current_pressed.get(keyboard.Key.left, False))
        x_minus = int(self.current_pressed.get(keyboard.Key.right, False))
        y_plus = int(self.current_pressed.get(keyboard.Key.down, False))
        y_minus = int(self.current_pressed.get(keyboard.Key.up, False))
        z_plus = int(self.current_pressed.get(keyboard.Key.shift_r, False))
        z_minus = int(self.current_pressed.get(keyboard.Key.shift, False))

        target_x = (x_plus - x_minus) * self.config.linear_step
        target_y = (y_plus - y_minus) * self.config.linear_step
        target_z = (z_plus - z_minus) * self.config.linear_step

        # Orientation controls are character-based for portability.
        target_wx = 0.0
        target_wy = 0.0
        target_wz = 0.0
        if self.config.use_orientation:
            wx_plus = int(self.current_pressed.get("i", False))
            wx_minus = int(self.current_pressed.get("k", False))
            wy_plus = int(self.current_pressed.get("j", False))
            wy_minus = int(self.current_pressed.get("l", False))
            wz_plus = int(self.current_pressed.get("u", False))
            wz_minus = int(self.current_pressed.get("o", False))
            target_wx = (wx_plus - wx_minus) * self.config.angular_step
            target_wy = (wy_plus - wy_minus) * self.config.angular_step
            target_wz = (wz_plus - wz_minus) * self.config.angular_step

        gripper_vel = 0.0
        if self.config.use_gripper:
            grip_open = int(self.current_pressed.get("x", False) or self.current_pressed.get(keyboard.Key.ctrl_r, False))
            grip_close = int(self.current_pressed.get("z", False) or self.current_pressed.get(keyboard.Key.ctrl_l, False))
            gripper_vel = (grip_open - grip_close) * self.config.gripper_step

        # While disabled, publish explicit no-motion command to keep behavior deterministic.
        if not enabled:
            target_x = 0.0
            target_y = 0.0
            target_z = 0.0
            target_wx = 0.0
            target_wy = 0.0
            target_wz = 0.0
            gripper_vel = 0.0

        for key, val in self.current_pressed.items():
            if val and isinstance(key, str) and key in {"s", "r", "q"}:
                self.misc_keys_queue.put(key)

        return {
            "enabled": float(enabled),
            "target_x": float(target_x),
            "target_y": float(target_y),
            "target_z": float(target_z),
            "target_wx": float(target_wx),
            "target_wy": float(target_wy),
            "target_wz": float(target_wz),
            "gripper_vel": float(gripper_vel),
        }

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
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
            "i",
            "k",
            "j",
            "l",
            "u",
            "o",
            "x",
            "z",
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

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


class KeyboardJointTeleop(Teleoperator):
    """
    键盘关节级遥操作：按键直接输出各关节增量，无需运动学求解。
    使用 termios 从 stdin 读取按键，不依赖 pynput/X11/curses，终端内直接可用。

    按键映射:
        Q/A: joint1 +/-
        W/S: joint2 +/-
        E/D: joint3 +/-
        R/F: joint4 +/-
        T/G: joint5 +/-
        Y/H: joint6 +/-
        U/J: joint7 +/-
        1/2: 夹爪 开/合
        ESC: 断开连接
    """

    config_class = KeyboardJointTeleopConfig
    name = "keyboard_joint"

    JOINT_PLUS = [ord("q"), ord("w"), ord("e"), ord("r"), ord("t"), ord("y"), ord("u")]
    JOINT_MINUS = [ord("a"), ord("s"), ord("d"), ord("f"), ord("g"), ord("h"), ord("j")]
    GRIPPER_OPEN = ord("1")
    GRIPPER_CLOSE = ord("2")

    def __init__(self, config: KeyboardJointTeleopConfig):
        super().__init__(config)
        self.config = config
        self.current_pressed: set[int] = set()
        self._old_term = None
        self._connected = False

    @property
    def action_features(self) -> dict:
        names = {f"joint{i+1}.delta": i for i in range(self.config.num_joints)}
        names["gripper.delta"] = self.config.num_joints
        names["enabled"] = self.config.num_joints + 1
        return {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self) -> None:
        import select
        import sys
        import termios
        import tty

        self._old_term = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        self._connected = True
        logging.info("KeyboardJointTeleop connected (termios stdin mode).")

    def calibrate(self) -> None:
        pass

    def configure(self):
        pass

    def _read_keys(self):
        import select
        import sys

        newly_pressed: set[int] = set()
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                break
            b = ord(ch)
            if b == 27:
                logging.info("ESC pressed, disconnecting.")
                self.disconnect()
                return
            newly_pressed.add(b)

        if newly_pressed:
            self.current_pressed = newly_pressed
        else:
            self.current_pressed = set()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._read_keys()

        action = {"enabled": 1.0}

        for i in range(self.config.num_joints):
            delta = 0.0
            if self.JOINT_PLUS[i] in self.current_pressed:
                delta += self.config.joint_step
            if self.JOINT_MINUS[i] in self.current_pressed:
                delta -= self.config.joint_step
            action[f"joint{i+1}.delta"] = float(delta)

        gripper_delta = 0.0
        if self.GRIPPER_OPEN in self.current_pressed:
            gripper_delta += self.config.gripper_step
        if self.GRIPPER_CLOSE in self.current_pressed:
            gripper_delta -= self.config.gripper_step
        action["gripper.delta"] = float(gripper_delta)

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_already_connected
    def disconnect(self) -> None:
        import sys
        import termios

        if self._old_term is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term)
            except Exception:
                pass
            self._old_term = None
        self._connected = False
        self.current_pressed = set()
        logging.info("KeyboardJointTeleop disconnected.")


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    Keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Provides intuitive WASD-style controls for driving a mobile robot:
    - Linear movement (forward/backward)
    - Angular movement (turning/rotation)
    - Speed adjustment
    - Emergency stop

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Turn left (with forward motion)
            - D: Turn right (with forward motion)
            - Q: Rotate left in place
            - E: Rotate right in place
            - X: Emergency stop

        Speed Control:
            - +/=: Increase speed
            - -: Decrease speed

        System:
            - ESC: Disconnect teleoperator

    Attributes:
        config: Teleoperator configuration
        current_linear_speed: Current linear velocity magnitude
        current_angular_speed: Current angular velocity magnitude

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """Return action format for rover (linear and angular velocities)."""
        return {
            "linear_velocity": float,
            "angular_velocity": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Rover teleop doesn't require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue without clearing held keys"""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # Only remove key if it's being released
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get the current action based on pressed keys.

        Returns:
            RobotAction with 'linear_velocity' and 'angular_velocity' keys.
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
