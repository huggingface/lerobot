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

import logging
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _hidapi_available, _pygame_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..utils import TeleopEvents

if TYPE_CHECKING or _pygame_available:
    import pygame
else:
    pygame = None  # type: ignore[assignment]

if TYPE_CHECKING or _hidapi_available:
    import hid
else:
    hid = None  # type: ignore[assignment]


class InputController:
    """Base class for input controllers that generate motion deltas for gamepad-style teleoperation.

    Subclasses override `start`, `stop`, `update`, and `get_deltas` to read an actual device; this base
    class returns inert defaults.
    """

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """Instantiate the controller's step sizes and reset its state.

        Args:
            x_step_size (`float`, *optional*, defaults to 1.0):
                Movement step size along X, in meters.
            y_step_size (`float`, *optional*, defaults to 1.0):
                Movement step size along Y, in meters.
            z_step_size (`float`, *optional*, defaults to 1.0):
                Movement step size along Z, in meters.
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, or a TeleopEvents member (SUCCESS, FAILURE, RERECORD_EPISODE)
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources. Subclasses open the actual device here."""
        pass

    def stop(self):
        """Stop the controller and release resources. Subclasses close the actual device here."""
        pass

    def get_deltas(self):
        """Get the current movement deltas.

        Returns:
            `tuple[float, float, float]`: `(dx, dy, dz)` in meters. Always `(0.0, 0.0, 0.0)` on the base
            class.
        """
        return 0.0, 0.0, 0.0

    def update(self):
        """Refresh the controller's internal state. Call this once per frame before reading deltas or events."""
        pass

    def __enter__(self):
        """Support for use in `with` statements. Calls `start`."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting a `with` block, even on error."""
        self.stop()

    def get_episode_end_status(self):
        """Read and clear the current episode end status.

        Returns:
            `TeleopEvents | None`: `None` if the episode should continue, otherwise whichever
            [`~teleoperators.TeleopEvents`] member (e.g. `SUCCESS`, `FAILURE`, `RERECORD_EPISODE`) a
            subclass most recently recorded.
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Whether the intervention flag is currently set.

        Returns:
            `bool`: `True` if a human is currently intervening.
        """
        return self.intervention_flag

    def gripper_command(self):
        """Derive a gripper command from the open/close button flags.

        Returns:
            `str`: `"open"` or `"close"` if exactly one of the flags is set, `"stay"` otherwise.
        """
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input via `pynput`, as an alternative to a physical gamepad.

    Arrow keys drive X/Y, shift/shift_r drive Z, `enter`/`backspace` end the episode with success/failure,
    and `esc` stops the listener.
    """

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """See `InputController.__init__`; the step sizes have the same meaning here."""
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "quit": False,
            "success": False,
            "failure": False,
        }
        self.listener = None

    def start(self):
        """Start the `pynput` keyboard listener, if the current session can capture key events."""
        if not pynput_can_capture():
            logging.warning(
                "Keyboard control is unavailable in this environment. pynput cannot capture keys "
                "on Wayland or headless machines, or on macOS without Accessibility / Input "
                "Monitoring permission. Keyboard motion will be inactive."
            )
            self.running = False
            return

        from pynput import keyboard

        def on_press(key):
            """Update key/episode state for a key-down event."""
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.esc:
                    self.key_states["quit"] = True
                    self.running = False
                    return False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = TeleopEvents.FAILURE
            except AttributeError:
                pass

        def on_release(key):
            """Update key state for a key-up event."""
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = False
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  ESC: Exit")

    def stop(self):
        """Stop the `pynput` keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from held-down arrow/shift keys.

        Returns:
            `tuple[float, float, float]`: `(dx, dy, dz)` in meters.
        """
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z


class GamepadController(InputController):
    """Generate motion deltas from gamepad input via `pygame`.

    Left stick drives X/Y, the right stick's vertical axis drives Z. Y/Triangle, A/Cross, and X/Square
    end the episode with success, failure, or rerecord respectively; RB/LT open and close the gripper;
    holding RB also sets the intervention flag.
    """

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1):
        """Instantiate the controller.

        Args:
            x_step_size (`float`, *optional*, defaults to 1.0):
                Movement step size along X, in meters.
            y_step_size (`float`, *optional*, defaults to 1.0):
                Movement step size along Y, in meters.
            z_step_size (`float`, *optional*, defaults to 1.0):
                Movement step size along Z, in meters.
            deadzone (`float`, *optional*, defaults to 0.1):
                Minimum absolute stick reading before it is treated as input, to filter out drift.

        Raises:
            ImportError: If `pygame` is not installed.
        """
        require_package("pygame", extra="gamepad")
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False

    def start(self):
        """Initialize `pygame` and connect to the first detected joystick."""
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            logging.error("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        logging.info(f"Initialized gamepad: {self.joystick.get_name()}")

        print("Gamepad controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print("  B/Circle button: Exit")
        print("  Y/Triangle button: End episode with SUCCESS")
        print("  A/Cross button: End episode with FAILURE")
        print("  X/Square button: Rerecord episode")

    def stop(self):
        """Clean up `pygame` joystick and display resources."""
        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Drain pending `pygame` events to refresh button, episode, and intervention state."""
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 3:
                    self.episode_end_status = TeleopEvents.SUCCESS
                # A button (1) for failure
                elif event.button == 1:
                    self.episode_end_status = TeleopEvents.FAILURE
                # X button (0) for rerecord
                elif event.button == 0:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE

                # RB button (6) for closing gripper
                elif event.button == 6:
                    self.close_gripper_command = True

                # LT button (7) for opening gripper
                elif event.button == 7:
                    self.open_gripper_command = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [0, 2, 3]:
                    self.episode_end_status = None

                elif event.button == 6:
                    self.close_gripper_command = False

                elif event.button == 7:
                    self.open_gripper_command = False

            # Check for RB button (typically button 5) for intervention flag
            if self.joystick.get_button(5):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    def get_deltas(self):
        """Get the current movement deltas from the joystick axes, after applying the deadzone.

        Returns:
            `tuple[float, float, float]`: `(dx, dy, dz)` in meters. `(0.0, 0.0, 0.0)` if reading the
            joystick raises `pygame.error` (e.g. the controller was disconnected).
        """
        try:
            # Read joystick axes
            # Left stick X and Y (typically axes 0 and 1)
            y_input = self.joystick.get_axis(0)  # Up/Down (often inverted)
            x_input = self.joystick.get_axis(1)  # Left/Right

            # Right stick Y (typically axis 3 or 4)
            z_input = self.joystick.get_axis(3)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Calculate deltas (note: may need to invert axes depending on controller)
            delta_x = -x_input * self.x_step_size  # Forward/backward
            delta_y = -y_input * self.y_step_size  # Left/right
            delta_z = -z_input * self.z_step_size  # Up/down

            return delta_x, delta_y, delta_z

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input by reading raw HID reports via `hidapi`.

    An alternative to `GamepadController` for controllers `pygame` does not reliably detect (notably on
    macOS). Byte offsets in `update` are tuned for the Logitech RumblePad 2 and may need adjusting for
    other controllers.
    """

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """Instantiate the controller.

        Args:
            x_step_size (`float`, *optional*, defaults to 1.0):
                Movement step size along X, in meters.
            y_step_size (`float`, *optional*, defaults to 1.0):
                Movement step size along Y, in meters.
            z_step_size (`float`, *optional*, defaults to 1.0):
                Movement step size along Z, in meters.
            deadzone (`float`, *optional*, defaults to 0.1):
                Minimum absolute stick reading before it is treated as input, to filter out drift.

        Raises:
            ImportError: If `hidapi` is not installed.
        """
        require_package("hidapi", extra="gamepad", import_name="hid")
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Button states
        self.buttons = {}

    def find_device(self):
        """Look for a supported gamepad among enumerated HID devices.

        Matches the first device whose product string contains `"Logitech"`, `"Xbox"`, `"PS4"`, or
        `"PS5"`.

        Returns:
            `dict | None`: The `hidapi` device info dict, or `None` if no matching device was found.
        """
        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5"]):
                return device

        logging.error(
            "No gamepad found, check the connection and the product string in HID to add your gamepad"
        )
        return None

    def start(self):
        """Find and open the gamepad's HID device in non-blocking mode."""
        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            logging.info(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            logging.info(f"Connected to {manufacturer} {product}")

            logging.info("Gamepad controls (HID mode):")
            logging.info("  Left analog stick: Move in X-Y plane")
            logging.info("  Right analog stick: Move in Z axis (vertical)")
            logging.info("  Button 1/B/Circle: Exit")
            logging.info("  Button 2/A/Cross: End episode with SUCCESS")
            logging.info("  Button 3/X/Square: End episode with FAILURE")

        except OSError as e:
            logging.error(f"Error opening gamepad: {e}")
            logging.error("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """Read and process the latest gamepad HID report.

        Reads the device 10 times in a row, since a single `hidapi` read can otherwise return stale data.
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            # Read data from the gamepad
            data = self.device.read(64)
            # Interpret gamepad data - this will vary by controller model
            # These offsets are for the Logitech RumblePad 2
            if data and len(data) >= 8:
                # Normalize joystick values from 0-255 to -1.0-1.0
                self.left_y = (data[1] - 128) / 128.0
                self.left_x = (data[2] - 128) / 128.0
                self.right_x = (data[3] - 128) / 128.0
                self.right_y = (data[4] - 128) / 128.0

                # Apply deadzone
                self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                # Parse button states (byte 5 in the Logitech RumblePad 2)
                buttons = data[5]

                # Check if RB is pressed then the intervention flag should be set
                self.intervention_flag = data[6] in [2, 6, 10, 14]

                # Check if RT is pressed
                self.open_gripper_command = data[6] in [8, 10, 12]

                # Check if LT is pressed
                self.close_gripper_command = data[6] in [4, 6, 12]

                # Check if Y/Triangle button (bit 7) is pressed for saving
                # Check if X/Square button (bit 5) is pressed for failure
                # Check if A/Cross button (bit 4) is pressed for rerecording
                if buttons & 1 << 7:
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif buttons & 1 << 5:
                    self.episode_end_status = TeleopEvents.FAILURE
                elif buttons & 1 << 4:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE
                else:
                    self.episode_end_status = None

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from the last-read HID report.

        Returns:
            `tuple[float, float, float]`: `(dx, dy, dz)` in meters.
        """
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_x * self.x_step_size  # Forward/backward
        delta_y = -self.left_y * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z
