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

from ..utils import TeleopEvents


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.wrist_roll_command = 0.0  # -1.0 = roll left, 0 = stop, +1.0 = roll right
        self.right_x = 0.0  # right stick horizontal, normalized -1..1

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
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
        """Start the keyboard listener."""
        from pynput import keyboard

        def on_press(key):
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
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
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
    """Generate motion deltas from gamepad input."""

    def __init__(
        self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1, device_name: str | None = None
    ):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False
        self.device_name = device_name

    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame

        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            logging.error("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        if self.device_name is not None:
            # Find joystick matching device_name
            for i in range(count):
                js = pygame.joystick.Joystick(i)
                if self.device_name.lower() in js.get_name().lower():
                    self.joystick = js
                    break
            if self.joystick is None:
                available = [pygame.joystick.Joystick(i).get_name() for i in range(count)]
                logging.error(
                    f"No gamepad matching '{self.device_name}' found. "
                    f"Available: {available}. "
                    f"Use --teleop.device_name=<name> to select one."
                )
                self.running = False
                return
        else:
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
        """Clean up pygame resources."""
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        import pygame

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
        """Get the current movement deltas from gamepad state."""
        import pygame

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
    """Generate motion deltas from gamepad input using HIDAPI."""

    # HID usage_page=0x01 (Generic Desktop), usage=0x05 (Game Pad) per USB HID spec
    GAMEPAD_USAGE_PAGE = 0x01
    GAMEPAD_USAGE = 0x05

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
        device_name: str | None = None,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
            deadzone: Joystick deadzone to prevent drift
            device_name: Substring to match against HID product name. If None,
                uses DEFAULT_GAMEPAD_NAMES to find the first recognized gamepad.
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None
        self.device_name = device_name

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Button states
        self.buttons = {}

    def find_device(self):
        """Find a gamepad HID device.

        Uses the USB HID usage page (Generic Desktop / Game Pad) to reliably
        identify gamepads regardless of brand. When ``device_name`` is set,
        further filters by substring match against the product or manufacturer string.
        """
        import hid

        devices = hid.enumerate()

        # First, collect all actual gamepads by HID usage
        gamepads = [
            d
            for d in devices
            if d.get("usage_page") == self.GAMEPAD_USAGE_PAGE and d.get("usage") == self.GAMEPAD_USAGE
        ]

        if self.device_name is not None:
            # Filter gamepads by user-specified name (matches product or manufacturer)
            needle = self.device_name.lower()
            for gp in gamepads:
                product = gp.get("product_string", "")
                manufacturer = gp.get("manufacturer_string", "")
                if needle in product.lower() or needle in manufacturer.lower():
                    return gp

            available = [
                f"{d.get('manufacturer_string', '')} {d.get('product_string', '')}".strip() for d in gamepads
            ]
            logging.error(
                f"No gamepad matching '{self.device_name}' found. "
                f"Detected gamepads: {available if available else '(none)'}"
            )
            return None

        # No device_name — return the first gamepad found
        if gamepads:
            return gamepads[0]

        available = sorted(
            {
                f"{d.get('manufacturer_string', '')} {d.get('product_string', '')}".strip()
                for d in devices
                if d.get("product_string")
            }
        )
        logging.error(
            f"No gamepad found (looked for HID usage_page={self.GAMEPAD_USAGE_PAGE:#x}, "
            f"usage={self.GAMEPAD_USAGE:#x}). "
            f"All HID devices: {available}. "
            f"Use --teleop.device_name=<name> to match by product/manufacturer name."
        )
        return None

    def start(self):
        """Connect to the gamepad using HIDAPI."""
        import hid

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
        """
        Read and process the latest gamepad data.
        Due to an issue with the HIDAPI, we need to read the read the device several times in order to get a stable reading
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            data = self.device.read(64)
            if not data:
                return

            # Xbox One / Xbox Series controller: 18-byte GIP report (packet type 0x20)
            # Layout:
            #   byte[0]    = 0x20 (packet type)
            #   byte[1]    = 0x00
            #   byte[2:4]  = u16 LE packet counter
            #   byte[4]    = buttons: sync(0), _(1), start(2), back(3), A(4), B(5), X(6), Y(7)
            #   byte[5]    = buttons: dpad_up(0), dpad_down(1), dpad_left(2), dpad_right(3), LB(4), RB(5), L3(6), R3(7)
            #   byte[6:8]  = u16 LE left trigger  (0-1023)
            #   byte[8:10] = u16 LE right trigger  (0-1023)
            #   byte[10:12]= s16 LE left stick X  (-32768 to 32767)
            #   byte[12:14]= s16 LE left stick Y  (-32768 to 32767)
            #   byte[14:16]= s16 LE right stick X (-32768 to 32767)
            #   byte[16:18]= s16 LE right stick Y (-32768 to 32767)
            if len(data) == 18 and data[0] == 0x20:
                self._update_xbox(data)
            elif len(data) >= 8:
                self._update_logitech(data)

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def _update_xbox(self, data: list[int]) -> None:
        """Parse an Xbox One / Xbox Series controller GIP report."""
        import struct

        # Sticks: signed 16-bit LE, normalize to -1.0..1.0
        lx = struct.unpack_from("<h", bytes(data), 10)[0] / 32768.0
        ly = struct.unpack_from("<h", bytes(data), 12)[0] / 32768.0
        rx = struct.unpack_from("<h", bytes(data), 14)[0] / 32768.0
        ry = struct.unpack_from("<h", bytes(data), 16)[0] / 32768.0

        # Negate Y axes to match SDL/Logitech convention (stick up = negative)
        self.left_x = 0.0 if abs(lx) < self.deadzone else lx
        self.left_y = 0.0 if abs(ly) < self.deadzone else -ly
        self.right_x = 0.0 if abs(rx) < self.deadzone else rx
        self.right_y = 0.0 if abs(ry) < self.deadzone else -ry

        # Triggers: unsigned 16-bit LE (0-1023)
        lt = struct.unpack_from("<H", bytes(data), 6)[0]
        rt = struct.unpack_from("<H", bytes(data), 8)[0]

        # Use triggers for gripper: RT = open, LT = close (threshold ~10% of 1023)
        self.open_gripper_command = rt > 100
        self.close_gripper_command = lt > 100

        # Buttons byte 4: A(bit4), B(bit5), X(bit6), Y(bit7)
        btn0 = data[4]
        # Buttons byte 5: LB(bit4), RB(bit5)
        btn1 = data[5]

        # LB/RB for wrist roll: LB = roll left (-1), RB = roll right (+1)
        lb = bool(btn1 & (1 << 4))
        rb = bool(btn1 & (1 << 5))
        if lb and not rb:
            self.wrist_roll_command = -1.0
        elif rb and not lb:
            self.wrist_roll_command = 1.0
        else:
            self.wrist_roll_command = 0.0

        # Y = success, X = failure, A = rerecord
        if btn0 & (1 << 7):  # Y
            self.episode_end_status = TeleopEvents.SUCCESS
        elif btn0 & (1 << 6):  # X
            self.episode_end_status = TeleopEvents.FAILURE
        elif btn0 & (1 << 4):  # A
            self.episode_end_status = TeleopEvents.RERECORD_EPISODE
        else:
            self.episode_end_status = None

    def _update_logitech(self, data: list[int]) -> None:
        """Parse a Logitech RumblePad 2 HID report."""
        # Normalize joystick values from 0-255 to -1.0..1.0
        self.left_y = (data[1] - 128) / 128.0
        self.left_x = (data[2] - 128) / 128.0
        self.right_x = (data[3] - 128) / 128.0
        self.right_y = (data[4] - 128) / 128.0

        self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
        self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
        self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
        self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

        buttons = data[5]

        self.intervention_flag = data[6] in [2, 6, 10, 14]
        self.open_gripper_command = data[6] in [8, 10, 12]
        self.close_gripper_command = data[6] in [4, 6, 12]

        if buttons & 1 << 7:
            self.episode_end_status = TeleopEvents.SUCCESS
        elif buttons & 1 << 5:
            self.episode_end_status = TeleopEvents.FAILURE
        elif buttons & 1 << 4:
            self.episode_end_status = TeleopEvents.RERECORD_EPISODE
        else:
            self.episode_end_status = None

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Left stick vertical (left_y) → forward/backward (delta_x)
        # Left stick horizontal (left_x) → left/right (delta_y)
        # Right stick vertical (right_y) → up/down (delta_z)
        # Convention: stick up/left = negative value (SDL/Logitech), negate to get positive forward/left/up
        delta_x = -self.left_y * self.x_step_size
        delta_y = -self.left_x * self.y_step_size
        delta_z = -self.right_y * self.z_step_size

        return delta_x, delta_y, delta_z
