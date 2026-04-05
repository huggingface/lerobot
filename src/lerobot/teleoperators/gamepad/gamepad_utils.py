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
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        pass

    def get_deltas(self):
        return 0.0, 0.0, 0.0

    def update(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_episode_end_status(self):
        status = self.episode_end_status
        self.episode_end_status = None
        return status

    def should_intervene(self):
        return self.intervention_flag

    def gripper_command(self):
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
            "success": False,
            "failure": False,
            "intervention": False,
            "rerecord": False,
        }
        self.listener = None

    def start(self):
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
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = True
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = True
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif key == keyboard.Key.esc:
                    self.key_states["failure"] = True
                    self.episode_end_status = TeleopEvents.FAILURE
                elif key == keyboard.Key.space:
                    self.key_states["intervention"] = not self.key_states["intervention"]
                elif hasattr(key, "char") and key.char == "r":
                    self.key_states["rerecord"] = True
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE
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
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = False
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift / Shift_R: Move in Z axis")
        print("  Ctrl_R / Ctrl_L: Open / Close gripper")
        print("  Space: Toggle intervention")
        print("  Enter: End episode with SUCCESS")
        print("  Esc: End episode with FAILURE")
        print("  R: Rerecord episode")

    def stop(self):
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
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

    def should_intervene(self):
        return self.key_states["intervention"]

    def reset(self):
        for key in self.key_states:
            self.key_states[key] = False


class GamepadController(InputController):
    """Generate motion deltas from gamepad input using pygame.

    Matches gym-hil button/axis conventions for Linux gamepads, including
    Xbox mappings.
    """

    # Face buttons (same across most controllers on Linux)
    BUTTON_A = 0
    BUTTON_B = 1
    BUTTON_X = 2
    BUTTON_Y = 3
    BUTTON_LB = 4
    BUTTON_RB = 5
    # Stick axes
    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 2
    AXIS_RIGHT_Y = 3

    # Default trigger buttons
    BUTTON_LT = 6
    BUTTON_RT = 7

    # Xbox (gym-hil mapping on Linux)
    XBOX_BUTTON_LT = 9
    XBOX_BUTTON_RT = 10

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False
        self.is_xbox = False
        self._xbox360_profile = False
        self._invert_left_x = False
        self._invert_left_y = True
        self._invert_right_y = True

    def _detect_xbox(self, name):
        name_lower = name.lower()
        return any(tag in name_lower for tag in ["xbox", "microsoft", "x-box"])

    def start(self):
        import pygame

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            logging.error("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        joystick_name = self.joystick.get_name()
        self.is_xbox = self._detect_xbox(joystick_name)
        self._xbox360_profile = joystick_name == "Xbox 360 Controller"
        if self._xbox360_profile:
            # gym-hil "Xbox 360 Controller" profile
            self.AXIS_RIGHT_X = 3
            self.AXIS_RIGHT_Y = 4
            self.BUTTON_LT = self.XBOX_BUTTON_LT
            self.BUTTON_RT = self.XBOX_BUTTON_RT
            self._invert_left_x = True
        else:
            # gym-hil default profile
            self.AXIS_RIGHT_X = 2
            self.AXIS_RIGHT_Y = 3
            self.BUTTON_LT = 6
            self.BUTTON_RT = 7
            self._invert_left_x = False
        logging.info(f"Initialized gamepad: {joystick_name} (xbox={self.is_xbox})")

        print("Gamepad controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print("  RB: Intervention toggle")
        print("  LT / RT: Close / Open gripper")
        print("  Y: End episode with SUCCESS")
        print("  A: End episode with FAILURE")
        print("  X: Rerecord episode")

    def stop(self):
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == self.BUTTON_Y:
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif event.button == self.BUTTON_A:
                    self.episode_end_status = TeleopEvents.FAILURE
                elif event.button == self.BUTTON_X:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE
                elif event.button == self.BUTTON_LT:
                    self.close_gripper_command = True
                elif event.button == self.BUTTON_RT:
                    self.open_gripper_command = True

            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [self.BUTTON_Y, self.BUTTON_A, self.BUTTON_X]:
                    self.episode_end_status = None
                elif event.button == self.BUTTON_LT:
                    self.close_gripper_command = False
                elif event.button == self.BUTTON_RT:
                    self.open_gripper_command = False

            if self.joystick.get_button(self.BUTTON_RB):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    def get_deltas(self):
        import pygame

        try:
            x_input = self.joystick.get_axis(self.AXIS_LEFT_X)
            y_input = self.joystick.get_axis(self.AXIS_LEFT_Y)
            z_input = self.joystick.get_axis(self.AXIS_RIGHT_Y)

            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            if self._invert_left_x:
                x_input = -x_input
            if self._invert_left_y:
                y_input = -y_input
            if self._invert_right_y:
                z_input = -z_input

            delta_x = y_input * self.y_step_size
            delta_y = x_input * self.x_step_size
            delta_z = z_input * self.z_step_size

            return delta_x, delta_y, delta_z

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI.

    Supports auto-detection of controller type for correct HID report parsing.
    Currently supported: Logitech RumblePad 2, 8BitDo Ultimate 2C Wireless.
    """

    CONTROLLER_LOGITECH = "logitech"
    CONTROLLER_8BITDO = "8bitdo"
    CONTROLLER_UNKNOWN = "unknown"

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None
        self.controller_type = self.CONTROLLER_UNKNOWN

        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        self.buttons = {}

    def find_device(self):
        import hid

        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5", "8BitDo"]):
                return device

        logging.error(
            "No gamepad found, check the connection and the product string in HID to add your gamepad"
        )
        return None

    def _detect_controller_type(self, product_string):
        product = product_string.lower() if product_string else ""
        if "8bitdo" in product:
            return self.CONTROLLER_8BITDO
        elif "logitech" in product:
            return self.CONTROLLER_LOGITECH
        return self.CONTROLLER_UNKNOWN

    def start(self):
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

            self.controller_type = self._detect_controller_type(product)
            logging.info(f"Detected controller type: {self.controller_type}")

            print("Gamepad controls (HID mode):")
            print("  Left analog stick: Move in X-Y plane")
            print("  Right analog stick: Move in Z axis (vertical)")
            print("  RB: Intervention toggle")
            if self.controller_type == self.CONTROLLER_8BITDO:
                print("  L3 (left stick click): Close gripper")
                print("  R3 (right stick click): Open gripper")
            else:
                print("  LT: Close gripper")
                print("  RT: Open gripper")
            print("  Y: End episode with SUCCESS")
            print("  X: End episode with FAILURE")
            print("  A: Rerecord episode")

        except OSError as e:
            logging.error(f"Error opening gamepad: {e}")
            logging.error("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """Read the device several times to drain the HID buffer and get a stable reading."""
        for _ in range(10):
            self._update()

    def _update(self):
        if not self.device or not self.running:
            return

        try:
            data = self.device.read(64)
            if not data:
                return

            if self.controller_type == self.CONTROLLER_8BITDO:
                self._parse_8bitdo(data)
            else:
                self._parse_logitech(data)

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def _apply_deadzone(self):
        self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
        self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
        self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
        self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

    def _parse_8bitdo(self, data):
        """Parse HID report from 8BitDo Ultimate 2C Wireless (Bluetooth on macOS).

        11-byte report layout:
            byte[0]:  Report ID (0x01)
            byte[1]:  D-pad hat switch (0=N, 2=E, 5=S, 6=W, 15=neutral)
            byte[2]:  Left Stick X  (0=left, 127=center, 255=right)
            byte[3]:  Left Stick Y  (0=up, 127=center, 255=down)
            byte[4]:  Right Stick X (inverted: 255=left, 0=right)
            byte[5]:  Right Stick Y (0=up, 127=center, 255=down)
            byte[6]:  RT analog trigger (0-255)
            byte[7]:  LT analog trigger (0-255)
            byte[8]:  Buttons -- bit0=A, bit1=B, bit3=X, bit4=Y, bit6=LB, bit7=RB
            byte[9]:  System  -- bit0=LT(digital), bit1=RT(digital), bit3=Select,
                                bit4=Start, bit5=L3, bit6=R3
            byte[10]: Unused
        """
        if len(data) < 11:
            return

        self.left_x = (data[2] - 127) / 128.0
        self.left_y = (data[3] - 127) / 128.0
        self.right_x = -(data[4] - 127) / 128.0
        self.right_y = (data[5] - 127) / 128.0

        self._apply_deadzone()

        buttons = data[8]

        # RB (bit 7) = intervention
        self.intervention_flag = bool(buttons & 0x80)

        # Stick clicks for gripper: R3 (byte[9] bit6) = open, L3 (byte[9] bit5) = close
        system = data[9]
        self.open_gripper_command = bool(system & 0x40)  # R3
        self.close_gripper_command = bool(system & 0x20)  # L3

        # Y (bit 4) = success, X (bit 3) = failure, A (bit 0) = rerecord
        if buttons & 0x10:
            self.episode_end_status = TeleopEvents.SUCCESS
        elif buttons & 0x08:
            self.episode_end_status = TeleopEvents.FAILURE
        elif buttons & 0x01:
            self.episode_end_status = TeleopEvents.RERECORD_EPISODE
        else:
            self.episode_end_status = None

    def _parse_logitech(self, data):
        """Parse HID report from Logitech RumblePad 2 (and similar Logitech gamepads).

        Report layout (8+ bytes):
            byte[1]: Left Stick X  (0-255, center=128)
            byte[2]: Left Stick Y  (0-255, center=128)
            byte[3]: Right Stick X (0-255, center=128)
            byte[4]: Right Stick Y (0-255, center=128)
            byte[5]: Face buttons bitmask
            byte[6]: Shoulder/trigger buttons bitmask
        """
        if len(data) < 8:
            return

        self.left_x = (data[1] - 128) / 128.0
        self.left_y = (data[2] - 128) / 128.0
        self.right_x = (data[3] - 128) / 128.0
        self.right_y = (data[4] - 128) / 128.0

        self._apply_deadzone()

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
        delta_x = -self.left_y * self.x_step_size
        delta_y = -self.left_x * self.y_step_size
        delta_z = -self.right_y * self.z_step_size

        return delta_x, delta_y, delta_z
