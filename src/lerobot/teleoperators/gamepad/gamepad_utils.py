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
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
        yaw_step_size=1.0,
        stage_advance_button=0,
    ):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.yaw_step_size = yaw_step_size
        self.stage_advance_button = stage_advance_button
        self.joystick = None
        self.intervention_flag = False
        self._stage_advance_pending = False
        # Latched gripper toggle state (False=open, True=close). Flipped on
        # each R2 press. Starts False; first R2 press commands "close".
        self._gripper_closed_desired = False
        # R2 edge-detect state. DualSense R2 is axis 5 (-1 idle, +1 pressed).
        # Hysteresis: crossing >0.7 arms a press (toggle once); must drop
        # <0.3 before next press can fire.
        self._r2_pressed = False
        self._r2_axis = 5
        # Optional callable returning current gripper state in [0,1] (0=open,
        # 1=closed). When set, R2 press emits the OPPOSITE of current state so
        # the toggle never desyncs from reality. When None, falls back to
        # flipping the latched _gripper_closed_desired flag.
        self.gripper_state_fn = None

    def get_yaw_delta(self):
        """Return yaw stick delta (right-stick horizontal, axis 3 on PS4/DualSense), deadzoned + scaled."""
        import pygame

        try:
            yaw_input = self.joystick.get_axis(3)
            if abs(yaw_input) < self.deadzone:
                return 0.0
            # Invert: stick left → positive yaw (CCW around world Z) feels natural.
            return -yaw_input * self.yaw_step_size
        except pygame.error:
            return 0.0

    def consume_stage_advance(self):
        """Read-and-clear the one-shot stage-advance flag. Returns True iff
        the configured button was pressed since the last call."""
        pending = self._stage_advance_pending
        self._stage_advance_pending = False
        return pending

    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame

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
        print("  Right analog stick (horizontal): Yaw rotation (if use_yaw=true)")
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

        # Poll intervention (RB / button 5) FIRST so event handlers below see
        # the current intervention state. Detect the False->True edge to reset
        # the R2 gripper toggle to neutral, so the policy's last gripper
        # state stays in effect until the user actually presses R2 during
        # intervention.
        # Pump events first so joystick state (buttons + axes) is fresh.
        pygame.event.pump()

        new_intervention = bool(self.joystick.get_button(5))
        if new_intervention and not self.intervention_flag:
            self._gripper_closed_desired = False
            self.close_gripper_command = False
            self.open_gripper_command = False
            # Also re-arm R2 edge so a trigger held through intervention start
            # doesn't immediately fire a toggle.
            self._r2_pressed = True
        self.intervention_flag = new_intervention

        # R2 trigger on DualSense is an analog AXIS (idx 5, idle=-1, pressed=+1),
        # not a button -- pygame never emits JOYBUTTONDOWN for it. Poll the axis
        # each tick and edge-detect a press with hysteresis. Only fires while
        # intervening so pre-intervention presses don't accumulate.
        try:
            r2_val = self.joystick.get_axis(self._r2_axis)
        except Exception:
            r2_val = 0.0
        if not self._r2_pressed and r2_val > 0.7:
            self._r2_pressed = True
            if self.intervention_flag:
                # Prefer live env state if available, so toggle never desyncs
                # from reality (policy or prior intervention may have moved
                # gripper without the gamepad knowing).
                if self.gripper_state_fn is not None:
                    try:
                        current = float(self.gripper_state_fn())
                    except Exception:
                        current = 1.0 if self._gripper_closed_desired else 0.0
                    desired_closed = current < 0.5
                else:
                    desired_closed = not self._gripper_closed_desired
                self._gripper_closed_desired = desired_closed
                self.close_gripper_command = desired_closed
                self.open_gripper_command = not desired_closed
        elif self._r2_pressed and r2_val < 0.3:
            self._r2_pressed = False

        # DBG: dump all gamepad events + intervention state. Remove once mapping
        # is confirmed.
        import os
        _dbg = os.environ.get("LEROBOT_GAMEPAD_DEBUG") == "1"

        for event in pygame.event.get():
            if _dbg:
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"[GP] BTN_DOWN idx={event.button} interv={self.intervention_flag}", flush=True)
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"[GP] BTN_UP   idx={event.button} interv={self.intervention_flag}", flush=True)
                elif event.type == pygame.JOYAXISMOTION:
                    if abs(event.value) > 0.5:
                        print(f"[GP] AXIS     idx={event.axis} val={event.value:.2f} interv={self.intervention_flag}", flush=True)
                elif event.type == pygame.JOYHATMOTION:
                    print(f"[GP] HAT      idx={event.hat} val={event.value}", flush=True)

            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 2:   # For ps4
                    self.episode_end_status = TeleopEvents.SUCCESS
                # A button (1) for failure
                elif event.button == 1:
                    self.episode_end_status = TeleopEvents.FAILURE
                # X button (0) for rerecord
                elif event.button == 3: # For ps4
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE

                # R2 trigger (button 6): toggle gripper close/open on press
                # (not hold). Only respond while intervening so pre-intervention
                # presses don't accumulate and fire at intervention start. Flip
                # the latched desired state and drive close_/open_gripper_command
                # as a mutually-exclusive pair so gripper_command() keeps
                # returning "close" or "open" each tick.
                elif event.button == 6:
                    if self.intervention_flag:
                        self._gripper_closed_desired = not self._gripper_closed_desired
                        self.close_gripper_command = self._gripper_closed_desired
                        self.open_gripper_command = not self._gripper_closed_desired

                # Stage-advance button (configurable; default 0 = Cross on
                # DualSense). JOYBUTTONDOWN fires once per press, so a single
                # tap latches one advance into _stage_advance_pending; held
                # press does NOT repeat.
                if event.button == self.stage_advance_button:
                    self._stage_advance_pending = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [2, 1, 3]:
                    self.episode_end_status = None

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        import pygame

        try:
            # Read joystick axes
            # Left stick X and Y (typically axes 0 and 1)
            y_input = self.joystick.get_axis(1)  # Up/Down (often inverted) # For ps4
            x_input = self.joystick.get_axis(0)  # Left/Right   # For ps4

            # Right stick Y (typically axis 3 or 4)
            z_input = self.joystick.get_axis(4)  # Up/Down for Z    # For ps4

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Calculate deltas (note: may need to invert axes depending on controller)
            delta_x = x_input * self.x_step_size  # Forward/backward    # For ps4
            delta_y = -y_input * self.y_step_size  # Left/right
            delta_z = -z_input * self.z_step_size  # Up/down
            # print(delta_x, delta_y, delta_z)
            return delta_x, delta_y, delta_z

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
        """
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
        """Look for the gamepad device by vendor and product ID."""
        import hid

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
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_x * self.x_step_size  # Forward/backward
        delta_y = -self.left_y * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z
