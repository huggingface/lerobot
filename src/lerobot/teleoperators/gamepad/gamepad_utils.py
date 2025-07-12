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

    def should_quit(self):
        """Return True if the user has requested to quit."""
        return not self.running

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
                    self.episode_end_status = "success"
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = "failure"
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

    def should_quit(self):
        """Return True if ESC was pressed."""
        return self.key_states["quit"]

    def should_save(self):
        """Return True if Enter was pressed (save episode)."""
        return self.key_states["success"] or self.key_states["failure"]


class GamepadController(InputController):
    """Generate motion deltas from gamepad input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False
        
        # Add D-pad state tracking
        self.dpad_up = False
        self.dpad_down = False
        self.dpad_left = False
        self.dpad_right = False
        
        # Add wrist flex state tracking
        self.wrist_flex_up = False
        self.wrist_flex_down = False
        
        # Add gripper toggle state
        self.gripper_toggle_state = False
        
        # Add pick and place mode state
        self.pick_and_place_mode = False
        
        # Pick and place mode configuration
        self.pick_place_z_scale = 0.3  # Reduced Z sensitivity
        self.pick_place_gripper_scale = 0.5  # Scale for trigger-based gripper control

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
        print("  D-pad up (button 11): Wrist flex up")
        print("  D-pad down (button 12): Wrist flex down")
        print("  Right Bumper (RB, button 10): Rotate wrist clockwise")
        print("  Left Bumper (LB, button 9): Rotate wrist counter-clockwise")
        print("  X button (button 0): Toggle gripper open/close")
        print("  Triangle button (button 3): Toggle pick and place mode")
        print("  B/Circle button: Exit")
        print("  Y/Triangle button: End episode with SUCCESS")
        print("  A/Cross button: End episode with FAILURE")
        print("  X/Square button: Rerecord episode")
        print("")
        print("Pick and Place Mode:")
        print("  - X and Y movement locked")
        print("  - Reduced Z sensitivity")
        print("  - Wrist roll control enabled")
        print("  - Wrist flex control enabled")
        print("  - X button toggles gripper")

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
                    # Toggle pick and place mode instead of success
                    self.pick_and_place_mode = not self.pick_and_place_mode
                    print(f"Pick and place mode: {'ON' if self.pick_and_place_mode else 'OFF'}")
                # A button (1) for failure
                elif event.button == 1:
                    self.episode_end_status = "failure"
                # X button (0) for gripper toggle
                elif event.button == 0:
                    self.gripper_toggle_state = not self.gripper_toggle_state
                    if self.gripper_toggle_state:
                        self.open_gripper_command = True
                        self.close_gripper_command = False
                    else:
                        self.open_gripper_command = False
                        self.close_gripper_command = True

                # D-pad up (button 11) for wrist flex up
                elif event.button == 12:
                    self.wrist_flex_up = True

                # D-pad down (button 12) for wrist flex down
                elif event.button == 11:
                    self.wrist_flex_down = True

                # RB button (10) for wrist roll clockwise
                elif event.button == 10:
                    self.dpad_right = True

                # LB button (9) for wrist roll counter-clockwise
                elif event.button == 9:
                    self.dpad_left = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [2, 3]:
                    self.episode_end_status = None

                # D-pad up (11) release
                elif event.button == 12:
                    self.wrist_flex_up = False

                # D-pad down (12) release
                elif event.button == 11:
                    self.wrist_flex_down = False

                # RB button (10) release
                elif event.button == 10:
                    self.dpad_right = False

                # LB button (9) release
                elif event.button == 9:
                    self.dpad_left = False

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
            y_input = self.joystick.get_axis(0)  # Left/Right
            x_input = self.joystick.get_axis(1)  # Up/Down (often inverted)

            # Right stick Y (typically axis 3 or 4) for Z-axis control
            z_input = self.joystick.get_axis(3)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Calculate X and Y deltas from analog stick
            if self.pick_and_place_mode:
                # Lock X and Y movement in pick and place mode
                delta_x = 0.0
                delta_y = 0.0
            else:
                delta_x = -x_input * self.x_step_size  # Forward/backward
                delta_y = -y_input * self.y_step_size  # Left/right
            
            # Calculate Z delta from right joystick
            z_scale = self.pick_place_z_scale if self.pick_and_place_mode else 1.0
            delta_z = -z_input * self.z_step_size * z_scale  # Up/down

            return delta_x, delta_y, delta_z

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0
    
    def get_wrist_roll_delta(self):
        """Get the current wrist roll delta from D-pad."""
        wrist_roll_delta = 0.0
        if self.dpad_right:
            wrist_roll_delta += self.z_step_size  # Use z_step_size for consistency
        if self.dpad_left:
            wrist_roll_delta -= self.z_step_size
        return wrist_roll_delta

    def get_wrist_flex_delta(self):
        """Get the current wrist flex delta from D-pad up/down."""
        wrist_flex_delta = 0.0
        if self.wrist_flex_up:
            wrist_flex_delta += self.z_step_size  # Use z_step_size for consistency
        if self.wrist_flex_down:
            wrist_flex_delta -= self.z_step_size
        return wrist_flex_delta


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
        self.quit_requested = False
        self.save_requested = False
        
        # Add D-pad state tracking
        self.dpad_up = False
        self.dpad_down = False
        self.dpad_left = False
        self.dpad_right = False
        
        # Add wrist flex state tracking
        self.wrist_flex_up = False
        self.wrist_flex_down = False
        
        # Add gripper toggle state
        self.gripper_toggle_state = False
        
        # Add pick and place mode state
        self.pick_and_place_mode = False
        
        # Pick and place mode configuration
        self.pick_place_z_scale = 0.3  # Reduced Z sensitivity
        self.pick_place_gripper_scale = 0.5  # Scale for trigger-based gripper control

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        import hid

        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5", "Wireless Controller"]):
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
            print(self.device_info)
            logging.info(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            logging.info(f"Connected to {manufacturer} {product}")

            logging.info("Gamepad controls (HID mode):")
            logging.info("  Left analog stick: Move in X-Y plane")
            logging.info("  Right analog stick (vertical): Move in Z axis")
            logging.info("  D-pad up: Wrist flex up")
            logging.info("  D-pad down: Wrist flex down")
            logging.info("  Right Bumper (RB): Rotate wrist clockwise")
            logging.info("  Left Bumper (LB): Rotate wrist counter-clockwise")
            logging.info("  X button: Toggle gripper open/close")
            logging.info("  Triangle button: Toggle pick and place mode")
            logging.info("  Button 1/B/Circle: Exit")
            logging.info("  Button 2/A/Cross: End episode with SUCCESS")
            logging.info("  Button 3/X/Square: End episode with FAILURE")
            logging.info("")
            logging.info("Pick and Place Mode:")
            logging.info("  - X and Y movement locked")
            logging.info("  - Reduced Z sensitivity")
            logging.info("  - Wrist roll control enabled")
            logging.info("  - Wrist flex control enabled")
            logging.info("  - X button toggles gripper")

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

                # Check if RB is pressed for wrist roll clockwise
                self.dpad_right = data[6] in [2, 6, 10, 14]

                # Check if LB is pressed for wrist roll counter-clockwise
                self.dpad_left = data[6] in [1, 5, 9, 13]

                # Parse D-pad from byte 7 (this may need adjustment for different controllers)
                # Note: For pygame controllers, D-pad events are JOYBUTTONDOWN events with buttons [11, 12, 13, 14]
                # For HID controllers, we parse D-pad from byte data
                # Note: Z-axis control is now handled by right joystick, not D-pad
                dpad_value = data[7] if len(data) > 7 else 0
                
                # D-pad mapping (this is controller-specific and may need adjustment)
                # For Logitech RumblePad 2, D-pad is typically in the lower 4 bits
                dpad_direction = dpad_value & 0x0F
                
                # Reset D-pad states (Z-axis control moved to right joystick)
                self.dpad_up = False
                self.dpad_down = False
                
                # Map D-pad up/down to wrist flex control
                # D-pad up (typically bit 0 or 1) for wrist flex up
                if dpad_direction in [1, 2, 3]:  # Adjust these values based on your controller
                    self.wrist_flex_up = True
                else:
                    self.wrist_flex_up = False
                
                # D-pad down (typically bit 2 or 3) for wrist flex down  
                if dpad_direction in [5, 6, 7]:  # Adjust these values based on your controller
                    self.wrist_flex_down = True
                else:
                    self.wrist_flex_down = False

                # Check if Y/Triangle button (bit 7) is pressed for pick and place mode toggle
                # Check if A/Cross button (bit 4) is pressed for failure
                # Check if B/Circle button (bit 5) is pressed for rerecording
                if buttons & 1 << 7:
                    # Toggle pick and place mode instead of success
                    self.pick_and_place_mode = not self.pick_and_place_mode
                    logging.info(f"Pick and place mode: {'ON' if self.pick_and_place_mode else 'OFF'}")
                elif buttons & 1 << 4:
                    self.episode_end_status = "failure"
                elif buttons & 1 << 5:
                    self.episode_end_status = "rerecord_episode"
                else:
                    self.episode_end_status = None

                # Check if X button is pressed for gripper toggle
                if buttons & 1 << 0:  # X button (bit 0)
                    self.gripper_toggle_state = not self.gripper_toggle_state
                    if self.gripper_toggle_state:
                        self.open_gripper_command = True
                        self.close_gripper_command = False
                    else:
                        self.open_gripper_command = False
                        self.close_gripper_command = True

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        if self.pick_and_place_mode:
            # Lock X and Y movement in pick and place mode
            delta_x = 0.0
            delta_y = 0.0
        else:
            delta_x = -self.left_x * self.x_step_size  # Forward/backward
            delta_y = -self.left_y * self.y_step_size  # Left/right
        
        # Calculate Z delta from right joystick
        z_scale = self.pick_place_z_scale if self.pick_and_place_mode else 1.0
        delta_z = -self.right_y * self.z_step_size * z_scale  # Up/down

        return delta_x, delta_y, delta_z
    
    def get_wrist_roll_delta(self):
        """Get the current wrist roll delta from D-pad."""
        wrist_roll_delta = 0.0
        if self.dpad_right:
            wrist_roll_delta += self.z_step_size  # Use z_step_size for consistency
        if self.dpad_left:
            wrist_roll_delta -= self.z_step_size
        return wrist_roll_delta

    def get_wrist_flex_delta(self):
        """Get the current wrist flex delta from D-pad up/down."""
        wrist_flex_delta = 0.0
        if self.wrist_flex_up:
            wrist_flex_delta += self.z_step_size  # Use z_step_size for consistency
        if self.wrist_flex_down:
            wrist_flex_delta -= self.z_step_size
        return wrist_flex_delta

    def should_quit(self):
        """Return True if quit button was pressed."""
        return self.quit_requested

    def should_save(self):
        """Return True if save button was pressed."""
        return self.save_requested
