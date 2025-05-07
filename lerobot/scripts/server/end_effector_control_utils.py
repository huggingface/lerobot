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

import argparse
import logging
import sys
import time

import numpy as np
import torch

from lerobot.common.robot_devices.utils import busy_wait
from lerobot.scripts.server.kinematics import RobotKinematics

logging.basicConfig(level=logging.INFO)


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
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
            return "no-op"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
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

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, deadzone=0.1):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False

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
                    self.episode_end_status = "success"
                # A button (1) for failure
                elif event.button == 1:
                    self.episode_end_status = "failure"
                # X button (0) for rerecord
                elif event.button == 0:
                    self.episode_end_status = "rerecord_episode"

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
            x_input = self.joystick.get_axis(0)  # Left/Right
            y_input = self.joystick.get_axis(1)  # Up/Down (often inverted)

            # Right stick Y (typically axis 3 or 4)
            z_input = self.joystick.get_axis(3)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Calculate deltas (note: may need to invert axes depending on controller)
            delta_x = -y_input * self.y_step_size  # Forward/backward
            delta_y = -x_input * self.x_step_size  # Left/right
            delta_z = -z_input * self.z_step_size  # Up/down

            return delta_x, delta_y, delta_z

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=0.01,
        y_step_size=0.01,
        z_step_size=0.01,
        deadzone=0.1,
        vendor_id=0x046D,
        product_id=0xC219,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
            vendor_id: USB vendor ID of the gamepad (default: Logitech)
            product_id: USB product ID of the gamepad (default: RumblePad 2)
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.vendor_id = vendor_id
        self.product_id = product_id
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

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        import hid

        devices = hid.enumerate()
        for device in devices:
            if device["vendor_id"] == self.vendor_id and device["product_id"] == self.product_id:
                logging.info(f"Found gamepad: {device.get('product_string', 'Unknown')}")
                return device

        logging.error(
            f"No gamepad with vendor ID 0x{self.vendor_id:04X} and product ID 0x{self.product_id:04X} found"
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
                self.left_x = (data[1] - 128) / 128.0
                self.left_y = (data[2] - 128) / 128.0
                self.right_x = (data[3] - 128) / 128.0
                self.right_y = (data[4] - 128) / 128.0

                # Apply deadzone
                self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
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
                    self.episode_end_status = "success"
                elif buttons & 1 << 5:
                    self.episode_end_status = "failure"
                elif buttons & 1 << 4:
                    self.episode_end_status = "rerecord_episode"
                else:
                    self.episode_end_status = None

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_y * self.x_step_size  # Forward/backward
        delta_y = -self.left_x * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z

    def should_quit(self):
        """Return True if quit button was pressed."""
        return self.quit_requested

    def should_save(self):
        """Return True if save button was pressed."""
        return self.save_requested


def test_forward_kinematics(robot, fps=10):
    logging.info("Testing Forward Kinematics")
    timestep = time.perf_counter()
    kinematics = RobotKinematics(robot.robot_type)
    while time.perf_counter() - timestep < 60.0:
        loop_start_time = time.perf_counter()
        robot.teleop_step()
        obs = robot.capture_observation()
        joint_positions = obs["observation.state"].cpu().numpy()
        ee_pos = kinematics.fk_gripper_tip(joint_positions)
        logging.info(f"EE Position: {ee_pos[:3, 3]}")
        busy_wait(1 / fps - (time.perf_counter() - loop_start_time))


def test_inverse_kinematics(robot, fps=10):
    logging.info("Testing Inverse Kinematics")
    timestep = time.perf_counter()
    while time.perf_counter() - timestep < 60.0:
        loop_start_time = time.perf_counter()
        obs = robot.capture_observation()
        joint_positions = obs["observation.state"].cpu().numpy()
        ee_pos = RobotKinematics.fk_gripper_tip(joint_positions)
        desired_ee_pos = ee_pos
        target_joint_state = RobotKinematics.ik(joint_positions, desired_ee_pos, position_only=True)
        robot.send_action(torch.from_numpy(target_joint_state))
        logging.info(f"Target Joint State: {target_joint_state}")
        busy_wait(1 / fps - (time.perf_counter() - loop_start_time))


def teleoperate_inverse_kinematics_with_leader(robot, fps=10):
    logging.info("Testing Inverse Kinematics")
    kinematics = RobotKinematics(robot.robot_type)
    timestep = time.perf_counter()
    while time.perf_counter() - timestep < 60.0:
        loop_start_time = time.perf_counter()
        obs = robot.capture_observation()
        joint_positions = obs["observation.state"].cpu().numpy()
        ee_pos = kinematics.fk_gripper_tip(joint_positions)

        leader_joint_positions = robot.leader_arms["main"].read("Present_Position")
        leader_ee = kinematics.fk_gripper_tip(leader_joint_positions)

        desired_ee_pos = leader_ee
        target_joint_state = kinematics.ik(
            joint_positions, desired_ee_pos, position_only=True, fk_func=kinematics.fk_gripper_tip
        )
        robot.send_action(torch.from_numpy(target_joint_state))
        logging.info(f"Leader EE: {leader_ee[:3, 3]}, Follower EE: {ee_pos[:3, 3]}")
        busy_wait(1 / fps - (time.perf_counter() - loop_start_time))


def teleoperate_delta_inverse_kinematics_with_leader(robot, fps=10):
    logging.info("Testing Delta End-Effector Control")
    timestep = time.perf_counter()

    # Initial position capture
    obs = robot.capture_observation()
    joint_positions = obs["observation.state"].cpu().numpy()

    kinematics = RobotKinematics(robot.robot_type)

    leader_joint_positions = robot.leader_arms["main"].read("Present_Position")
    initial_leader_ee = kinematics.fk_gripper_tip(leader_joint_positions)

    desired_ee_pos = np.diag(np.ones(4))
    joint_positions = robot.follower_arms["main"].read("Present_Position")
    fixed_ee_pos = kinematics.fk_gripper_tip(joint_positions)

    while time.perf_counter() - timestep < 60.0:
        loop_start_time = time.perf_counter()

        # Get leader state for teleoperation
        leader_joint_positions = robot.leader_arms["main"].read("Present_Position")
        leader_ee = kinematics.fk_gripper_tip(leader_joint_positions)

        # Get current state
        # obs = robot.capture_observation()
        # joint_positions = obs["observation.state"].cpu().numpy()
        joint_positions = robot.follower_arms["main"].read("Present_Position")
        current_ee_pos = kinematics.fk_gripper_tip(joint_positions)

        # Calculate delta between leader and follower end-effectors
        # Scaling factor can be adjusted for sensitivity
        scaling_factor = 1.0
        ee_delta = -np.clip((leader_ee - initial_leader_ee) * scaling_factor, -0.05, 0.05)

        # Apply delta to current position
        desired_ee_pos[0, 3] = fixed_ee_pos[0, 3]  # current_ee_pos[0, 3] + ee_delta[0, 3] * 0
        desired_ee_pos[1, 3] = fixed_ee_pos[1, 3]  # current_ee_pos[1, 3] + ee_delta[1, 3] * 0
        desired_ee_pos[2, 3] = current_ee_pos[2, 3] - ee_delta[2, 3]

        # Compute joint targets via inverse kinematics
        target_joint_state = kinematics.ik(
            joint_positions, desired_ee_pos, position_only=True, fk_func=kinematics.fk_gripper_tip
        )

        initial_leader_ee = leader_ee.copy()

        # Send command to robot
        robot.send_action(torch.from_numpy(target_joint_state))

        # Logging
        logging.info(f"Current EE: {current_ee_pos[:3, 3]}, Desired EE: {desired_ee_pos[:3, 3]}")
        logging.info(f"Delta EE: {ee_delta[:3, 3]}")

        busy_wait(1 / fps - (time.perf_counter() - loop_start_time))


def teleoperate_delta_inverse_kinematics(robot, controller, fps=10, bounds=None, fk_func=None):
    """
    Control a robot using delta end-effector movements from any input controller.

    Args:
        robot: Robot instance to control
        controller: InputController instance (keyboard, gamepad, etc.)
        fps: Control frequency in Hz
        bounds: Optional position limits
        fk_func: Forward kinematics function to use
    """
    if fk_func is None:
        fk_func = RobotKinematics.fk_gripper_tip

    logging.info(f"Testing Delta End-Effector Control with {controller.__class__.__name__}")

    # Initial position capture
    obs = robot.capture_observation()
    joint_positions = obs["observation.state"].cpu().numpy()
    kinematics = RobotKinematics(robot.robot_type)
    current_ee_pos = kinematics.fk_gripper_tip(joint_positions)

    # Initialize desired position with current position
    desired_ee_pos = np.eye(4)  # Identity matrix

    timestep = time.perf_counter()
    with controller:
        while not controller.should_quit() and time.perf_counter() - timestep < 60.0:
            loop_start_time = time.perf_counter()

            # Process input events
            controller.update()

            # Get current robot state
            joint_positions = robot.follower_arms["main"].read("Present_Position")
            current_ee_pos = kinematics.fk_gripper_tip(joint_positions)

            # Get movement deltas from the controller
            delta_x, delta_y, delta_z = controller.get_deltas()

            # Update desired position
            desired_ee_pos[0, 3] = current_ee_pos[0, 3] + delta_x
            desired_ee_pos[1, 3] = current_ee_pos[1, 3] + delta_y
            desired_ee_pos[2, 3] = current_ee_pos[2, 3] + delta_z

            # Apply bounds if provided
            if bounds is not None:
                desired_ee_pos[:3, 3] = np.clip(desired_ee_pos[:3, 3], bounds["min"], bounds["max"])

            # Only send commands if there's actual movement
            if any(abs(v) > 0.001 for v in [delta_x, delta_y, delta_z]):
                # Compute joint targets via inverse kinematics
                target_joint_state = kinematics.ik(joint_positions, desired_ee_pos, position_only=True)

                # Send command to robot
                robot.send_action(torch.from_numpy(target_joint_state))

            busy_wait(1 / fps - (time.perf_counter() - loop_start_time))


def teleoperate_gym_env(env, controller, fps: int = 30):
    """
    Control a robot through a gym environment using keyboard inputs.

    Args:
        env: A gym environment created with make_robot_env
        fps: Target control frequency
    """

    logging.info("Testing Keyboard Control of Gym Environment")
    print("Keyboard controls:")
    print("  Arrow keys: Move in X-Y plane")
    print("  Shift and Shift_R: Move in Z axis")
    print("  ESC: Exit")

    # Reset the environment to get initial observation
    obs, info = env.reset()

    try:
        with controller:
            while not controller.should_quit():
                loop_start_time = time.perf_counter()

                # Process input events
                controller.update()

                # Get movement deltas from the controller
                delta_x, delta_y, delta_z = controller.get_deltas()

                # Create the action vector
                action = np.array([delta_x, delta_y, delta_z])

                # Skip if no movement
                if any(abs(v) > 0.001 for v in [delta_x, delta_y, delta_z]):
                    # Step the environment - pass action as a tensor with intervention flag
                    action_tensor = torch.from_numpy(action.astype(np.float32))
                    obs, reward, terminated, truncated, info = env.step((action_tensor, False))

                    # Log information
                    logging.info(f"Action: [{delta_x:.4f}, {delta_y:.4f}, {delta_z:.4f}]")
                    logging.info(f"Reward: {reward}")

                    # Reset if episode ended
                    if terminated or truncated:
                        logging.info("Episode ended, resetting environment")
                        obs, info = env.reset()

                # Maintain target frame rate
                busy_wait(1 / fps - (time.perf_counter() - loop_start_time))

    finally:
        # Close the environment
        env.close()


if __name__ == "__main__":
    from lerobot.common.envs.configs import EEActionSpaceConfig, EnvWrapperConfig, HILSerlRobotEnvConfig
    from lerobot.common.robot_devices.robots.configs import RobotConfig
    from lerobot.common.robot_devices.robots.utils import make_robot_from_config
    from lerobot.scripts.server.gym_manipulator import make_robot_env

    parser = argparse.ArgumentParser(description="Test end-effector control")
    parser.add_argument(
        "--mode",
        type=str,
        default="keyboard",
        choices=[
            "keyboard",
            "gamepad",
            "keyboard_gym",
            "gamepad_gym",
            "leader_delta",
            "leader",
        ],
        help="Control mode to use",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="so100",
        help="Robot type (so100, koch, aloha, etc.)",
    )

    args = parser.parse_args()

    robot_config = RobotConfig.get_choice_class(args.robot_type)(mock=False)
    robot = make_robot_from_config(robot_config)

    if not robot.is_connected:
        robot.connect()

    # Example bounds
    bounds = {
        "max": np.array([0.32170487, 0.201285, 0.10273342]),
        "min": np.array([0.16631757, -0.08237468, 0.03364977]),
    }

    try:
        # Determine controller type based on mode prefix
        controller = None
        if args.mode.startswith("keyboard"):
            controller = KeyboardController(x_step_size=0.01, y_step_size=0.01, z_step_size=0.05)
        elif args.mode.startswith("gamepad"):
            if sys.platform == "darwin":
                controller = GamepadControllerHID(x_step_size=0.01, y_step_size=0.01, z_step_size=0.05)
            else:
                controller = GamepadController(x_step_size=0.01, y_step_size=0.01, z_step_size=0.05)

        # Handle mode categories
        if args.mode in ["keyboard", "gamepad"]:
            # Direct robot control modes
            teleoperate_delta_inverse_kinematics(robot, controller, bounds=bounds, fps=10)

        elif args.mode in ["keyboard_gym", "gamepad_gym"]:
            # Gym environment control modes
            cfg = HILSerlRobotEnvConfig(robot=robot_config, wrapper=EnvWrapperConfig())
            cfg.wrapper.ee_action_space_params = EEActionSpaceConfig(
                x_step_size=0.03, y_step_size=0.03, z_step_size=0.03, bounds=bounds
            )
            cfg.wrapper.ee_action_space_params.use_gamepad = False
            cfg.device = "cpu"
            env = make_robot_env(cfg, robot)
            teleoperate_gym_env(env, controller, fps=cfg.fps)

        elif args.mode == "leader_delta":
            # Leader-follower modes don't use controllers
            teleoperate_delta_inverse_kinematics_with_leader(robot)

        elif args.mode == "leader":
            teleoperate_inverse_kinematics_with_leader(robot)

    finally:
        if robot.is_connected:
            robot.disconnect()
