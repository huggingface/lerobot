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

"""
RealMan Robot Follower for LeRobot.

This module provides integration with RealMan robotic arms (R1D2, RM65, RM75, etc.)
for use as follower robots in teleoperation scenarios.

Supports teleoperation from SO101 leader arm with automatic joint mapping:
- SO101 has 5 arm joints + gripper (6 total)
- RealMan R1D2 has 6 arm joints + gripper (7 total)
- Joint 4 on R1D2 is held at a fixed position during SO101 teleoperation

Joint Mapping (SO101 → RealMan R1D2):
=====================================
SO101 uses normalized values from calibration:
- Arm joints: -100 to 100 (RANGE_M100_100 mode)
- Gripper: 0 to 100 (RANGE_0_100 mode)

RealMan R1D2 uses degrees with these limits (from realman_r1d2_joint_limits.yaml):
- joint_1: [-178°, 178°]  ← shoulder_pan
- joint_2: [-130°, 130°]  ← shoulder_lift  
- joint_3: [-135°, 135°]  ← elbow_flex
- joint_4: [-178°, 178°]  ← FIXED at 0° (not mapped from SO101)
- joint_5: [-128°, 128°]  ← wrist_flex
- joint_6: [-360°, 360°]  ← wrist_roll
- gripper: [1, 1000]      ← gripper (scaled from 0-100)
"""

import json
import logging
import time
from functools import cached_property
from pathlib import Path

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_realman_follower import RealManFollowerConfig

logger = logging.getLogger(__name__)


# Joint name mapping from SO101 (leader) to RealMan R1D2 (follower)
# SO101 has 5 arm joints + gripper, R1D2 has 6 arm joints + gripper
# Joint 4 on R1D2 is skipped when mapping from SO101
SO101_TO_REALMAN_JOINT_MAP = {
    "shoulder_pan": 0,   # SO101 joint 1 -> R1D2 joint_1 [-178°, 178°]
    "shoulder_lift": 1,  # SO101 joint 2 -> R1D2 joint_2 [-130°, 130°]
    "elbow_flex": 2,     # SO101 joint 3 -> R1D2 joint_3 [-135°, 135°]
    # R1D2 joint_4 (index 3) is fixed at center position (0°)
    "wrist_flex": 4,     # SO101 joint 4 -> R1D2 joint_5 [-128°, 128°]
    "wrist_roll": 5,     # SO101 joint 5 -> R1D2 joint_6 [-360°, 360°]
}

# Reverse mapping for observation conversion
REALMAN_TO_SO101_JOINT_MAP = {v: k for k, v in SO101_TO_REALMAN_JOINT_MAP.items()}

# RealMan joint names (indexed 0-5)
REALMAN_JOINT_NAMES = [
    "joint_1",
    "joint_2", 
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

# SO101-compatible joint names for action features (what leader sends)
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class RealManFollower(Robot):
    """
    RealMan robot follower for teleoperation with SO101 or similar leader arms.
    
    This class integrates RealMan robotic arms (R1D2, RM65, RM75, etc.) into the
    LeRobot framework, enabling teleoperation, recording, and replay functionality.
    
    Key features:
    - Automatic joint mapping from 5-joint SO101 leader to 6-joint RealMan arm
    - Joint 4 held at fixed position during SO101 teleoperation
    - Integrated gripper control
    - Safety limits and collision detection
    - Camera support for recording
    """

    config_class = RealManFollowerConfig
    name = "realman_follower"

    def __init__(self, config: RealManFollowerConfig):
        # Don't call super().__init__ yet - we need to set up calibration path first
        # because RealMan uses a different calibration format
        self.robot_type = self.name
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir if config.calibration_dir 
            else Path.home() / ".cache/huggingface/lerobot/calibration/robots" / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"
        
        # RealMan calibration is just home position, not MotorCalibration
        self.calibration: dict = {}
        if self.calibration_fpath.is_file():
            self._load_calibration()
        
        self.config = config
        self._connected = False
        self._robot_controller = None
        
        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Track last gripper position for state
        self._last_gripper_position = 500  # Middle position
        self._last_gripper_command = 500  # Last commanded gripper position
        self._last_joint_angles = None  # Cache for current joint angles
        self._last_joint_read_time = 0.0  # Time of last joint read
        
        logger.info(f"Initialized RealManFollower for {config.model} at {config.ip}:{config.port}")
        if config.invert_joints:
            msg = f"Joint inversions configured: {config.invert_joints}"
            logger.info(msg)
            print(f"🔄 {msg}")  # Also print to ensure it's visible
        
        # Check for min_z_position - use calibration value as fallback if config not set
        effective_min_z = config.min_z_position
        if effective_min_z is None and self.calibration.get("min_z_position"):
            effective_min_z = self.calibration["min_z_position"]
            # Store it in a runtime attribute so send_action can use it
            self._effective_min_z = effective_min_z
            msg = f"Z safety limit from calibration: min_z={effective_min_z:.3f}m, action={config.z_limit_action}"
            logger.info(msg)
            print(f"🛡️ {msg}")
        elif effective_min_z is not None:
            self._effective_min_z = effective_min_z
            msg = f"Z safety limit from config: min_z={effective_min_z:.3f}m, action={config.z_limit_action}"
            logger.info(msg)
            print(f"🛡️ {msg}")
        else:
            self._effective_min_z = None

    def _load_calibration(self, fpath: Path | None = None) -> None:
        """Load RealMan calibration (home position) from file."""
        fpath = self.calibration_fpath if fpath is None else fpath
        try:
            with open(fpath) as f:
                self.calibration = json.load(f)
            logger.info(f"Loaded calibration from {fpath}")
        except Exception as e:
            logger.warning(f"Could not load calibration: {e}")
            self.calibration = {}

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """Save RealMan calibration (home position) to file."""
        fpath = self.calibration_fpath if fpath is None else fpath
        with open(fpath, "w") as f:
            json.dump(self.calibration, f, indent=4)
        logger.info(f"Calibration saved to {fpath}")

    def _get_robot_controller(self):
        """Lazy import and create RobotController from local module."""
        if self._robot_controller is None:
            try:
                from .robot_controller import RobotController
            except ImportError as e:
                raise ImportError(
                    "RobotController module not found. This is a required component of realman_follower.\n"
                    "Please ensure robot_controller.py exists in the realman_follower directory."
                ) from e
            
            self._robot_controller = RobotController(
                ip=self.config.ip,
                port=self.config.port,
                model=self.config.model,
                dof=self.config.dof,
            )
        return self._robot_controller

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features using SO101-compatible naming for action compatibility."""
        return {f"{joint}.pos": float for joint in SO101_JOINT_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Features returned by get_observation()."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Features expected by send_action() - SO101-compatible format."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot and cameras are connected."""
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        return self._connected and cameras_connected

    @property
    def is_calibrated(self) -> bool:
        """RealMan uses absolute encoders, so always calibrated."""
        return True

    def _log_joint_mapping_info(self) -> None:
        """Log the joint mapping configuration for debugging."""
        logger.info("=" * 60)
        logger.info("RealMan R1D2 Joint Mapping Configuration")
        logger.info("=" * 60)
        logger.info("SO101 Leader → RealMan R1D2 Follower Mapping:")
        logger.info("-" * 60)
        
        for so101_name, realman_idx in SO101_TO_REALMAN_JOINT_MAP.items():
            joint_name = REALMAN_JOINT_NAMES[realman_idx]
            if joint_name in self.config.joint_limits:
                min_deg, max_deg = self.config.joint_limits[joint_name]
                logger.info(
                    f"  {so101_name:15} (-100..100) → {joint_name} [{min_deg:>7.1f}°, {max_deg:>7.1f}°]"
                )
        
        # Log fixed joint 4
        if "joint_4" in self.config.joint_limits:
            min_deg, max_deg = self.config.joint_limits["joint_4"]
            logger.info(
                f"  {'(FIXED)':15}            → joint_4 [{min_deg:>7.1f}°, {max_deg:>7.1f}°] = {self.config.fixed_joint_4_position}°"
            )
        
        # Log gripper
        if "gripper" in self.config.joint_limits:
            min_grip, max_grip = self.config.joint_limits["gripper"]
            logger.info(
                f"  {'gripper':15} (0..100)    → gripper [{min_grip:>7.0f}, {max_grip:>7.0f}]"
            )
        
        logger.info("-" * 60)
        logger.info(f"Joint limit enforcement: {'ENABLED' if self.config.enforce_joint_limits else 'DISABLED'}")
        logger.info(f"Max relative target: {self.config.max_relative_target}°/step")
        logger.info("=" * 60)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the RealMan robot.
        
        Args:
            calibrate: If True, move to home position after connecting.
        """
        logger.info(f"Connecting to RealMan {self.config.model} at {self.config.ip}:{self.config.port}")
        
        robot = self._get_robot_controller()
        
        if not robot.connect():
            raise ConnectionError(
                f"Failed to connect to RealMan robot at {self.config.ip}:{self.config.port}"
            )
        
        self._connected = True
        
        # Log joint mapping configuration
        self._log_joint_mapping_info()
        
        # Set collision detection level (with fallback for API differences)
        if self.config.collision_level > 0:
            try:
                robot.set_collision_level(self.config.collision_level)
                logger.info(f"Collision detection set to level {self.config.collision_level}")
            except AttributeError as e:
                logger.warning(f"Could not set collision level (API not available): {e}")
        
        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()
        
        # Optionally move to home/calibration position
        if calibrate:
            self.calibrate()
        
        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        """
        Calibrate the RealMan robot by recording joint ranges.
        
        This calibration records:
        1. Center position (matching SO101 center/home position)
        2. Min/Max positions for each joint
        
        This allows proper mapping from SO101 normalized values to RealMan degrees.
        """
        if not self._connected:
            logger.warning("Robot not connected, skipping calibration")
            return
        
        robot = self._get_robot_controller()
        
        # Check if calibration file exists
        if self.calibration and "joint_ranges" in self.calibration:
            user_input = input(
                f"Calibration exists for {self.id}. Press ENTER to use it, "
                "or type 'c' to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration for {self.id}")
                return
        
        print("\n" + "=" * 60)
        print("RealMan Follower Calibration")
        print("=" * 60)
        print("\nThis calibration will record joint positions to map properly")
        print("with the SO101 leader arm.\n")
        print("📊 Real-time joint positions will be displayed while you move the arm.")
        print("   Press ENTER when the arm is in the desired position.\n")
        
        def wait_with_live_positions(prompt: str, highlight_joint: int = None) -> list[float]:
            """Wait for user input while showing live joint positions and end effector Z."""
            import sys
            import select
            import termios
            import tty
            
            print(prompt)
            print("   (Positions update in real-time. Press ENTER when ready)")
            print()
            
            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                # Set terminal to raw mode for non-blocking input
                tty.setcbreak(sys.stdin.fileno())
                
                while True:
                    # Check for keypress (non-blocking)
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == '\n' or key == '\r':
                            break
                    
                    # Get current joint positions
                    positions = robot.get_current_joint_angles()
                    # Get current end effector pose for Z display
                    ee_pose = robot.get_current_pose()
                    
                    if positions:
                        # Build position display string
                        pos_strs = []
                        for i, pos in enumerate(positions):
                            joint_name = f"J{i+1}"
                            if highlight_joint is not None and i == highlight_joint:
                                # Highlight the joint being calibrated
                                pos_strs.append(f"[{joint_name}:{pos:+7.2f}°]")
                            else:
                                pos_strs.append(f"{joint_name}:{pos:+7.2f}°")
                        
                        # Add end effector Z position
                        z_str = f"Z={ee_pose[2]:.3f}m" if ee_pose else "Z=?.???m"
                        
                        # Print on same line (carriage return)
                        print(f"\r   📍 {' | '.join(pos_strs)} | 🎯 {z_str}   ", end='', flush=True)
                
                print()  # New line after ENTER
                return robot.get_current_joint_angles()
            finally:
                # Restore terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        # Step 1: Record center position
        print("STEP 1: CENTER POSITION")
        print("-" * 40)
        center_pos = wait_with_live_positions(
            "Move the RealMan arm to its CENTER/HOME position.\n"
            "This should match where SO101 outputs 0 for all joints."
        )
        if center_pos is None:
            logger.error("Failed to read joint positions")
            return
        
        print(f"✅ Center joint positions recorded: {[f'{p:.1f}°' for p in center_pos]}")
        
        # Also record center end effector position for Z safety checks
        center_ee_pose = robot.get_current_pose()
        if center_ee_pose:
            print(f"✅ Center end effector position: X={center_ee_pose[0]:.3f}m, Y={center_ee_pose[1]:.3f}m, Z={center_ee_pose[2]:.3f}m")
        
        # Step 2: Record min/max for each mapped joint
        print("\nSTEP 2: JOINT RANGES")
        print("-" * 40)
        print("For each joint, move it to its MIN and MAX positions.")
        print("This establishes the range that maps to SO101's -100 to +100.\n")
        
        joint_ranges = {}
        
        # Map of SO101 names to RealMan indices we need to calibrate
        joints_to_calibrate = [
            ("shoulder_pan", 0, "joint_1"),
            ("shoulder_lift", 1, "joint_2"),
            ("elbow_flex", 2, "joint_3"),
            ("wrist_flex", 4, "joint_5"),
            ("wrist_roll", 5, "joint_6"),
        ]
        
        for so101_name, realman_idx, realman_name in joints_to_calibrate:
            print(f"\n--- Calibrating {realman_name} (maps to SO101 {so101_name}) ---")
            
            min_pos = wait_with_live_positions(
                f"Move {realman_name} to its MINIMUM position",
                highlight_joint=realman_idx
            )
            if min_pos is None:
                logger.error("Failed to read joint positions")
                return
            min_val = min_pos[realman_idx]
            print(f"   📌 MIN recorded: {min_val:.2f}°")
            
            max_pos = wait_with_live_positions(
                f"Move {realman_name} to its MAXIMUM position",
                highlight_joint=realman_idx
            )
            if max_pos is None:
                logger.error("Failed to read joint positions")
                return
            max_val = max_pos[realman_idx]
            print(f"   📌 MAX recorded: {max_val:.2f}°")
            
            # Ensure min < max
            if min_val > max_val:
                min_val, max_val = max_val, min_val
                print("   ⚠️  Swapped min/max (min was greater than max)")
            
            joint_ranges[realman_name] = {
                "min": min_val,
                "max": max_val,
                "center": center_pos[realman_idx],
                "so101_name": so101_name,
                "realman_idx": realman_idx,
            }
            
            print(f"   ✅ {realman_name}: [{min_val:.1f}°, {max_val:.1f}°], center: {center_pos[realman_idx]:.1f}°")
        
        # Save calibration
        self.calibration = {
            "home_position": center_pos,  # Joint angles at center position
            "center_ee_pose": center_ee_pose,  # End effector Cartesian pose [x, y, z, rx, ry, rz]
            "joint_ranges": joint_ranges,
        }
        self._save_calibration()
        
        print("\n" + "=" * 60)
        print("Calibration complete!")
        print("=" * 60)
        print(f"Saved to: {self.calibration_fpath}")
        
        # Return to center position
        input("\nPress ENTER to move robot back to center position...")
        robot.movej(center_pos, velocity=self.config.velocity, block=True)
        print("Robot at center position.")
        
        # Step 3: Configure Z safety limit
        print("\n" + "=" * 60)
        print("STEP 3: Z SAFETY LIMIT (Optional)")
        print("=" * 60)
        print("\nSet a minimum Z height to prevent the arm from going too low.")
        print("This protects against collisions with the table/workspace.\n")
        
        current_pose = robot.get_current_pose()
        current_z = current_pose[2] if current_pose else 0.0
        print(f"   Current end effector Z position: {current_z:.3f}m")
        print(f"   Current min_z_position config:   {self.config.min_z_position}")
        
        print("\nOptions:")
        print("   1. Move arm to the LOWEST safe position, then press 1")
        print("   2. Enter a custom Z value (in meters)")
        print("   3. Keep current config / skip (press ENTER)")
        
        choice = input("\nChoice [1/2/ENTER]: ").strip()
        
        if choice == "1":
            print("\n🔧 INTERACTIVE Z LIMIT CALIBRATION")
            print("-" * 40)
            print("Move the end effector DOWN until it is just a few millimeters")
            print("ABOVE the table/ground surface. This position will be used as")
            print("the safety limit to prevent collisions.")
            print("")
            print("💡 TIP: Position the gripper ~5-10mm above the table surface.")
            
            # Show live Z position while user moves the arm
            import sys
            import select
            import termios
            import tty
            
            print("\n   (Z position updates in real-time. Press ENTER when positioned)")
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                while True:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key == '\n' or key == '\r':
                            break
                    
                    pose = robot.get_current_pose()
                    if pose:
                        z_mm = pose[2] * 1000  # Convert to mm for easier reading
                        print(f"\r   🎯 Current Z: {pose[2]:.4f}m ({z_mm:.1f}mm)   ", end='', flush=True)
                
                print()
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
            # Record the Z position
            final_pose = robot.get_current_pose()
            if final_pose:
                recorded_z = final_pose[2]
                recorded_z_mm = recorded_z * 1000
                
                print(f"\n   📌 Recorded end effector Z: {recorded_z:.4f}m ({recorded_z_mm:.1f}mm)")
                
                # Ask for additional safety margin
                print("\n   How much ADDITIONAL safety margin would you like? (default: 0mm)")
                print("   This will be added to the recorded position.")
                margin_input = input("   Enter margin in mm [0]: ").strip()
                
                try:
                    margin_mm = float(margin_input) if margin_input else 0.0
                except ValueError:
                    margin_mm = 0.0
                    print("   (Invalid input, using 0mm margin)")
                
                margin_m = margin_mm / 1000.0
                min_z_final = recorded_z + margin_m
                min_z_final_mm = min_z_final * 1000
                
                print(f"\n   📐 Recorded Z:      {recorded_z:.4f}m ({recorded_z_mm:.1f}mm)")
                print(f"   📐 Safety margin:  +{margin_m:.4f}m ({margin_mm:.1f}mm)")
                print(f"   📐 Final min_z:     {min_z_final:.4f}m ({min_z_final_mm:.1f}mm)")
                
                # Save to calibration
                self.calibration["min_z_position"] = min_z_final
                self._save_calibration()
                
                print(f"\n   ✅ min_z_position saved to calibration file!")
                print(f"\n   ⚠️  To ENABLE this limit during teleoperation, update realman_config.yaml:")
                print(f"       safety:")
                print(f"         min_z_position: {min_z_final:.4f}")
        
        elif choice == "2":
            try:
                custom_z = float(input("   Enter min Z value (meters, e.g., 0.05): ").strip())
                self.calibration["min_z_position"] = custom_z
                self._save_calibration()
                print(f"\n   ✅ min_z_position saved to calibration file: {custom_z:.4f}m")
                print(f"   ⚠️  To enable, also set 'min_z_position: {custom_z:.3f}' in realman_config.yaml")
            except ValueError:
                print("   ❌ Invalid input, skipping Z limit configuration.")
        
        else:
            print("   ⏭️  Skipped Z safety limit configuration.")

    def configure(self) -> None:
        """Apply runtime configuration to the robot."""
        # Configuration is handled during connect()
        pass

    def _clamp_to_joint_limits(self, joint_idx: int, value: float) -> float:
        """
        Clamp a joint value to its configured limits.
        
        Args:
            joint_idx: Index of the joint (0-5 for R1D2)
            value: Desired joint value in degrees
            
        Returns:
            Clamped value within joint limits
        """
        joint_name = REALMAN_JOINT_NAMES[joint_idx]
        if joint_name in self.config.joint_limits:
            min_deg, max_deg = self.config.joint_limits[joint_name]
            clamped = max(min_deg, min(max_deg, value))
            if clamped != value:
                logger.warning(
                    f"Joint {joint_name} value {value:.2f}° clamped to [{min_deg}, {max_deg}] -> {clamped:.2f}°"
                )
            return clamped
        return value

    def _get_joint_range(self, so101_name: str, realman_idx: int) -> tuple[float, float, float]:
        """
        Get the calibrated range for a joint.
        
        Returns:
            Tuple of (min_deg, max_deg, center_deg)
        """
        joint_name = REALMAN_JOINT_NAMES[realman_idx]
        
        # First check if we have calibration data
        if self.calibration and "joint_ranges" in self.calibration:
            joint_cal = self.calibration["joint_ranges"].get(joint_name)
            if joint_cal:
                return (joint_cal["min"], joint_cal["max"], joint_cal["center"])
        
        # Fall back to config joint limits
        if joint_name in self.config.joint_limits:
            min_deg, max_deg = self.config.joint_limits[joint_name]
            center_deg = (min_deg + max_deg) / 2.0
            return (min_deg, max_deg, center_deg)
        
        # Default fallback
        return (-180.0, 180.0, 0.0)

    def _so101_normalized_to_realman_degrees(self, so101_name: str, normalized_value: float) -> float:
        """
        Convert SO101 normalized value (-100 to 100) to RealMan degrees.
        
        SO101's normalized values come from raw encoder values (0-4095) that are
        normalized based on calibration (range_min, range_max). The STS3215 motors
        use 4096 counts per full 360° revolution.
        
        SO101 normalized mapping:
        - -100 corresponds to range_min encoder position
        - +100 corresponds to range_max encoder position
        - 0 is the midpoint (where homing was done)
        
        We convert this to physical degrees first, then map to RealMan's range.
        
        Args:
            so101_name: SO101 joint name (e.g., "shoulder_pan")
            normalized_value: Value in range [-100, 100]
            
        Returns:
            Joint angle in degrees for RealMan
        """
        realman_idx = SO101_TO_REALMAN_JOINT_MAP.get(so101_name)
        if realman_idx is None:
            return normalized_value
        
        # Clamp input to valid range
        normalized_value = max(-100.0, min(100.0, normalized_value))
        
        # Check if joint should be inverted
        joint_name = REALMAN_JOINT_NAMES[realman_idx]
        should_invert = self.config.invert_joints.get(joint_name, False)
        if should_invert:
            original_value = normalized_value
            normalized_value = -normalized_value
            msg = f"🔄 Inverting {so101_name} ({joint_name}): {original_value:+6.1f} -> {normalized_value:+6.1f}"
            logger.debug(msg)
            # Print first few times for visibility
            if not hasattr(self, '_invert_print_count'):
                self._invert_print_count = {}
            if self._invert_print_count.get(joint_name, 0) < 3:
                print(msg)
                self._invert_print_count[joint_name] = self._invert_print_count.get(joint_name, 0) + 1
        
        # Check if we have RealMan calibration data
        if self.calibration and "joint_ranges" in self.calibration:
            joint_cal = self.calibration["joint_ranges"].get(joint_name)
            if joint_cal:
                min_deg = joint_cal["min"]
                max_deg = joint_cal["max"]
                center_deg = joint_cal["center"]
                
                # Map SO101 normalized (-100 to 100) to RealMan calibrated range
                # -100 -> min_deg, 0 -> center_deg, +100 -> max_deg
                if normalized_value < 0:
                    # Map -100..0 to min_deg..center_deg
                    degrees = center_deg + (normalized_value / 100.0) * (center_deg - min_deg)
                else:
                    # Map 0..100 to center_deg..max_deg
                    degrees = center_deg + (normalized_value / 100.0) * (max_deg - center_deg)
                
                logger.debug(
                    f"SO101→RealMan {so101_name}({joint_name}): "
                    f"norm={normalized_value:+6.1f} → deg={degrees:+7.2f}° "
                    f"[range: {min_deg:+6.1f}° to {max_deg:+6.1f}°, center={center_deg:+6.1f}°]"
                )
                
                return degrees
        
        # Fallback: Direct mapping using config limits
        # This assumes SO101 and RealMan have similar ranges (not ideal)
        joint_name = REALMAN_JOINT_NAMES[realman_idx]
        if joint_name in self.config.joint_limits:
            min_deg, max_deg = self.config.joint_limits[joint_name]
            # Linear mapping: -100 -> min_deg, +100 -> max_deg
            degrees = min_deg + (normalized_value + 100.0) / 200.0 * (max_deg - min_deg)
            return degrees
        
        return normalized_value

    def _realman_degrees_to_so101_normalized(self, realman_idx: int, degrees: float) -> float:
        """
        Convert RealMan degrees to SO101 normalized value (-100 to 100).
        
        Args:
            realman_idx: RealMan joint index (0-5)
            degrees: Joint angle in degrees
            
        Returns:
            Normalized value in range [-100, 100]
        """
        # Get SO101 name for this realman index
        so101_name = REALMAN_TO_SO101_JOINT_MAP.get(realman_idx)
        if so101_name is None:
            return degrees
        
        joint_name = REALMAN_JOINT_NAMES[realman_idx]
        
        # Check if we have RealMan calibration data
        if self.calibration and "joint_ranges" in self.calibration:
            joint_cal = self.calibration["joint_ranges"].get(joint_name)
            if joint_cal:
                min_deg = joint_cal["min"]
                max_deg = joint_cal["max"]
                center_deg = joint_cal["center"]
                
                # Clamp degrees to calibrated range
                degrees = max(min_deg, min(max_deg, degrees))
                
                # Map RealMan degrees to normalized (-100 to 100)
                if degrees < center_deg:
                    # Map min_deg..center_deg to -100..0
                    if center_deg != min_deg:
                        normalized = -100.0 * (center_deg - degrees) / (center_deg - min_deg)
                    else:
                        normalized = 0.0
                else:
                    # Map center_deg..max_deg to 0..100
                    if max_deg != center_deg:
                        normalized = 100.0 * (degrees - center_deg) / (max_deg - center_deg)
                    else:
                        normalized = 0.0
                
                # Apply inversion if configured
                should_invert = self.config.invert_joints.get(joint_name, False)
                if should_invert:
                    original_normalized = normalized
                    normalized = -normalized
                    logger.debug(
                        f"RealMan→SO101 {joint_name}: "
                        f"deg={degrees:+7.2f}° → norm={original_normalized:+6.1f} (inverted→{normalized:+6.1f})"
                    )
                else:
                    logger.debug(
                        f"RealMan→SO101 {joint_name}: "
                        f"deg={degrees:+7.2f}° → norm={normalized:+6.1f}"
                    )
                
                return normalized
        
        # Fallback: Use config limits
        joint_name = REALMAN_JOINT_NAMES[realman_idx]
        if joint_name in self.config.joint_limits:
            min_deg, max_deg = self.config.joint_limits[joint_name]
            
            if max_deg == min_deg:
                return 0.0
            
            degrees = max(min_deg, min(max_deg, degrees))
            normalized = (degrees - min_deg) / (max_deg - min_deg) * 200.0 - 100.0
            return normalized
        
        return degrees

    def _convert_so101_action_to_realman(self, action: RobotAction) -> tuple[list[float], float | None]:
        """
        Convert SO101-format action to RealMan joint angles with limit enforcement.
        
        Args:
            action: Action dict with SO101 joint names (e.g., "shoulder_pan.pos")
            
        Returns:
            Tuple of (joint_angles_list, gripper_position)
            joint_angles_list has 6 values for R1D2 (indices 0-5)
            All values are guaranteed to be within joint limits.
        """
        # Start with cached joint positions as base (avoid extra read)
        current_joints = self._last_joint_angles
        
        if current_joints is None:
            # Fall back to reading if no cache
            robot = self._get_robot_controller()
            current_joints = robot.get_current_joint_angles()
        
        if current_joints is None:
            # Use zeros if can't read current position
            current_joints = [0.0] * self.config.dof
        
        # Copy current positions
        target_joints = list(current_joints)
        
        # Map SO101 joints to RealMan joints
        for so101_name, realman_idx in SO101_TO_REALMAN_JOINT_MAP.items():
            key = f"{so101_name}.pos"
            if key in action:
                normalized_value = action[key]
                
                # Convert from SO101 normalized range (-100 to 100) to degrees
                degrees = self._so101_normalized_to_realman_degrees(so101_name, normalized_value)
                
                # Enforce joint limits if configured
                if self.config.enforce_joint_limits:
                    degrees = self._clamp_to_joint_limits(realman_idx, degrees)
                
                target_joints[realman_idx] = degrees
        
        # Set joint 4 (index 3) to fixed position - also enforce limits
        if len(target_joints) > 3:
            fixed_pos = self.config.fixed_joint_4_position
            if self.config.enforce_joint_limits:
                fixed_pos = self._clamp_to_joint_limits(3, fixed_pos)
            target_joints[3] = fixed_pos
        
        # Handle gripper
        gripper_pos = None
        gripper_key = "gripper.pos"
        if gripper_key in action:
            gripper_value = action[gripper_key]
            
            # SO101 gripper uses RANGE_0_100 mode (0-100)
            # RealMan gripper uses 1-1000 range
            # Clamp input first
            gripper_value = max(0.0, min(100.0, gripper_value))
            
            # Map 0-100 to 1-1000
            gripper_pos = int(1 + gripper_value * 9.99)
            
            # Enforce gripper limits from config
            if "gripper" in self.config.joint_limits:
                min_grip, max_grip = self.config.joint_limits["gripper"]
                gripper_pos = max(int(min_grip), min(int(max_grip), gripper_pos))
        
        return target_joints, gripper_pos

    def _convert_realman_observation_to_so101(self, joint_angles: list[float], gripper_pos: int) -> dict[str, float]:
        """
        Convert RealMan joint angles to SO101-format observation.
        
        Args:
            joint_angles: List of 6 joint angles in degrees
            gripper_pos: Gripper position (1-1000)
            
        Returns:
            Observation dict with SO101 joint names, values in [-100, 100]
        """
        obs = {}
        
        # Map RealMan joints back to SO101 names
        for so101_name, realman_idx in SO101_TO_REALMAN_JOINT_MAP.items():
            if realman_idx < len(joint_angles):
                degrees = joint_angles[realman_idx]
                
                # Convert degrees to normalized range (-100 to 100)
                normalized = self._realman_degrees_to_so101_normalized(realman_idx, degrees)
                
                obs[f"{so101_name}.pos"] = normalized
        
        # Convert gripper (1-1000 -> 0-100)
        if gripper_pos is not None:
            # Clamp to valid range first
            gripper_pos = max(1, min(1000, int(gripper_pos)))
            obs["gripper.pos"] = (gripper_pos - 1) / 9.99
        else:
            obs["gripper.pos"] = 50.0  # Default to middle
        
        return obs

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Get current robot state and camera images.
        
        Returns:
            Observation dict with joint positions and camera images.
        """
        start = time.perf_counter()
        
        robot = self._get_robot_controller()
        
        # Get joint angles (with caching to avoid redundant reads)
        now = time.perf_counter()
        if self._last_joint_angles is None or (now - self._last_joint_read_time) > 0.005:
            # Only read if cache is stale (>5ms old) or empty
            joint_angles = robot.get_current_joint_angles()
            if joint_angles is not None:
                self._last_joint_angles = joint_angles
                self._last_joint_read_time = now
            else:
                logger.warning("Failed to read joint angles")
                joint_angles = self._last_joint_angles or [0.0] * self.config.dof
        else:
            joint_angles = self._last_joint_angles
        
        # Use tracked gripper position (skip slow gripper_get_state() call)
        # The gripper position is updated when we send commands
        gripper_pos = self._last_gripper_position
        
        # Convert to SO101-compatible format
        obs_dict = self._convert_realman_observation_to_so101(joint_angles, gripper_pos)
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send action command to the robot.
        
        Args:
            action: Action dict with SO101-compatible joint names.
            
        Returns:
            The action that was actually sent (may be clipped for safety).
        """
        start = time.perf_counter()
        
        robot = self._get_robot_controller()
        
        # Convert SO101 action to RealMan format
        target_joints, gripper_pos = self._convert_so101_action_to_realman(action)
        
        # Log the action mapping summary
        action_summary = []
        for so101_name, realman_idx in SO101_TO_REALMAN_JOINT_MAP.items():
            key = f"{so101_name}.pos"
            if key in action:
                action_summary.append(f"{so101_name}={action[key]:+6.1f}→{target_joints[realman_idx]:+7.2f}°")
        if action_summary:
            msg = f"Action: {', '.join(action_summary)}"
            logger.debug(msg)
            # Print first few actions for visibility
            if not hasattr(self, '_action_print_count'):
                self._action_print_count = 0
            if self._action_print_count < 5:
                print(f"📤 {msg}")
                self._action_print_count += 1
        
        # Apply safety limits if configured - use cached joint angles for speed
        if self.config.max_relative_target is not None:
            current_joints = self._last_joint_angles
            if current_joints is None:
                # Fall back to reading if no cache
                current_joints = robot.get_current_joint_angles()
            
            if current_joints is not None:
                max_delta = self.config.max_relative_target
                if isinstance(max_delta, (int, float)):
                    for i in range(len(target_joints)):
                        delta = target_joints[i] - current_joints[i]
                        if abs(delta) > max_delta:
                            target_joints[i] = current_joints[i] + max_delta * (1 if delta > 0 else -1)
                            logger.debug(f"Clamped joint {i} delta from {delta:.2f} to {max_delta}")
        
        # Check Z position safety limit if configured (from config or calibration)
        effective_min_z = getattr(self, '_effective_min_z', None)
        if effective_min_z is not None:
            current_pose = robot.get_current_pose()
            if current_pose:
                current_z = current_pose[2]  # Z is the 3rd element [x, y, z, rx, ry, rz]
                min_z = effective_min_z
                
                # Track the last "safe" joint positions (when Z was above limit)
                if not hasattr(self, '_last_safe_joints'):
                    self._last_safe_joints = list(target_joints)
                    self._last_safe_z = current_z
                
                if current_z > min_z:
                    # ABOVE limit - update safe position and clear warning
                    self._last_safe_joints = list(self._last_joint_angles or target_joints)
                    self._last_safe_z = current_z
                    if hasattr(self, '_z_limit_active'):
                        del self._z_limit_active
                        print(f"✅ Z limit cleared: Z={current_z:.3f}m")
                else:
                    # AT or BELOW limit - enforce safety
                    if not hasattr(self, '_z_limit_active'):
                        self._z_limit_active = True
                        logger.warning(
                            f"⚠️ Z position safety limit! Z={current_z:.3f}m, Limit={min_z:.3f}m"
                        )
                        print(f"⚠️ Z SAFETY: Z={current_z:.3f}m at limit ({min_z:.3f}m)")
                    
                    # REACTIVE SAFETY: Use last safe joints as base, but allow upward motion
                    # Compare target Z tendency: if moving joints would lower Z, use safe joints
                    # Since we can't predict FK, we use the safe joints as the command
                    # but allow individual joints to move if they're moving AWAY from limit direction
                    
                    if self._last_safe_joints:
                        # Blend: use safe joints as baseline
                        safe_joints = self._last_safe_joints
                        
                        # Allow joints to move ONLY if they would increase Z
                        # Heuristic: joints 1,2,3 (shoulder/elbow) most affect Z
                        # If current Z is at limit, only allow joint changes that 
                        # moved us UP last time
                        
                        # Simple approach: hold position at safe joints
                        # but allow small movements to test if Z increases
                        for i in range(len(target_joints)):
                            delta = target_joints[i] - safe_joints[i]
                            # Allow small movements (< 2 degrees) to "feel" for safe direction
                            if abs(delta) > 2.0:
                                target_joints[i] = safe_joints[i] + (2.0 if delta > 0 else -2.0)
                        
                        # After this limited movement, the next cycle will check Z again
                        # If Z improved, safe_joints will update; if not, we stay clamped
        
        # Send joint command using CANFD for low-latency teleoperation
        # follow=False means immediately override current motion (better responsiveness)
        result = robot.movej_canfd(target_joints, follow=False)
        if result != 0:
            logger.warning(f"movej_canfd returned non-zero status: {result}")
        
        # Send gripper command only if position changed significantly (reduces latency)
        if gripper_pos is not None:
            gripper_change = abs(gripper_pos - self._last_gripper_command)
            if gripper_change > 10:  # Only send if >1% change (10/1000)
                robot.gripper_set_position(gripper_pos, block=False)
                self._last_gripper_command = gripper_pos
                self._last_gripper_position = gripper_pos
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} send action: {dt_ms:.1f}ms")
        
        # Return the action in SO101 format (what was actually commanded)
        return action

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect from the robot and cameras."""
        robot = self._get_robot_controller()
        
        # Stop any ongoing motion
        robot.stop()
        
        # Optionally move to safe position before disconnecting
        if self.config.disable_torque_on_disconnect:
            # For RealMan, we don't disable torque but can move to home
            pass
        
        # Disconnect from robot
        robot.disconnect()
        self._connected = False
        
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        logger.info(f"{self} disconnected.")

    def move_to_home(self, velocity: int | None = None) -> None:
        """
        Move robot to home position.
        
        Args:
            velocity: Movement velocity (1-100). Uses config default if None.
        """
        if not self._connected:
            logger.warning("Robot not connected")
            return
        
        robot = self._get_robot_controller()
        vel = velocity if velocity is not None else self.config.velocity
        
        result = robot.move_to_home(velocity=vel)
        if result != 0:
            logger.warning(f"move_to_home returned non-zero status: {result}")

    def stop(self) -> None:
        """Emergency stop - immediately halt all motion."""
        if self._robot_controller is not None:
            self._robot_controller.stop()
            logger.warning("Emergency stop triggered")

