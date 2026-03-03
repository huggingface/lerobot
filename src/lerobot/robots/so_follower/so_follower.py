#!/usr/bin/env python

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

import logging
import time
from functools import cached_property
from typing import TypeAlias

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so_follower import SOFollowerRobotConfig

logger = logging.getLogger(__name__)


class SOFollower(Robot):
    """
    Generic SO follower base implementing common functionality for SO-100/101/10X.
    Designed to be subclassed with a per-hardware-model `config_class` and `name`.
    """

    config_class = SOFollowerRobotConfig
    name = "so_follower"

    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        # choose normalization mode depending on config if available
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def _display_joint_positions_live(self, timeout_seconds: float = 60.0) -> bool:
        """
        Display live joint positions with color-coded feedback.
        Green = position is in valid range for calibration (magnitude <= 2047)
        Red = position is out of range and needs adjustment
        
        Returns True when user presses Enter and all joints are in valid range.
        """
        import sys
        import threading
        
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.live import Live
            from rich.text import Text
            has_rich = True
        except ImportError:
            has_rich = False
        
        # The valid range for calibration: magnitude from center (2048) must be <= 2047
        CENTER = 2048
        MAX_MAGNITUDE = 2047
        
        # Event to signal when user presses Enter
        enter_pressed = threading.Event()
        user_input_value = [None]
        
        def wait_for_enter():
            user_input_value[0] = input()
            enter_pressed.set()
        
        input_thread = threading.Thread(target=wait_for_enter, daemon=True)
        input_thread.start()
        
        if has_rich:
            console = Console()
            
            with Live(console=console, refresh_per_second=4, transient=True) as live:
                while not enter_pressed.is_set():
                    # Read current positions
                    try:
                        positions = self.bus.sync_read("Present_Position", normalize=False)
                    except Exception:
                        time.sleep(0.1)
                        continue
                    
                    # Create table
                    table = Table(title=f"🔧 Joint Positions - {self.id}", show_header=True)
                    table.add_column("Joint", style="bold")
                    table.add_column("Position", justify="right")
                    table.add_column("From Center", justify="right")
                    table.add_column("Status", justify="center")
                    
                    all_valid = True
                    for motor, pos in positions.items():
                        magnitude = abs(pos - CENTER)
                        
                        if magnitude <= MAX_MAGNITUDE:
                            status = Text("✓ OK", style="bold green")
                            pos_style = "green"
                            mag_style = "green"
                        else:
                            status = Text("✗ MOVE", style="bold red")
                            pos_style = "red"
                            mag_style = "red"
                            all_valid = False
                        
                        table.add_row(
                            motor,
                            Text(str(pos), style=pos_style),
                            Text(f"{magnitude:+d}" if pos >= CENTER else f"{-magnitude:+d}", style=mag_style),
                            status
                        )
                    
                    # Add footer with instructions
                    if all_valid:
                        footer = Text("\n✅ All joints in valid range! Press ENTER to continue...", style="bold green")
                    else:
                        footer = Text("\n⚠️  Move RED joints closer to center (target: 2048). Press ENTER when ready...", style="bold yellow")
                    
                    # Update display
                    from rich.panel import Panel
                    panel = Panel(table, subtitle=footer)
                    live.update(panel)
                    
                    time.sleep(0.25)
            
            # Check final positions
            positions = self.bus.sync_read("Present_Position", normalize=False)
            all_valid = all(abs(pos - CENTER) <= MAX_MAGNITUDE for pos in positions.values())
            
            if not all_valid:
                console.print("\n[yellow]⚠️  Warning: Some joints are out of optimal range. Calibration may fail.[/yellow]")
                console.print("[yellow]   Consider moving joints closer to center and re-running calibration.[/yellow]\n")
            
            return all_valid
        else:
            # Fallback without rich
            print("\nCurrent joint positions (target: ~2048, valid range: 1-4095):")
            while not enter_pressed.is_set():
                try:
                    positions = self.bus.sync_read("Present_Position", normalize=False)
                    pos_str = " | ".join([f"{m}: {p}" for m, p in positions.items()])
                    print(f"\r{pos_str}    ", end="", flush=True)
                except Exception:
                    pass
                time.sleep(0.25)
            print()
            return True

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        print(f"\n📍 Move {self} to the MIDDLE of its range of motion.")
        print("   Watch the live display below - GREEN means ready, RED means adjust position.\n")
        
        self._display_joint_positions_live()
        homing_offsets = self.bus.set_half_turn_homings()

        # Attempt to call record_ranges_of_motion with a reduced motor set when appropriate.
        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)  # 50% of max torque to avoid burnout
                    self.bus.write("Protection_Current", motor, 250)  # 50% of max current to avoid burnout
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


SO100Follower: TypeAlias = SOFollower
SO101Follower: TypeAlias = SOFollower
