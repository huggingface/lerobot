#!/usr/bin/env python

"""
Inference Logger for capturing robot state, policy outputs, and trajectory data during inference.
Saves data to CSV format for visualization and analysis.
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class InferenceLogger:
    """
    Logs comprehensive inference data including robot state, policy outputs, and trajectories.
    Data is saved to CSV files for easy visualization and analysis.
    """

    def __init__(self, output_dir: Path, robot_name: str = "robot", target_fps: float = 30.0):
        """
        Initialize the inference logger.

        Args:
            output_dir: Directory to save CSV files
            robot_name: Name identifier for the robot (used in filenames)
            target_fps: Target control loop frequency for timing analysis
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.robot_name = robot_name
        self.target_fps = target_fps

        # Initialize CSV files and writers
        self.csv_files = {}
        self.csv_writers = {}

        # Data buffers
        self.inference_data = []
        self.robot_state_data = []

        # Timing
        self.start_time = time.perf_counter()
        self.step_count = 0
        self.last_step_time = self.start_time  # Track time of last step for frequency calculation

        logger.info(f"InferenceLogger initialized. Saving to: {self.output_dir}")

    def _get_csv_writer(self, filename: str, headers: list[str]):
        """Get or create a CSV writer for the given filename."""
        if filename not in self.csv_files:
            filepath = self.output_dir / filename
            self.csv_files[filename] = open(filepath, "w", newline="")
            self.csv_writers[filename] = csv.writer(self.csv_files[filename])
            self.csv_writers[filename].writerow(headers)
            logger.info(f"Created CSV file: {filepath}")
        return self.csv_writers[filename]

    def log_robot_state(self, robot, observation: dict[str, Any], control_loop_freq: float = 0.0):
        """
        Log comprehensive robot state information.

        Args:
            robot: Robot instance
            observation: Current robot observation
            control_loop_freq: Control loop frequency in Hz (optional)
        """
        try:
            # Get timestamp
            timestamp = time.perf_counter() - self.start_time

            # Basic robot info
            state_data = {
                "timestamp": timestamp,
                "step": self.step_count,
                "robot_connected": robot.is_connected if hasattr(robot, "is_connected") else "unknown",
                "control_loop_freq_hz": control_loop_freq,
            }

            # Motor positions from observation
            motor_positions = {}
            for key, value in observation.items():
                if ".pos" in key and "image" not in key:
                    motor_positions[key] = (
                        float(value)
                        if isinstance(value, (int, float, np.ndarray, torch.Tensor))
                        else str(value)
                    )

            state_data.update(motor_positions)

            # Try to get additional motor status if available
            if hasattr(robot, "bus") and hasattr(robot.bus, "sync_read"):
                try:
                    # Get raw motor positions
                    raw_positions = robot.bus.sync_read("Present_Position")
                    for motor, pos in raw_positions.items():
                        state_data[f"{motor}_raw_pos"] = float(pos)

                    # Try to get motor temperatures
                    try:
                        temperatures = robot.bus.sync_read("Present_Temperature")
                        for motor, temp in temperatures.items():
                            state_data[f"{motor}_temp"] = float(temp)
                    except:
                        pass

                    # Try to get motor voltages
                    try:
                        voltages = robot.bus.sync_read("Present_Voltage")
                        for motor, volt in voltages.items():
                            state_data[f"{motor}_voltage"] = float(volt)
                    except:
                        pass

                    # Try to get motor currents
                    try:
                        currents = robot.bus.sync_read("Present_Current")
                        for motor, curr in currents.items():
                            state_data[f"{motor}_current"] = float(curr)
                    except:
                        pass

                except Exception as e:
                    logger.debug(f"Could not read additional motor status: {e}")

            # Camera info
            if hasattr(robot, "cameras"):
                state_data["num_cameras"] = len(robot.cameras)
                for cam_name in robot.cameras.keys():
                    if f"observation.images.{cam_name}" in observation:
                        img = observation[f"observation.images.{cam_name}"]
                        if hasattr(img, "shape"):
                            state_data[f"{cam_name}_img_shape"] = str(img.shape)

            # Write to CSV
            if not hasattr(self, "_robot_state_headers"):
                self._robot_state_headers = list(state_data.keys())

            writer = self._get_csv_writer(f"{self.robot_name}_robot_state.csv", self._robot_state_headers)

            # Ensure all headers have values
            row_data = [state_data.get(header, "") for header in self._robot_state_headers]
            writer.writerow(row_data)
            self.csv_files[f"{self.robot_name}_robot_state.csv"].flush()

        except Exception as e:
            logger.error(f"Error logging robot state: {e}")

    def log_policy_inference(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        policy_output: torch.Tensor | None = None,
        inference_time: float = 0.0,
        task: str | None = None,
        control_loop_freq: float = 0.0,
    ):
        """
        Log policy inference data including inputs, outputs, and timing.

        Args:
            observation: Robot observation used as policy input
            action: Action output from policy
            policy_output: Raw policy output tensor (optional)
            inference_time: Time taken for inference in seconds
            task: Task description (optional)
            control_loop_freq: Pre-calculated control loop frequency in Hz
        """
        try:
            timestamp = time.perf_counter() - self.start_time

            inference_data = {
                "timestamp": timestamp,
                "step": self.step_count,
                "inference_time_ms": inference_time * 1000,
                "control_loop_freq_hz": control_loop_freq,
                "task": task or "",
            }

            # Log observation features (non-image)
            for key, value in observation.items():
                if "image" not in key.lower():
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        if value.numel() == 1:  # Single value
                            inference_data[f"obs_{key}"] = float(value.item())
                        elif value.numel() <= 10:  # Small arrays
                            inference_data[f"obs_{key}"] = str(
                                value.tolist() if hasattr(value, "tolist") else value
                            )
                    elif isinstance(value, (int, float)):
                        inference_data[f"obs_{key}"] = float(value)

            # Log action outputs
            for key, value in action.items():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    inference_data[f"action_{key}"] = float(value.item() if hasattr(value, "item") else value)
                elif isinstance(value, (int, float)):
                    inference_data[f"action_{key}"] = float(value)
                else:
                    inference_data[f"action_{key}"] = str(value)

            # Log raw policy output stats if available
            if policy_output is not None and isinstance(policy_output, torch.Tensor):
                inference_data["policy_output_shape"] = str(list(policy_output.shape))
                inference_data["policy_output_mean"] = float(policy_output.mean().item())
                inference_data["policy_output_std"] = float(policy_output.std().item())
                inference_data["policy_output_min"] = float(policy_output.min().item())
                inference_data["policy_output_max"] = float(policy_output.max().item())

            # Write to CSV
            if not hasattr(self, "_inference_headers"):
                self._inference_headers = list(inference_data.keys())

            writer = self._get_csv_writer(f"{self.robot_name}_policy_inference.csv", self._inference_headers)

            # Ensure all headers have values
            row_data = [inference_data.get(header, "") for header in self._inference_headers]
            writer.writerow(row_data)
            self.csv_files[f"{self.robot_name}_policy_inference.csv"].flush()

        except Exception as e:
            logger.error(f"Error logging policy inference: {e}")

    def log_trajectory_point(
        self,
        robot_pose: dict[str, float] | None = None,
        target_pose: dict[str, float] | None = None,
        error_metrics: dict[str, float] | None = None,
    ):
        """
        Log trajectory information including current pose, target pose, and error metrics.

        Args:
            robot_pose: Current robot pose/position
            target_pose: Target robot pose/position
            error_metrics: Error metrics like position error, orientation error, etc.
        """
        try:
            timestamp = time.perf_counter() - self.start_time

            trajectory_data = {
                "timestamp": timestamp,
                "step": self.step_count,
            }

            # Add robot pose data
            if robot_pose:
                for key, value in robot_pose.items():
                    trajectory_data[f"current_{key}"] = float(value)

            # Add target pose data
            if target_pose:
                for key, value in target_pose.items():
                    trajectory_data[f"target_{key}"] = float(value)

            # Add error metrics
            if error_metrics:
                for key, value in error_metrics.items():
                    trajectory_data[f"error_{key}"] = float(value)

            # Write to CSV
            if not hasattr(self, "_trajectory_headers"):
                self._trajectory_headers = list(trajectory_data.keys())

            writer = self._get_csv_writer(f"{self.robot_name}_trajectory.csv", self._trajectory_headers)

            # Ensure all headers have values
            row_data = [trajectory_data.get(header, "") for header in self._trajectory_headers]
            writer.writerow(row_data)
            self.csv_files[f"{self.robot_name}_trajectory.csv"].flush()

        except Exception as e:
            logger.error(f"Error logging trajectory: {e}")

    def _print_timing_breakdown(self, robot, inference_time: float):
        """
        Print detailed timing breakdown to terminal (not saved to CSV).
        Shows all variable components that contribute to control loop frequency modulation.
        """
        try:
            print(f"📊 TIMING BREAKDOWN:")
            
            # 1. Policy Inference Time
            print(f"   🧠 Policy Inference: {inference_time * 1000:.1f}ms")
            
            # 2. Robot State Reading (from robot logs if available)
            if hasattr(robot, 'logs'):
                # Motor state reading timing
                if "read_follower_arm_pos_dt_s" in robot.logs:
                    print(f"   🔧 Motor Read: {robot.logs['read_follower_arm_pos_dt_s'] * 1000:.1f}ms")
                
                # Legacy format for other robots
                for name in getattr(robot, 'follower_arms', []):
                    key = f"read_follower_{name}_pos_dt_s"
                    if key in robot.logs:
                        print(f"   🔧 Motor Read ({name}): {robot.logs[key] * 1000:.1f}ms")
                
                # Camera reading timing
                for name in robot.cameras.keys():
                    key = f"read_camera_{name}_dt_s"
                    if key in robot.logs:
                        print(f"   📷 Camera Read ({name}): {robot.logs[key] * 1000:.1f}ms")
                    
                    # Also check async read timing
                    async_key = f"async_read_camera_{name}_dt_s"
                    if async_key in robot.logs:
                        print(f"   📷 Camera Async ({name}): {robot.logs[async_key] * 1000:.1f}ms")
                
                # Motor writing timing
                if "write_follower_arm_goal_pos_dt_s" in robot.logs:
                    print(f"   ✍️  Motor Write: {robot.logs['write_follower_arm_goal_pos_dt_s'] * 1000:.1f}ms")
                    
                # Legacy format for other robots
                for name in getattr(robot, 'follower_arms', []):
                    key = f"write_follower_{name}_goal_pos_dt_s"
                    if key in robot.logs:
                        print(f"   ✍️  Motor Write ({name}): {robot.logs[key] * 1000:.1f}ms")
            
            # 3. Calculate total processing time vs target
            if hasattr(self, '_current_control_loop_freq') and self._current_control_loop_freq > 0:
                actual_loop_time = 1.0 / self._current_control_loop_freq * 1000  # ms
                target_loop_time = 1000.0 / self.target_fps  # ms based on target FPS
                overhead = actual_loop_time - (inference_time * 1000)
                
                print(f"   ⚙️  Other Overhead: {overhead:.1f}ms")
                print(f"   🎯 Target Loop Time: {target_loop_time:.1f}ms ({self.target_fps:.0f} Hz)")
                print(f"   📊 Actual Loop Time: {actual_loop_time:.1f}ms")
                
                if actual_loop_time > target_loop_time:
                    deficit = actual_loop_time - target_loop_time
                    print(f"   ⚠️  Time Deficit: +{deficit:.1f}ms (why frequency < {self.target_fps:.0f} Hz)")
                else:
                    surplus = target_loop_time - actual_loop_time
                    print(f"   ✅ Time Surplus: -{surplus:.1f}ms (could run faster)")
            
            # 4. Explain frequency variation causes
            print(f"   🔄 Frequency Variation Causes:")
            print(f"      • USB bandwidth fluctuation (cameras)")
            print(f"      • Serial bus timing (motor communication)")  
            print(f"      • Neural network execution variance")
            print(f"      • Python GIL and OS scheduling")
            print(f"      • System load and background processes")
            
        except Exception as e:
            logger.debug(f"Error in timing breakdown: {e}")

    def log_step_summary(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        robot,
        policy_output: torch.Tensor | None = None,
        inference_time: float = 0.0,
        task: str | None = None,
    ):
        """
        Log a complete inference step with all relevant data.

        Args:
            observation: Robot observation
            action: Action sent to robot
            robot: Robot instance
            policy_output: Raw policy output (optional)
            inference_time: Inference timing
            task: Task description
        """
        # Calculate control loop frequency BEFORE incrementing step count
        current_time = time.perf_counter()
        if self.step_count > 0:  # Skip first step (no previous step to compare)
            step_interval = current_time - self.last_step_time
            control_loop_freq = 1.0 / step_interval if step_interval > 0 else 0.0
        else:
            control_loop_freq = 0.0  # First step, no frequency yet
        
        # Store for console output
        self._current_control_loop_freq = control_loop_freq
        
        # Update timing for next calculation
        self.last_step_time = current_time
        self.step_count += 1

        # Log all components
        self.log_robot_state(robot, observation, control_loop_freq)
        self.log_policy_inference(observation, action, policy_output, inference_time, task, control_loop_freq)

        # Print summary to console
        print(f"\n📊 INFERENCE STEP {self.step_count} @ {time.perf_counter() - self.start_time:.2f}s")
        print("=" * 60)

        # Print servo positions
        print("🔧 SERVO POSITIONS:")
        motor_positions = {k: v for k, v in observation.items() if ".pos" in k and "image" not in k}
        for motor, pos in motor_positions.items():
            value = float(pos) if isinstance(pos, (int, float, np.ndarray, torch.Tensor)) else pos
            print(f"   {motor:15}: {value:8.2f}")

        # Print policy outputs
        print("\n🎯 POLICY OUTPUT:")
        for key, value in action.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                val = float(value.item() if hasattr(value, "item") else value)
            else:
                val = float(value) if isinstance(value, (int, float)) else value
            print(f"   {key:15}: {val:8.2f}")

        # Print timing info including control loop frequency
        print(f"\n⏱️  TIMING: Inference took {inference_time * 1000:.1f}ms")
        if hasattr(self, '_current_control_loop_freq') and self._current_control_loop_freq > 0:
            print(f"🔄 CONTROL LOOP: {self._current_control_loop_freq:.1f} Hz")
            
        # Print detailed timing breakdown for terminal (not CSV)
        self._print_timing_breakdown(robot, inference_time)

        if task:
            print(f"📋 TASK: {task}")

        print("=" * 60)

    def close(self):
        """Close all CSV files and cleanup."""
        for file in self.csv_files.values():
            file.close()
        self.csv_files.clear()
        self.csv_writers.clear()

        logger.info(f"InferenceLogger closed. Data saved to: {self.output_dir}")
        print(f"\n📁 Inference logs saved to: {self.output_dir}")
        print("   📊 Files created:")
        for csv_file in self.output_dir.glob("*.csv"):
            print(f"      - {csv_file.name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
