"""
Mock Unitree G1 Dex3 Robot for simulation testing.

Provides a simulated G1 Dex3 robot that:
- Tracks joint states internally
- Updates state when actions are sent
- Works with existing visualization and recording scripts
- No external dependencies (no DDS, no hardware)
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

logger = logging.getLogger(__name__)


# Joint configuration (matches real robot)
ARM_JOINT_NAMES = [
    # Left arm
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint", 
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

LEFT_HAND_JOINT_NAMES = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]

RIGHT_HAND_JOINT_NAMES = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]

ALL_JOINT_NAMES = ARM_JOINT_NAMES + LEFT_HAND_JOINT_NAMES + RIGHT_HAND_JOINT_NAMES


@RobotConfig.register_subclass("mock_unitree_g1_dex3")
@dataclass
class MockUnitreeG1Dex3Config(RobotConfig):
    """Configuration for mock G1 Dex3 robot."""
    
    # Initial pose
    initial_pose: dict[str, float] = field(default_factory=dict)
    
    # Simulation parameters
    noise_std: float = 0.001  # Gaussian noise on observations
    action_smoothing: float = 0.5  # Blend factor for action application
    
    # Cameras (optional)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    
    # Control rate (Hz)
    control_rate: float = 100.0


class MockUnitreeG1Dex3(Robot):
    """
    Mock Unitree G1 Robot with Dex3 hands for simulation.
    
    Usage:
        from lerobot.robots.unitree_g1.mock_unitree_g1_dex3 import (
            MockUnitreeG1Dex3, MockUnitreeG1Dex3Config
        )
        
        robot = MockUnitreeG1Dex3(MockUnitreeG1Dex3Config())
        robot.connect()
        
        obs = robot.get_observation()
        robot.send_action({"left_shoulder_pitch_joint.q": 0.5, ...})
        
        robot.disconnect()
    """
    
    config_class = MockUnitreeG1Dex3Config
    name = "mock_unitree_g1_dex3"
    
    def __init__(self, config: MockUnitreeG1Dex3Config):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._is_calibrated = True
        
        # Joint state storage
        self._joint_positions: dict[str, float] = {}
        self._joint_velocities: dict[str, float] = {}
        self._joint_torques: dict[str, float] = {}
        
        # Initialize to neutral pose
        for name in ALL_JOINT_NAMES:
            self._joint_positions[name] = config.initial_pose.get(name, 0.0)
            self._joint_velocities[name] = 0.0
            self._joint_torques[name] = 0.0
        
        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # State lock for thread safety
        self._state_lock = threading.Lock()
        
        # Timestamp
        self._last_update_time = time.time()
        
        logger.info(f"MockUnitreeG1Dex3: Initialized with {len(ALL_JOINT_NAMES)} joints")
    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation space."""
        features = {}
        
        # Joint positions
        for name in ALL_JOINT_NAMES:
            features[f"{name}.q"] = float
        
        # Joint velocities (optional)
        for name in ALL_JOINT_NAMES:
            features[f"{name}.dq"] = float
        
        # Cameras
        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)
        
        return features
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action space (position commands)."""
        return {f"{name}.q": float for name in ALL_JOINT_NAMES}
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated
    
    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Connect to mock robot."""
        logger.info("MockUnitreeG1Dex3: Connecting...")
        
        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()
        
        self._is_connected = True
        self._last_update_time = time.time()
        
        if calibrate:
            self.calibrate()
        
        logger.info("MockUnitreeG1Dex3: Connected")
    
    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect from mock robot."""
        logger.info("MockUnitreeG1Dex3: Disconnecting...")
        
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        self._is_connected = False
        logger.info("MockUnitreeG1Dex3: Disconnected")
    
    @check_if_not_connected
    def calibrate(self) -> None:
        """Calibrate mock robot (no-op)."""
        self._is_calibrated = True
    
    def configure(self) -> None:
        """Configure mock robot (no-op)."""
        pass
    
    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Get current observation."""
        obs = {}
        
        with self._state_lock:
            # Joint positions with noise
            for name in ALL_JOINT_NAMES:
                noise = np.random.normal(0, self.config.noise_std)
                obs[f"{name}.q"] = float(self._joint_positions[name] + noise)
                obs[f"{name}.dq"] = float(self._joint_velocities[name])
        
        # Camera images
        for cam_name, cam in self.cameras.items():
            obs[cam_name] = cam.read()
        
        return obs
    
    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Apply action to mock robot."""
        current_time = time.time()
        dt = current_time - self._last_update_time
        self._last_update_time = current_time
        
        with self._state_lock:
            applied_action = {}
            
            for key, target in action.items():
                if key.endswith(".q"):
                    joint_name = key[:-2]  # Remove ".q"
                    
                    if joint_name in self._joint_positions:
                        current = self._joint_positions[joint_name]
                        
                        # Smooth interpolation toward target
                        alpha = self.config.action_smoothing
                        new_pos = current * (1 - alpha) + target * alpha
                        
                        # Update velocity estimate
                        if dt > 0:
                            self._joint_velocities[joint_name] = (new_pos - current) / dt
                        
                        self._joint_positions[joint_name] = new_pos
                        applied_action[key] = new_pos
            
            return applied_action
    
    def reset_to_pose(self, pose: dict[str, float] | None = None) -> None:
        """Reset robot to a specific pose."""
        with self._state_lock:
            if pose is None:
                pose = self.config.initial_pose
            
            for name in ALL_JOINT_NAMES:
                self._joint_positions[name] = pose.get(name, 0.0)
                self._joint_velocities[name] = 0.0
                self._joint_torques[name] = 0.0
        
        logger.info("MockUnitreeG1Dex3: Reset to pose")
    
    def set_joint_position(self, name: str, position: float) -> None:
        """Directly set a joint position (for testing)."""
        with self._state_lock:
            if name in self._joint_positions:
                self._joint_positions[name] = position
    
    def get_joint_positions(self) -> dict[str, float]:
        """Get all joint positions."""
        with self._state_lock:
            return dict(self._joint_positions)
