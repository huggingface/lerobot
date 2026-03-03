"""
Robot Controller Module

Main interface for controlling RealMan robotic arms.
"""

import logging
from typing import Optional, Dict, List, Tuple
from Robotic_Arm.rm_robot_interface import *

logger = logging.getLogger(__name__)


class RobotController:
    """
    Main controller class for RealMan robotic arms.
    
    Provides high-level interface to robot control, state monitoring,
    and safety features.
    """
    
    # Robot model configurations
    ROBOT_MODELS = {
        "RM65": {
            "dof": 6,
            "model_enum": rm_robot_arm_model_e.RM_MODEL_RM_65_E,
            "force_type": rm_force_type_e.RM_MODEL_RM_B_E,
            "default_joints": [0, 20, 70, 0, 90, 0],
        },
        "RM75": {
            "dof": 7,
            "model_enum": rm_robot_arm_model_e.RM_MODEL_RM_75_E,
            "force_type": rm_force_type_e.RM_MODEL_RM_B_E,
            "default_joints": [0, 20, 0, 70, 0, 90, 0],
        },
        "RML63": {
            "dof": 6,
            "model_enum": rm_robot_arm_model_e.RM_MODEL_RM_63_II_E,
            "force_type": rm_force_type_e.RM_MODEL_RM_B_E,
            "default_joints": [0, 20, 70, 0, 90, 0],
        },
        "ECO65": {
            "dof": 6,
            "model_enum": rm_robot_arm_model_e.RM_MODEL_ECO_65_E,
            "force_type": rm_force_type_e.RM_MODEL_RM_B_E,
            "default_joints": [0, 20, 70, 0, -90, 0],
        },
        "GEN72": {
            "dof": 7,
            "model_enum": rm_robot_arm_model_e.RM_MODEL_GEN_72_E,
            "force_type": rm_force_type_e.RM_MODEL_RM_B_E,
            "default_joints": [0, 20, 0, 70, 0, 90, 0],
        },
        "R1D2": {
            "dof": 6,
            "model_enum": rm_robot_arm_model_e.RM_MODEL_RM_65_E,  # Using RM65 as base
            "force_type": rm_force_type_e.RM_MODEL_RM_B_E,
            "default_joints": [0, 20, 70, 0, 90, 0],
        },
    }
    
    def __init__(
        self,
        ip: str = "192.168.10.18",
        port: int = 8080,
        model: str = "RM65",
        dof: Optional[int] = None,
        thread_mode: rm_thread_mode_e = rm_thread_mode_e.RM_TRIPLE_MODE_E
    ):
        """
        Initialize robot controller.
        
        Args:
            ip: Robot IP address
            port: Robot port (default 8080)
            model: Robot model name (RM65, RM75, RML63, ECO65, GEN72, R1D2)
            dof: Degrees of freedom (6 or 7). If None, uses model default and auto-detects on connect.
            thread_mode: Threading mode for API
        """
        self.ip = ip
        self.port = port
        self.model_name = model.upper()
        
        if self.model_name not in self.ROBOT_MODELS:
            raise ValueError(f"Unknown robot model: {model}. "
                           f"Supported models: {list(self.ROBOT_MODELS.keys())}")
        
        self.model_config = self.ROBOT_MODELS[self.model_name]
        
        # Use provided DOF or fall back to model default
        if dof is not None:
            self.dof = dof
            self.model_config["dof"] = dof
            logger.info(f"Using configured DOF: {dof}")
        else:
            self.dof = self.model_config["dof"]
        
        # Initialize robot API
        self.robot = RoboticArm(thread_mode)
        self.handle = None
        self.connected = False
        
        logger.info(f"Initialized controller for {self.model_name} robot")
    
    def connect(self) -> bool:
        """
        Connect to the robot.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.handle = self.robot.rm_create_robot_arm(self.ip, self.port)
            
            if self.handle.id == 0:
                logger.error("Failed to connect: Invalid handle ID")
                return False
            
            self.connected = True
            logger.info(f"Connected to robot at {self.ip}:{self.port} (ID: {self.handle.id})")
            
            # Auto-detect actual DOF from robot
            self._detect_dof()
            
            # Get and log robot info
            self._log_robot_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the robot.
        
        Returns:
            True if disconnection successful
        """
        if not self.connected:
            return True
        
        try:
            result = self.robot.rm_delete_robot_arm()
            self.connected = False
            logger.info("Disconnected from robot")
            return result == 0
            
        except Exception as e:
            logger.error(f"Disconnection failed: {e}")
            return False
    
    def _detect_dof(self):
        """Detect actual DOF from robot by reading joint state."""
        try:
            result, state = self.robot.rm_get_current_arm_state()
            if result == 0:
                joints = state.get('joint', [])
                if joints and len(joints) > 0:
                    actual_dof = len(joints)
                    if actual_dof != self.dof:
                        logger.warning(f"DOF mismatch! Config says {self.dof}, robot has {actual_dof}")
                        logger.info(f"Auto-updating DOF to {actual_dof}")
                        self.dof = actual_dof
                        self.model_config["dof"] = actual_dof
                    else:
                        logger.info(f"Detected DOF: {self.dof}")
        except Exception as e:
            logger.warning(f"Could not auto-detect DOF: {e}")
    
    def _log_robot_info(self):
        """Log robot software information."""
        try:
            result, info = self.robot.rm_get_arm_software_info()
            if result == 0:
                logger.info("=" * 60)
                logger.info("Robot Software Information")
                logger.info("=" * 60)
                logger.info(f"Model: {info.get('product_version', 'Unknown')}")
                logger.info(f"Algorithm Version: {info.get('algorithm_info', {}).get('version', 'Unknown')}")
                logger.info(f"Control Layer: {info.get('ctrl_info', {}).get('version', 'Unknown')}")
                logger.info(f"Dynamics Version: {info.get('dynamic_info', {}).get('model_version', 'Unknown')}")
                logger.info(f"Planning Layer: {info.get('plan_info', {}).get('version', 'Unknown')}")
                logger.info("=" * 60)
        except Exception as e:
            logger.warning(f"Could not retrieve robot info: {e}")
    
    # ========== Motion Control Methods ==========
    
    def movej(
        self,
        joint_angles: List[float],
        velocity: int = 20,
        block: bool = True
    ) -> int:
        """
        Move robot to joint angles.
        
        Args:
            joint_angles: Target joint angles in degrees
            velocity: Movement velocity (1-100)
            block: Whether to block until motion complete
            
        Returns:
            Status code (0 = success)
        """
        if not self.connected:
            logger.error("Not connected to robot")
            return -1
        
        if len(joint_angles) != self.dof:
            logger.error(f"Expected {self.dof} joint angles, got {len(joint_angles)}")
            return -1
        
        return self.robot.rm_movej(joint_angles, velocity, 0, 0, int(block))
    
    def movej_p(
        self,
        pose: List[float],
        velocity: int = 20,
        block: bool = True
    ) -> int:
        """
        Move to Cartesian pose using joint interpolation.
        
        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in meters and radians
            velocity: Movement velocity (1-100)
            block: Whether to block until motion complete
            
        Returns:
            Status code (0 = success)
        """
        if not self.connected:
            logger.error("Not connected to robot")
            return -1
        
        return self.robot.rm_movej_p(pose, velocity, 0, 0, int(block))
    
    def movel(
        self,
        pose: List[float],
        velocity: int = 20,
        block: bool = True
    ) -> int:
        """
        Move to Cartesian pose using linear interpolation.
        
        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in meters and radians
            velocity: Movement velocity (1-100)
            block: Whether to block until motion complete
            
        Returns:
            Status code (0 = success)
        """
        if not self.connected:
            logger.error("Not connected to robot")
            return -1
        
        return self.robot.rm_movel(pose, velocity, 0, 0, int(block))
    
    def movej_canfd(
        self,
        joint_angles: List[float],
        follow: bool = False,
        expand: float = 0.0
    ) -> int:
        """
        High-frequency joint movement command via CANFD.
        
        This is faster than movej() for real-time teleoperation as it doesn't
        wait for motion completion and uses CANFD bus for faster communication.
        
        Args:
            joint_angles: Target joint angles in degrees
            follow: If True, wait for previous motion to complete before starting.
                   If False, immediately override current motion (better for teleop).
            expand: Position of the expand joint (if applicable)
            
        Returns:
            Status code (0 = success)
        """
        if not self.connected:
            logger.error("Not connected to robot")
            return -1
        
        if len(joint_angles) != self.dof:
            logger.error(f"Expected {self.dof} joint angles, got {len(joint_angles)}")
            return -1
        
        return self.robot.rm_movej_canfd(joint_angles, follow, expand)
    
    def stop(self) -> int:
        """
        Emergency stop - immediately halt all motion.
        
        Returns:
            Status code (0 = success)
        """
        if not self.connected:
            return -1
        
        logger.warning("Emergency stop triggered!")
        return self.robot.rm_set_arm_stop()
    
    # ========== State Query Methods ==========
    
    def get_current_joint_angles(self) -> Optional[List[float]]:
        """
        Get current joint angles.
        
        Returns:
            List of joint angles in degrees, or None if error
        """
        if not self.connected:
            return None
        
        result, state = self.robot.rm_get_current_arm_state()
        if result == 0:
            return state.get('joint', [])
        return None
    
    def get_current_pose(self) -> Optional[List[float]]:
        """
        Get current end-effector pose.
        
        Returns:
            Pose [x, y, z, rx, ry, rz] or None if error
        """
        if not self.connected:
            return None
        
        result, state = self.robot.rm_get_current_arm_state()
        if result == 0:
            pose_data = state.get('pose', [])
            
            # Handle both list format and nested dict format
            if isinstance(pose_data, list) and len(pose_data) == 6:
                return pose_data
            elif isinstance(pose_data, dict):
                return [
                    pose_data.get('position', {}).get('x', 0),
                    pose_data.get('position', {}).get('y', 0),
                    pose_data.get('position', {}).get('z', 0),
                    pose_data.get('euler', {}).get('rx', 0),
                    pose_data.get('euler', {}).get('ry', 0),
                    pose_data.get('euler', {}).get('rz', 0),
                ]
        return None
    
    def get_joint_velocities(self) -> Optional[List[float]]:
        """
        Get current joint velocities.
        
        Returns:
            List of joint velocities in deg/s, or None if error
        """
        if not self.connected:
            return None
        
        result, state = self.robot.rm_get_current_arm_state()
        if result == 0:
            return state.get('joint_speed', [])
        return None
    
    def is_moving(self) -> bool:
        """
        Check if robot is currently moving.
        
        Returns:
            True if moving, False otherwise
        """
        if not self.connected:
            return False
        
        result, state = self.robot.rm_get_current_arm_state()
        if result == 0:
            return state.get('arm_err', 0) == 0 and state.get('sys_err', 0) == 0
        return False
    
    # ========== Safety Methods ==========
    
    def set_collision_level(self, level: int) -> int:
        """
        Set collision detection sensitivity.
        
        Args:
            level: Collision level (0-8, higher = more sensitive)
            
        Returns:
            Status code (0 = success)
        """
        if not self.connected:
            return -1
        
        if not 0 <= level <= 8:
            logger.error("Collision level must be 0-8")
            return -1
        
        # Try different API methods depending on SDK version
        try:
            if hasattr(self.robot, 'rm_set_collision_stage'):
                return self.robot.rm_set_collision_stage(level)
            elif hasattr(self.robot, 'rm_set_collision_state'):
                return self.robot.rm_set_collision_state(level)
            else:
                logger.warning("Collision level API not available in this SDK version")
                return -1
        except Exception as e:
            logger.warning(f"Failed to set collision level: {e}")
            return -1
    
    def clear_errors(self) -> int:
        """
        Clear robot error state.
        
        Returns:
            Status code (0 = success)
        """
        if not self.connected:
            return -1
        
        return self.robot.rm_clear_system_err()
    
    # ========== Gripper Control Methods ==========
    
    def gripper_open(self, speed: int = 500, block: bool = True, timeout: int = 3000) -> int:
        """
        Open the gripper fully.
        
        Args:
            speed: Opening speed (1-1000, higher = faster)
            block: Wait for completion
            timeout: Timeout in milliseconds
            
        Returns:
            Status code (0 = success)
        """
        if not self.connected:
            logger.error("Robot not connected")
            return -1
        
        logger.info(f"Opening gripper (speed={speed})")
        return self.robot.rm_set_gripper_release(speed, block, timeout)
    
    def gripper_close(self, speed: int = 500, force: int = 500, block: bool = True, timeout: int = 3000) -> int:
        """
        Close the gripper with force control.
        
        Args:
            speed: Closing speed (1-1000)
            force: Gripping force threshold (1-1000)
            block: Wait for completion
            timeout: Timeout in milliseconds
            
        Returns:
            Status code (0 = success)
        """
        if not self.connected:
            logger.error("Robot not connected")
            return -1
        
        logger.info(f"Closing gripper (speed={speed}, force={force})")
        return self.robot.rm_set_gripper_pick(speed, force, block, timeout)
    
    def gripper_set_position(self, position: int, block: bool = True, timeout: int = 3000) -> int:
        """
        Set gripper to a specific position.
        
        Args:
            position: Gripper opening (1-1000, 1=closed, 1000=open)
            block: Wait for completion
            timeout: Timeout in milliseconds
            
        Returns:
            Status code (0 = success)
        """
        if not self.connected:
            logger.error("Robot not connected")
            return -1
        
        logger.debug(f"Setting gripper position to {position}")
        return self.robot.rm_set_gripper_position(position, block, timeout)
    
    def gripper_get_state(self) -> Optional[Dict]:
        """
        Get current gripper state.
        
        Returns:
            Dictionary with gripper state or None if error
        """
        if not self.connected:
            return None
        
        result, state = self.robot.rm_get_gripper_state()
        if result == 0:
            return state
        return None
    
    # ========== Utility Methods ==========
    
    def move_to_home(self, velocity: int = 20) -> int:
        """
        Move robot to home position.
        
        Args:
            velocity: Movement velocity (1-100)
            
        Returns:
            Status code (0 = success)
        """
        home_position = self.model_config["default_joints"]
        logger.info(f"Moving to home position: {home_position}")
        return self.movej(home_position, velocity, block=True)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
