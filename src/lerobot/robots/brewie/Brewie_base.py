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
import time
import threading
import base64
from functools import cached_property
from typing import Any
import numpy as np
import cv2
from io import BytesIO

import roslibpy
from roslibpy import Ros, Service, ServiceRequest, Topic, Message

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_Brewie import BrewieConfig

logger = logging.getLogger(__name__)


class JoystickSubscriber:
    """
    Class for subscribing to ROS topic /joy and receiving joystick data.
    """
    
    def __init__(self, ros_client, joy_topic):
        """
        Initialize joystick subscriber.
        
        Args:
            ros_client: ROS client for connection
            joy_topic: ROS topic with joystick data
        """
        self.last_joy_data = None
        self.client = ros_client
        self.joy_topic = joy_topic
        self.joy_lock = threading.Lock()
        self.last_message = None
        
    def on_joy_received(self, message):
        """
        Callback that is called when a new joystick message is received.
        
        Args:
            message: ROS message with joystick data
        """
        try:
            with self.joy_lock:
                self.last_message = message
                # Extract joystick data
                joy_data = self._extract_joy_data(message)
                if joy_data is not None:
                    self.last_joy_data = joy_data
                else:
                    logger.warning("[JoystickSubscriber] Failed to extract joystick data from message")
        except Exception as e:
            logger.error(f"[JoystickSubscriber] Error in on_joy_received: {e}")
    
    def _extract_joy_data(self, message):
        """
        Extract joystick data from ROS message.
        
        Args:
            message: ROS message with joystick data
            
        Returns:
            dict or None: Joystick data or None on error
        """
        try:
            if message is None:
                logger.warning("[JoystickSubscriber] Message is None")
                return None
                
            # Extract axes and buttons
            axes = message.get('axes', [])
            buttons = message.get('buttons', [])
            
            if not axes and not buttons:
                logger.warning("[JoystickSubscriber] No axes or buttons data in message")
                return None
            
            # Create structured data
            joy_data = {
                'axes': axes,
                'buttons': buttons,
                'timestamp': time.time()
            }
            
            return joy_data
            
        except Exception as e:
            logger.error(f"[JoystickSubscriber] Error extracting joystick data: {e}")
            return None
    
    def get_last_joy_data(self):
        """
        Method that returns the latest joystick data.
        
        Returns:
            dict or None: Latest joystick data or None if no data available
        """
        with self.joy_lock:
            if self.last_joy_data is None:
                logger.debug("[JoystickSubscriber] No joystick data received")
                return None
            return self.last_joy_data.copy()
    
    def get_last_message(self):
        """
        Returns the last received message.
        
        Returns:
            dict or None: Last ROS message or None
        """
        with self.joy_lock:
            return self.last_message
    
    def subscribe(self):
        """Subscribe to joystick data topic."""
        try:
            self.joy_topic.subscribe(self.on_joy_received)
            logger.info("[JoystickSubscriber] Successfully subscribed to joystick topic")
        except Exception as e:
            logger.error(f"[JoystickSubscriber] Failed to subscribe to joystick topic: {e}")
            raise


class IMUSubscriber:
    """
    Class for subscribing to ROS topic /imu and receiving IMU data.
    """
    
    def __init__(self, ros_client, imu_topic):
        """
        Initialize IMU subscriber.
        
        Args:
            ros_client: ROS client for connection
            imu_topic: ROS topic with IMU data
        """
        self.last_imu_data = None
        self.client = ros_client
        self.imu_topic = imu_topic
        self.imu_lock = threading.Lock()
        self.last_message = None
        
    def on_imu_received(self, message):
        """
        Callback that is called when a new IMU message is received.
        
        Args:
            message: ROS message with IMU data
        """
        try:
            with self.imu_lock:
                self.last_message = message
                # Extract IMU data
                imu_data = self._extract_imu_data(message)
                if imu_data is not None:
                    self.last_imu_data = imu_data
                else:
                    logger.warning("[IMUSubscriber] Failed to extract IMU data from message")
        except Exception as e:
            logger.error(f"[IMUSubscriber] Error in on_imu_received: {e}")
    
    def _extract_imu_data(self, message):
        """
        Extract IMU data from ROS message.
        
        Args:
            message: ROS message with IMU data
            
        Returns:
            dict or None: IMU data or None on error
        """
        try:
            if message is None:
                logger.warning("[IMUSubscriber] Message is None")
                return None
                
            # Extract orientation
            orientation = message.get('orientation', {})
            orientation_data = {
                'x': orientation.get('x', 0.0),
                'y': orientation.get('y', 0.0),
                'z': orientation.get('z', 0.0),
                'w': orientation.get('w', 0.0)
            }
            
            # Extract angular velocity
            angular_velocity = message.get('angular_velocity', {})
            angular_velocity_data = {
                'x': angular_velocity.get('x', 0.0),
                'y': angular_velocity.get('y', 0.0),
                'z': angular_velocity.get('z', 0.0)
            }
            
            # Extract linear acceleration
            linear_acceleration = message.get('linear_acceleration', {})
            linear_acceleration_data = {
                'x': linear_acceleration.get('x', 0.0),
                'y': linear_acceleration.get('y', 0.0),
                'z': linear_acceleration.get('z', 0.0)
            }
            
            # Create structured data
            imu_data = {
                'orientation': orientation_data,
                'angular_velocity': angular_velocity_data,
                'linear_acceleration': linear_acceleration_data,
                'timestamp': time.time()
            }
            
            return imu_data
            
        except Exception as e:
            logger.error(f"[IMUSubscriber] Error extracting IMU data: {e}")
            return None
    
    def get_last_imu_data(self):
        """
        Method that returns the latest IMU data.
        
        Returns:
            dict or None: Latest IMU data or None if no data available
        """
        with self.imu_lock:
            if self.last_imu_data is None:
                logger.debug("[IMUSubscriber] No IMU data received")
                return None
            return self.last_imu_data.copy()
    
    def get_last_message(self):
        """
        Returns the last received message.
        
        Returns:
            dict or None: Last ROS message or None
        """
        with self.imu_lock:
            return self.last_message
    
    def subscribe(self):
        """Subscribe to IMU data topic."""
        try:
            self.imu_topic.subscribe(self.on_imu_received)
            logger.info("[IMUSubscriber] Successfully subscribed to IMU topic")
        except Exception as e:
            logger.error(f"[IMUSubscriber] Failed to subscribe to IMU topic: {e}")
            raise


class CameraSubscriber:
    """
    Class for subscribing to ROS topic with images and receiving the latest snapshot.
    Based on user example for reliable image processing.
    """
    
    def __init__(self, ros_client, image_topic):
        """
        Initialize image subscriber.
        
        Args:
            ros_client: ROS client for connection
            image_topic: ROS topic with images
        """
        self.last_image = None
        self.client = ros_client
        self.image_topic = image_topic
        self.image_lock = threading.Lock()
        self.last_message = None
        
    def on_image_received(self, message):
        """
        Callback that is called when a new message is received.
        
        Args:
            message: ROS message with image
        """
        try:
            with self.image_lock:
                self.last_message = message
                # Decode image immediately upon receipt
                decoded_image = self._decode_image_from_message(message)
                if decoded_image is not None:
                    self.last_image = decoded_image
                else:
                    logger.warning("[CameraSubscriber] Failed to decode image from message")
        except Exception as e:
            logger.error(f"[CameraSubscriber] Error in on_image_received: {e}")
    
    def _decode_image_from_message(self, message):
        """
        Decode image from ROS message.
        
        Args:
            message: ROS message with image
            
        Returns:
            np.ndarray or None: Decoded image or None on error
        """
        try:
            if message is None:
                logger.warning("[CameraSubscriber] Message is None")
                return None
                
            # Get image data
            img_data = message.get('data')
            if img_data is None:
                logger.warning("[CameraSubscriber] No 'data' field in message")
                return None
            
            # Handle different data formats
            if isinstance(img_data, str):
                # If data is a string, try to decode as Base64
                try:
                    image_bytes = base64.b64decode(img_data)
                except Exception as e:
                    logger.warning(f"[CameraSubscriber] Failed to decode Base64 string: {e}")
                    # If not Base64, try as regular string
                    image_bytes = img_data.encode('latin-1')
            else:
                # If data is already in bytes format
                image_bytes = img_data
            
            # Convert byte array to NumPy array
            img_np = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image from JPEG/PNG using OpenCV
            img_cv = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
            
            if img_cv is None:
                logger.warning("[CameraSubscriber] Failed to decode image with OpenCV")
                return None
                
            return img_cv
            
        except Exception as e:
            logger.error(f"[CameraSubscriber] Error decoding image: {e}")
            return None
    
    def get_last_image(self):
        """
        Method that returns the last saved snapshot.
        
        Returns:
            np.ndarray or None: Last image or None if no data available
        """
        with self.image_lock:
            if self.last_image is None:
                logger.debug("[CameraSubscriber] No image data received")
                return None
            return self.last_image.copy()
    
    def get_last_message(self):
        """
        Returns the last received message.
        
        Returns:
            dict or None: Last ROS message or None
        """
        with self.image_lock:
            return self.last_message
    
    def subscribe(self):
        """Subscribe to image topic."""
        try:
            self.image_topic.subscribe(self.on_image_received)
            logger.info("[CameraSubscriber] Successfully subscribed to image topic")
        except Exception as e:
            logger.error(f"[CameraSubscriber] Failed to subscribe to image topic: {e}")
            raise


class BrewieBase(Robot):

    config_class = BrewieConfig
    name = "BrewieBase"

    def __init__(self, config: BrewieConfig):
        super().__init__(config)
        self.config = config
        
        # ROS connection
        self.ros_client = None
        self.position_service = None
        self.set_position_topic = None
        self.camera_topic = None
        self.joy_topic = None
        self.imu_topic = None
        
        # Camera data
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.camera_subscriber = None
        
        # Additional sensor data
        self.joystick_subscriber = None
        self.imu_subscriber = None
        
        # Servo positions cache
        self.current_positions = {}
        self.position_lock = threading.Lock()
        
        # Create reverse mapping (joint_name -> servo_id)
        self.joint_to_id = {v: k for k, v in config.servo_mapping.items()}
        
        # Initialize cameras if configured
        self.cameras = make_cameras_from_configs(config.cameras) if config.cameras else {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Return features for all joints defined in servo_mapping
        return {f"{joint}.pos": float for joint in self.config.servo_mapping.values()}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        if self.cameras:
            return {
                cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
            }
        else:
            # Default camera dimensions for ROS camera
            return {"camera": (480, 640, 3)}

    @property
    def _joystick_ft(self) -> dict[str, type]:
        """Joystick features: individual axes and buttons."""
        # Create separate features for each axis and button
        features = {}
        # 8 joystick axes
        for i in range(8):
            features[f"joystick.axis_{i}"] = float
        # 15 joystick buttons (also float for LeRobot compatibility)
        for i in range(15):
            features[f"joystick.button_{i}"] = float
        return features

    @property
    def _imu_ft(self) -> dict[str, type]:
        """IMU features: orientation, angular velocity, linear acceleration."""
        return {
            "imu.orientation.x": float,
            "imu.orientation.y": float,
            "imu.orientation.z": float,
            "imu.orientation.w": float,
            "imu.angular_velocity.x": float,
            "imu.angular_velocity.y": float,
            "imu.angular_velocity.z": float,
            "imu.linear_acceleration.x": float,
            "imu.linear_acceleration.y": float,
            "imu.linear_acceleration.z": float,
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft, **self._joystick_ft, **self._imu_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        ros_connected = self.ros_client is not None and self.ros_client.is_connected
        cameras_connected = all(cam.is_connected for cam in self.cameras.values()) if self.cameras else True
        return ros_connected and cameras_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to ROS and initialize services and topics.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to ROS
        self.ros_client = Ros(host=self.config.master_ip, port=self.config.master_port)
        self.ros_client.run()
        
        # Wait for connection
        timeout = 10
        start_time = time.time()
        while not self.ros_client.is_connected and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not self.ros_client.is_connected:
            raise ConnectionError(f"Failed to connect to ROS at {self.config.master_ip}:{self.config.master_port}")
        
        # Initialize services and topics
        self._setup_ros_services()
        self._setup_ros_topics()
        
        # Initialize default positions for all servos
        self._initialize_default_positions()
        
        # Connect cameras if configured
        for cam in self.cameras.values():
            cam.connect()
        
        logger.info(f"{self} connected to ROS at {self.config.master_ip}:{self.config.master_port}")

    def _setup_ros_services(self):
        """Setup ROS services for servo control."""
        self.position_service = Service(
            self.ros_client, 
            self.config.position_service, 
            'ros_robot_controller/GetBusServosPosition'
        )

    def _setup_ros_topics(self):
        """Setup ROS topics for servo control, camera, joystick and IMU."""
        self.set_position_topic = Topic(
            self.ros_client,
            self.config.set_position_topic,
            'ros_robot_controller/SetBusServosPosition'
        )
        
        # Setup camera topic if not using standard cameras
        if not self.cameras:
            self.camera_topic = Topic(
                self.ros_client,
                self.config.camera_topic,
                'sensor_msgs/CompressedImage'
            )
            # Create CameraSubscriber for reliable image processing
            self.camera_subscriber = CameraSubscriber(self.ros_client, self.camera_topic)
            self.camera_subscriber.subscribe()
        
        # Setup joystick topic
        self.joy_topic = Topic(
            self.ros_client,
            self.config.joy_topic,
            'sensor_msgs/Joy'
        )
        self.joystick_subscriber = JoystickSubscriber(self.ros_client, self.joy_topic)
        self.joystick_subscriber.subscribe()
        
        # Setup IMU topic
        self.imu_topic = Topic(
            self.ros_client,
            self.config.imu_topic,
            'sensor_msgs/Imu'
        )
        self.imu_subscriber = IMUSubscriber(self.ros_client, self.imu_topic)
        self.imu_subscriber.subscribe()

    def _initialize_default_positions(self):
        """Initialize default positions for all servos."""
        with self.position_lock:
            for servo_id in self.config.servo_ids:
                joint_name = self.config.servo_mapping.get(servo_id)
                if joint_name:
                    joint_key = f"{joint_name}.pos"
                    if joint_key not in self.current_positions:
                        # Use middle position as default (500 out of 0-1000 range)
                        self.current_positions[joint_key] = 500.0
                        logger.debug(f"Initialized default position for {joint_name}: 500.0")


    @property
    def is_calibrated(self) -> bool:
        # For ROS-based control, we assume the robot is always "calibrated"
        # since calibration is handled by the ROS controller
        return True

    def calibrate(self) -> None:
        """
        For ROS-based control, calibration is handled by the ROS controller.
        This method is kept for compatibility but does nothing.
        """
        logger.info("Calibration is handled by the ROS controller. Skipping local calibration.")

    def configure(self) -> None:
        """
        For ROS-based control, configuration is handled by the ROS controller.
        This method is kept for compatibility but does nothing.
        """
        logger.info("Configuration is handled by the ROS controller. Skipping local configuration.")

    def setup_motors(self) -> None:
        """
        For ROS-based control, motor setup is handled by the ROS controller.
        This method is kept for compatibility but does nothing.
        """
        logger.info("Motor setup is handled by the ROS controller. Skipping local setup.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}
        total_start = time.perf_counter()
        
        # Read servo positions via ROS service
        start = time.perf_counter()
        try:
            request = ServiceRequest({'id': self.config.servo_ids})
            result = self.position_service.call(request)
            if result.get('success', False):
                positions = result.get('position', [])
                received_servo_ids = set()
                
                # Process received positions
                for pos_data in positions:
                    servo_id = pos_data['id']
                    position = pos_data['position']
                    joint_name = self.config.servo_mapping.get(servo_id)
                    if joint_name:
                        # Ensure position is float32 and within valid range (0-1000)
                        position = float(max(0.0, min(1000.0, float(position))))
                        obs_dict[f"{joint_name}.pos"] = position
                        received_servo_ids.add(servo_id)
                
                # Fill missing servo positions with previous values
                with self.position_lock:
                    for servo_id in self.config.servo_ids:
                        if servo_id not in received_servo_ids:
                            joint_name = self.config.servo_mapping.get(servo_id)
                            if joint_name:
                                joint_key = f"{joint_name}.pos"
                                if joint_key in self.current_positions:
                                    obs_dict[joint_key] = self.current_positions[joint_key]
                                    logger.debug(f"Using previous position for missing servo {servo_id} ({joint_name}): {self.current_positions[joint_key]}")
                                else:
                                    # If no previous value available, use default position (middle of range)
                                    default_position = 500.0  # Middle of 0-1000 range
                                    obs_dict[joint_key] = default_position
                                    logger.warning(f"No previous position for servo {servo_id} ({joint_name}), using default: {default_position}")
                    
                    # Update current positions cache
                    self.current_positions = obs_dict.copy()
            else:
                logger.warning("Failed to get servo positions from ROS service")
                # Use cached positions if available
                with self.position_lock:
                    obs_dict = self.current_positions.copy()
                
        except Exception as e:
            logger.error(f"Error reading servo positions: {e}")
            # Use cached positions if available
            with self.position_lock:
                obs_dict = self.current_positions.copy()
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read servo state: {dt_ms:.1f}ms")

        # Get camera data
        if self.cameras:
            # Use standard cameras
            for cam_key, cam in self.cameras.items():
                start = time.perf_counter()
                obs_dict[cam_key] = cam.async_read()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        else:
            # Use ROS camera with improved CameraSubscriber
            start = time.perf_counter()
            if self.camera_subscriber is not None:
                received_img = self.camera_subscriber.get_last_image()
                if received_img is None:
                    logger.warning("[Image] No data received from subscriber")
                    # Return empty image if no data available
                    obs_dict["camera"] = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    obs_dict["camera"] = received_img
                    logger.debug(f"[Image] Successfully received image with shape: {received_img.shape}")
            else:
                logger.warning("[Image] Camera subscriber not initialized")
                obs_dict["camera"] = np.zeros((480, 640, 3), dtype=np.uint8)
            
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read camera: {dt_ms:.1f}ms")

        # Get joystick data
        start = time.perf_counter()
        if self.joystick_subscriber is not None:
            joy_data = self.joystick_subscriber.get_last_joy_data()
            if joy_data is not None:
                # Convert joystick data to separate features
                axes = joy_data['axes']
                buttons = joy_data['buttons']
                
                # Add axis data (up to 8 axes)
                for i in range(8):
                    if i < len(axes):
                        obs_dict[f"joystick.axis_{i}"] = float(axes[i])
                    else:
                        obs_dict[f"joystick.axis_{i}"] = 0.0
                
                # Add button data (up to 15 buttons)
                for i in range(15):
                    if i < len(buttons):
                        obs_dict[f"joystick.button_{i}"] = float(buttons[i])
                    else:
                        obs_dict[f"joystick.button_{i}"] = 0.0
                
                logger.debug(f"[Joystick] Successfully received data: {len(axes)} axes, {len(buttons)} buttons")
            else:
                # Return zero values if no data available
                for i in range(8):
                    obs_dict[f"joystick.axis_{i}"] = 0.0
                for i in range(15):
                    obs_dict[f"joystick.button_{i}"] = 0.0
                logger.debug("[Joystick] No data received, using default values")
        else:
            logger.warning("[Joystick] Joystick subscriber not initialized")
            for i in range(8):
                obs_dict[f"joystick.axis_{i}"] = 0.0
            for i in range(15):
                obs_dict[f"joystick.button_{i}"] = 0.0
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read joystick: {dt_ms:.1f}ms")

        # Get IMU data
        start = time.perf_counter()
        if self.imu_subscriber is not None:
            imu_data = self.imu_subscriber.get_last_imu_data()
            if imu_data is not None:
                # Extract IMU data
                orientation = imu_data['orientation']
                angular_velocity = imu_data['angular_velocity']
                linear_acceleration = imu_data['linear_acceleration']
                
                obs_dict["imu.orientation.x"] = float(orientation['x'])
                obs_dict["imu.orientation.y"] = float(orientation['y'])
                obs_dict["imu.orientation.z"] = float(orientation['z'])
                obs_dict["imu.orientation.w"] = float(orientation['w'])
                
                obs_dict["imu.angular_velocity.x"] = float(angular_velocity['x'])
                obs_dict["imu.angular_velocity.y"] = float(angular_velocity['y'])
                obs_dict["imu.angular_velocity.z"] = float(angular_velocity['z'])
                
                obs_dict["imu.linear_acceleration.x"] = float(linear_acceleration['x'])
                obs_dict["imu.linear_acceleration.y"] = float(linear_acceleration['y'])
                obs_dict["imu.linear_acceleration.z"] = float(linear_acceleration['z'])
                
                logger.debug(f"[IMU] Successfully received data: orientation=({orientation['x']:.3f}, {orientation['y']:.3f}, {orientation['z']:.3f}, {orientation['w']:.3f})")
            else:
                # Return zero values if no data available
                obs_dict["imu.orientation.x"] = 0.0
                obs_dict["imu.orientation.y"] = 0.0
                obs_dict["imu.orientation.z"] = 0.0
                obs_dict["imu.orientation.w"] = 0.0
                obs_dict["imu.angular_velocity.x"] = 0.0
                obs_dict["imu.angular_velocity.y"] = 0.0
                obs_dict["imu.angular_velocity.z"] = 0.0
                obs_dict["imu.linear_acceleration.x"] = 0.0
                obs_dict["imu.linear_acceleration.y"] = 0.0
                obs_dict["imu.linear_acceleration.z"] = 0.0
                logger.debug("[IMU] No data received, using default values")
        else:
            logger.warning("[IMU] IMU subscriber not initialized")
            obs_dict["imu.orientation.x"] = 0.0
            obs_dict["imu.orientation.y"] = 0.0
            obs_dict["imu.orientation.z"] = 0.0
            obs_dict["imu.orientation.w"] = 0.0
            obs_dict["imu.angular_velocity.x"] = 0.0
            obs_dict["imu.angular_velocity.y"] = 0.0
            obs_dict["imu.angular_velocity.z"] = 0.0
            obs_dict["imu.linear_acceleration.x"] = 0.0
            obs_dict["imu.linear_acceleration.y"] = 0.0
            obs_dict["imu.linear_acceleration.z"] = 0.0
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read IMU: {dt_ms:.1f}ms")

        # Log total observation time
        total_dt_ms = (time.perf_counter() - total_start) * 1e3
        logger.info(f"{self} TOTAL observation time: {total_dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration via ROS.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the servos, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position.
        if self.config.max_relative_target is not None:
            with self.position_lock:
                present_pos = {key.removesuffix(".pos"): val for key, val in self.current_positions.items()}
            goal_present_pos = {key: (g_pos, present_pos.get(key, 0)) for key, g_pos in goal_pos.items()}
            # Convert max_relative_target to float if it's not None
            max_relative = float(self.config.max_relative_target)
            goal_pos = ensure_safe_goal_position(goal_present_pos, max_relative)

        # Convert joint positions to servo positions and send via ROS
        try:
            new_positions = []
            for joint_name, position in goal_pos.items():
                servo_id = self.joint_to_id.get(joint_name)
                if servo_id is not None:
                    # Ensure position is within valid range (0-1000)
                    position = float(max(0.0, min(1000.0, float(position))))
                    new_positions.append({'id': servo_id, 'position': position})
            
            if new_positions:
                servo_msg = Message({
                    'duration': self.config.servo_duration,
                    'position': new_positions,
                })
                self.set_position_topic.publish(servo_msg)
                
        except Exception as e:
            logger.error(f"Error sending servo positions: {e}")

        return {f"{joint}.pos": val for joint, val in goal_pos.items()}

    def test_camera_connection(self) -> dict[str, Any]:
        """
        Test camera connection and return status information.
        
        Returns:
            dict: Camera status information
        """
        result = {
            "camera_available": False,
            "last_image_shape": None,
            "last_message_received": False,
            "error_message": None
        }
        
        try:
            if self.camera_subscriber is not None:
                # Check last message
                last_msg = self.camera_subscriber.get_last_message()
                result["last_message_received"] = last_msg is not None
                
                # Check last image
                last_img = self.camera_subscriber.get_last_image()
                if last_img is not None:
                    result["camera_available"] = True
                    result["last_image_shape"] = last_img.shape
                    logger.info(f"[CameraTest] Camera working, image shape: {last_img.shape}")
                else:
                    result["error_message"] = "No image data received"
                    logger.warning("[CameraTest] No image data available")
            else:
                result["error_message"] = "Camera subscriber not initialized"
                logger.warning("[CameraTest] Camera subscriber not initialized")
                
        except Exception as e:
            result["error_message"] = str(e)
            logger.error(f"[CameraTest] Error testing camera: {e}")
            
        return result

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        # Disconnect ROS
        if self.ros_client and self.ros_client.is_connected:
            self.ros_client.close()
            self.ros_client = None

        logger.info(f"{self} disconnected from ROS.")

