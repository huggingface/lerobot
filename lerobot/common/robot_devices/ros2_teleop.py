#!/usr/bin/env python3
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

"""ROS2 nodes for robot teleoperation between leader and follower robots."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
import torch

from lerobot.common.robot_devices.control_utils import busy_wait
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.robots.errors import RobotDeviceNotConnectedError


@dataclass
class ROS2TeleopConfig:
    """Configuration for ROS2 teleoperation."""
    node_name: str = "lerobot_teleop"
    topic_name: str = "joint_states"
    publish_rate_hz: float = 200.0  # Default publish rate
    use_best_effort_qos: bool = True  # Use best effort QoS for real-time performance


class LeaderNode(Node):
    """ROS2 node for the leader robot that publishes joint states."""
    
    def __init__(self, robot: ManipulatorRobot, config: ROS2TeleopConfig):
        super().__init__(config.node_name + "_leader")
        
        self.robot = robot
        self.config = config
        
        # Set up QoS profile - for real-time control, best effort is often better than reliable
        qos = QoSProfile(depth=10)
        if config.use_best_effort_qos:
            qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
            
        # Create publisher
        self.publisher = self.create_publisher(
            JointState,
            config.topic_name,
            qos
        )
        
        # Create timer for publishing at specified rate
        period = 1.0 / config.publish_rate_hz
        self.timer = self.create_timer(period, self.publish_joint_states)
        
        self.get_logger().info(f"Leader node initialized, publishing to {config.topic_name}")
        
    def publish_joint_states(self):
        """Read joint states from leader robot and publish them."""
        if not self.robot.is_connected:
            self.get_logger().error("Robot is not connected")
            return
            
        # Read positions from all leader arms
        leader_pos = {}
        joint_names = []
        positions = []
        
        try:
            for name in self.robot.leader_arms:
                pos = self.robot.leader_arms[name].read("Present_Position")
                leader_pos[name] = pos
                
                # Extend the joint names and positions lists
                motor_names = self.robot.leader_arms[name].motor_names
                joint_names.extend([f"{name}_{motor_name}" for motor_name in motor_names])
                positions.extend(pos.tolist())
                
            # Create and publish the message
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = positions
            
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish joint states: {e}")


class FollowerNode(Node):
    """ROS2 node for the follower robot that subscribes to joint states."""
    
    def __init__(self, robot: ManipulatorRobot, config: ROS2TeleopConfig):
        super().__init__(config.node_name + "_follower")
        
        self.robot = robot
        self.config = config
        self.joint_positions = {}  # Store the latest joint positions
        
        # Set up QoS profile
        qos = QoSProfile(depth=10)
        if config.use_best_effort_qos:
            qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
            
        # Create subscription
        self.subscription = self.create_subscription(
            JointState,
            config.topic_name,
            self.joint_states_callback,
            qos
        )
        
        self.get_logger().info(f"Follower node initialized, subscribing to {config.topic_name}")
        
    def joint_states_callback(self, msg: JointState):
        """Process incoming joint states and command the follower robot."""
        if not self.robot.is_connected:
            self.get_logger().error("Robot is not connected")
            return
            
        try:
            # Parse the message and organize by arm name
            arm_positions = {}
            
            for i, joint_name in enumerate(msg.name):
                # Joint names are expected to be in format: "arm_name_motor_name"
                parts = joint_name.split("_", 1)
                if len(parts) < 2:
                    continue
                    
                arm_name, _ = parts
                
                if arm_name not in arm_positions:
                    arm_positions[arm_name] = []
                    
                arm_positions[arm_name].append(msg.position[i])
            
            # Send positions to follower arms that match leader arms
            for name, positions in arm_positions.items():
                if name in self.robot.follower_arms:
                    # Convert to numpy array
                    pos_array = np.array(positions, dtype=np.float32)
                    
                    # Store the latest positions
                    self.joint_positions[name] = pos_array
                    
                    # Send to the follower robot
                    self.robot.follower_arms[name].write("Goal_Position", pos_array)
                    
        except Exception as e:
            self.get_logger().error(f"Failed to process joint states: {e}")
        

def run_ros2_leader(robot: ManipulatorRobot, config: Optional[ROS2TeleopConfig] = None):
    """Run the ROS2 leader node until interrupted."""
    if not robot.is_connected:
        robot.connect()
        
    if config is None:
        config = ROS2TeleopConfig()
        
    rclpy.init()
    node = LeaderNode(robot, config)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        

def run_ros2_follower(robot: ManipulatorRobot, config: Optional[ROS2TeleopConfig] = None):
    """Run the ROS2 follower node until interrupted."""
    if not robot.is_connected:
        robot.connect()
        
    if config is None:
        config = ROS2TeleopConfig()
        
    rclpy.init()
    node = FollowerNode(robot, config)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
