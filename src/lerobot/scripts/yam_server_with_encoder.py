#!/usr/bin/env python3
"""
Enhanced Yam arm server that exposes encoder data for teaching handles.

This script wraps the i2rt robot to expose encoder button states through
the portal RPC interface, allowing LeRobot to read gripper commands from
the teaching handle.

Based on i2rt's minimum_gello.py but with encoder support.
"""

import time
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import portal
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import GripperType

DEFAULT_ROBOT_PORT = 11333


class EnhancedYamRobot(Robot):
    """
    Wrapper around MotorChainRobot that exposes encoder data.
    
    For teaching handles, reads encoder position and button states
    to provide gripper control information.
    """

    def __init__(self, robot: MotorChainRobot, is_teaching_handle: bool = False):
        self._robot = robot
        self._motor_chain = robot.motor_chain
        self._is_teaching_handle = is_teaching_handle

    def num_dofs(self) -> int:
        """Get the number of joints in the robot."""
        return self._robot.num_dofs()

    def get_joint_pos(self) -> np.ndarray:
        """Get the current joint positions."""
        joint_pos = self._robot.get_joint_pos()
        
        # For teaching handles, add gripper state from encoder
        if self._is_teaching_handle:
            try:
                encoder_states = self._motor_chain.get_same_bus_device_states()
                if encoder_states and len(encoder_states) > 0:
                    # Encoder position mapped to gripper (0=closed, 1=open)
                    gripper_pos = 1 - encoder_states[0].position
                    joint_pos = np.concatenate([joint_pos, [gripper_pos]])
            except Exception as e:
                print(f"Warning: Could not read encoder state: {e}")
                # Fallback to default open position
                joint_pos = np.concatenate([joint_pos, [1.0]])
        
        return joint_pos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the robot to a given joint position."""
        # For teaching handles, ignore gripper command if included
        if self._is_teaching_handle and len(joint_pos) > self._robot.num_dofs():
            joint_pos = joint_pos[: self._robot.num_dofs()]
        
        self._robot.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command the robot to a given state."""
        self._robot.command_joint_state(joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """
        Get the current observations of the robot.
        
        For teaching handles, includes encoder data:
        - joint_pos: 6 joint positions
        - gripper_pos: Encoder position mapped to gripper (0=closed, 1=open)
        - io_inputs: Button states from encoder
        """
        obs = self._robot.get_observations()
        
        # For teaching handles, add encoder data
        if self._is_teaching_handle:
            try:
                encoder_states = self._motor_chain.get_same_bus_device_states()
                if encoder_states and len(encoder_states) > 0:
                    # Add gripper position from encoder
                    gripper_pos = 1 - encoder_states[0].position
                    obs["gripper_pos"] = np.array([gripper_pos])
                    
                    # Add button states
                    obs["io_inputs"] = encoder_states[0].io_inputs
            except Exception as e:
                print(f"Warning: Could not read encoder state: {e}")
                # Provide defaults
                obs["gripper_pos"] = np.array([1.0])
                obs["io_inputs"] = np.array([0.0])
        
        return obs


class ServerRobot:
    """A simple server for a robot."""

    def __init__(self, robot: Robot, port: int):
        self._robot = robot
        self._server = portal.Server(port)
        print(f"Enhanced Robot Server Binding to {port}, Robot: {robot}")

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        """Serve the robot."""
        self._server.start()


@dataclass
class Args:
    gripper: Literal["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"] = (
        "yam_teaching_handle"
    )
    mode: Literal["follower", "leader"] = "follower"
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT
    can_channel: str = "can0"


def main(args: Args) -> None:
    """Main function to start the enhanced Yam server."""
    gripper_type = GripperType.from_string_name(args.gripper)
    is_teaching_handle = gripper_type == GripperType.YAM_TEACHING_HANDLE

    # Get the base robot from i2rt
    base_robot = get_yam_robot(channel=args.can_channel, gripper_type=gripper_type)

    # Wrap it with encoder support
    robot = EnhancedYamRobot(base_robot, is_teaching_handle=is_teaching_handle)

    # Start the server
    server_robot = ServerRobot(robot, args.server_port)
    
    print(f"\n{'='*60}")
    print(f"Enhanced Yam Server Started")
    print(f"  CAN Channel: {args.can_channel}")
    print(f"  Gripper Type: {args.gripper}")
    print(f"  Teaching Handle: {is_teaching_handle}")
    print(f"  Port: {args.server_port}")
    if is_teaching_handle:
        print(f"  Encoder Support: ENABLED ✓")
    print(f"{'='*60}\n")
    
    server_robot.serve()


if __name__ == "__main__":
    main(tyro.cli(Args))

