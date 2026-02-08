#!/usr/bin/env python3

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

"""
Server script for Yam arms using i2rt and portal.

This script starts a server process for a single Yam arm (follower or leader).
For bimanual setup, you need to run this script 4 times with different configurations.

Example usage:
    # Right follower arm
    python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
        --can_channel can_follower_r --gripper v3 --mode follower --server_port 1234

    # Left follower arm
    python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
        --can_channel can_follower_l --gripper v3 --mode follower --server_port 1235

    # Right leader arm
    python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
        --can_channel can_leader_r --gripper v3 --mode leader --server_port 5001

    # Left leader arm
    python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
        --can_channel can_leader_l --gripper v3 --mode leader --server_port 5002
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import portal
import tyro
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot

DEFAULT_ROBOT_PORT = 11333


class ServerRobot:
    """A simple server for a robot arm."""

    def __init__(self, robot: Robot, port: int):
        self._robot = robot
        self._server = portal.Server(port)
        print(f"Robot Server Binding to port {port}, Robot: {robot}")

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        """Start serving the robot."""
        self._server.start()


class ClientRobot(Robot):
    """A simple client for a robot arm."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        """Get the current joint positions of the robot.

        Returns:
            np.ndarray: The current joint positions.
        """
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the robot to the given joint positions.

        Args:
            joint_pos: The joint positions to command.
        """
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        """Command the robot to the given joint state.

        Args:
            joint_state: The joint state to command.
        """
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> dict[str, np.ndarray]:
        """Get the current observations of the robot.

        Returns:
            Dict[str, np.ndarray]: The current observations.
        """
        return self._client.get_observations().result()


class YAMLeaderRobot:
    """Wrapper for YAM leader arm with teaching handle support."""

    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> tuple[np.ndarray, np.ndarray]:
        """Get leader arm state including gripper encoder and button inputs.

        Returns:
            tuple: (joint positions with gripper, button states)
        """
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.01)
        # Gripper command from encoder (inverted: 0 = open, 1 = closed)
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the leader arm joint positions (6 joints, excluding gripper).

        Args:
            joint_pos: Joint positions (6 values).
        """
        assert joint_pos.shape[0] == 6, f"Expected 6 joints, got {joint_pos.shape[0]}"
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """Update the PD controller gains.

        Args:
            kp: Proportional gains.
            kd: Derivative gains.
        """
        self._robot.update_kp_kd(kp, kd)


@dataclass
class Args:
    """Command line arguments for the Yam arm server."""

    gripper: Literal[
        "crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper", "v3"
    ] = "yam_teaching_handle"
    """Type of gripper attached to the arm."""

    mode: Literal["follower", "leader", "visualizer_local", "visualizer_remote"] = "follower"
    """Operating mode: follower (execution), leader (teaching), or visualizer."""

    server_host: str = "localhost"
    """Hostname for the server (used in leader mode to connect to follower)."""

    server_port: int = DEFAULT_ROBOT_PORT
    """Port number for the server."""

    can_channel: str = "can0"
    """CAN interface name (e.g., can0, can_follower_r, can_leader_l)."""

    bilateral_kp: float = 0.0
    """Bilateral force feedback gain (used in leader mode)."""


def main(args: Args) -> None:
    """Main entry point for the Yam arm server."""
    from i2rt.robots.utils import GripperType

    # Map "v3" to the correct gripper type
    gripper_name = args.gripper
    if gripper_name == "v3":
        gripper_name = "linear_4310"  # v3 is typically linear_4310

    gripper_type = GripperType.from_string_name(gripper_name)

    # Initialize robot (except for remote visualizer mode)
    if "remote" not in args.mode:
        robot = get_yam_robot(channel=args.can_channel, gripper_type=gripper_type)

    if args.mode == "follower":
        # Run as follower: serve robot state and accept commands
        server_robot = ServerRobot(robot, args.server_port)
        print(f"Starting follower server on port {args.server_port}...")
        print("Press Ctrl+C to stop.")
        server_robot.serve()

    elif args.mode == "leader":
        # Run as leader: read teaching handle state and mirror to follower
        robot = YAMLeaderRobot(robot)
        robot_current_kp = robot._robot._kp
        client_robot = ClientRobot(args.server_port, host=args.server_host)

        # Sync the robot state
        current_joint_pos, current_button = robot.get_info()
        current_follower_joint_pos = client_robot.get_joint_pos()
        print(f"Current leader joint pos: {current_joint_pos}")
        print(f"Current follower joint pos: {current_follower_joint_pos}")

        def slow_move(joint_pos: np.ndarray, duration: float = 1.0) -> None:
            """Smoothly move follower to match leader position."""
            for i in range(100):
                current_joint_pos_local = joint_pos
                follower_command_joint_pos = (
                    current_joint_pos_local * i / 100 + current_follower_joint_pos * (1 - i / 100)
                )
                client_robot.command_joint_pos(follower_command_joint_pos)
                time.sleep(0.03)

        synchronized = False
        print("\nLeader mode started. Press the teaching handle button to toggle synchronization.")
        print("Press Ctrl+C to stop.")

        while True:
            current_joint_pos, current_button = robot.get_info()

            # Button press toggles synchronization
            if current_button[0] > 0.5:
                if not synchronized:
                    print("Synchronizing: Follower will now mirror leader.")
                    robot.update_kp_kd(kp=robot_current_kp * args.bilateral_kp, kd=np.ones(6) * 0.0)
                    robot.command_joint_pos(current_joint_pos[:6])
                    slow_move(current_joint_pos)
                else:
                    print("Desynchronizing: Clearing bilateral PD.")
                    robot.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)
                    robot.command_joint_pos(current_follower_joint_pos[:6])

                synchronized = not synchronized

                # Wait for button release
                while current_button[0] > 0.5:
                    time.sleep(0.03)
                    current_joint_pos, current_button = robot.get_info()

            current_follower_joint_pos = client_robot.get_joint_pos()

            if synchronized:
                # Mirror leader to follower
                client_robot.command_joint_pos(current_joint_pos)
                # Set bilateral force proportional to bilateral_kp
                robot.command_joint_pos(current_follower_joint_pos[:6])

            time.sleep(0.01)

    elif "visualizer" in args.mode:
        # Run visualizer using MuJoCo
        import mujoco
        import mujoco.viewer

        if args.mode == "visualizer_remote":
            robot = ClientRobot(args.server_port, host=args.server_host)

        xml_path = gripper_type.get_xml_path()
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        dt: float = 0.01
        print(f"Starting MuJoCo visualizer. Loading model from: {xml_path}")
        print("Press Ctrl+C to stop.")

        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                step_start = time.time()
                joint_pos = robot.get_joint_pos()
                data.qpos[:] = joint_pos[: model.nq]

                # Sync the model state
                mujoco.mj_kinematics(model, data)
                viewer.sync()

                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main(tyro.cli(Args))
