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
Server script for a single YAM follower arm or a MuJoCo visualizer.

For the standard bimanual teleoperation/recording workflow, use
``run_bimanual_yam_server.py`` instead — it starts all four arms (2 followers +
2 leaders) in one process and binds the RPC endpoints required by
``bi_yam_follower`` / ``bi_yam_leader``.

Example usage:
    # Single follower arm
    python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
        --can_channel can_follower_r --gripper v3 --mode follower --server_port 1234

    # MuJoCo visualizer (local)
    python src/lerobot/robots/bi_yam_follower/run_yam_server.py \
        --can_channel can_follower_r --gripper v3 --mode visualizer_local
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import portal
import tyro
from i2rt.robots.get_robot import get_yam_robot
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


@dataclass
class Args:
    """Command line arguments for the Yam arm server."""

    gripper: Literal[
        "crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper", "v3"
    ] = "yam_teaching_handle"
    """Type of gripper attached to the arm."""

    mode: Literal["follower", "visualizer_local", "visualizer_remote"] = "follower"
    """Operating mode: follower (execution) or visualizer."""

    server_host: str = "localhost"
    """Hostname for the server (used by visualizer_remote to connect to a follower)."""

    server_port: int = DEFAULT_ROBOT_PORT
    """Port number for the server."""

    can_channel: str = "can0"
    """CAN interface name (e.g., can0, can_follower_r)."""


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
