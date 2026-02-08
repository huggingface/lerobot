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
Unified server script for bimanual Yam arms setup.

This script starts all 4 arm servers (2 followers + 2 leaders) in a single process,
making it much easier to set up compared to running 4 separate scripts.

Example usage:
    # Start all 4 arm servers with default configuration
    python src/lerobot/robots/bi_yam_follower/run_bimanual_yam_server.py

    # Customize CAN channels and ports
    python src/lerobot/robots/bi_yam_follower/run_bimanual_yam_server.py \
        --right_follower_can can_follower_r \
        --left_follower_can can_follower_l \
        --right_leader_can can_leader_r \
        --left_leader_can can_leader_l

    # Run in follower-only mode (no teaching handles)
    python src/lerobot/robots/bi_yam_follower/run_bimanual_yam_server.py \
        --mode follower_only
"""

import sys
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import portal
import tyro
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import GripperType


class ServerRobot:
    """A simple server for a robot arm."""

    def __init__(self, robot: Robot, port: int, name: str):
        self._robot = robot
        self._name = name
        self._server = portal.Server(port)
        self._port = port
        self._running = False

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve_background(self) -> threading.Thread:
        """Start serving the robot in a background thread."""

        def serve_thread():
            try:
                self._running = True
                self._server.start()
            except Exception as e:
                print(f"[{self._name}] server error: {e}")
            finally:
                self._running = False

        thread = threading.Thread(target=serve_thread, daemon=True, name=self._name)
        thread.start()
        return thread

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


class YAMLeaderRobot:
    """Wrapper for YAM leader arm with teaching handle support."""

    def __init__(self, robot: MotorChainRobot, name: str):
        self._robot = robot
        self._motor_chain = robot.motor_chain
        self._name = name

    def num_dofs(self) -> int:
        """Get number of DOFs (6 joints + 1 gripper)."""
        return 7

    def get_joint_pos(self) -> np.ndarray:
        """Get leader arm state including gripper encoder.

        Returns:
            np.ndarray: Joint positions with gripper (7 values)
        """
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.01)
        # Gripper command from encoder (inverted: 0 = open, 1 = closed)
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Not used for leader arms in teaching mode."""
        pass

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        """Not used for leader arms in teaching mode."""
        pass

    def get_observations(self) -> dict[str, np.ndarray]:
        """Get observations from leader arm."""
        joint_pos = self.get_joint_pos()
        return {"joint_pos": joint_pos}


@dataclass
class Args:
    """Command line arguments for the bimanual Yam arm server."""

    mode: Literal["full", "follower_only"] = "full"
    """Operating mode: 'full' (followers + leaders) or 'follower_only' (just followers)."""

    # Follower arms
    right_follower_can: str = "can_follower_r"
    """CAN interface for right follower arm."""

    left_follower_can: str = "can_follower_l"
    """CAN interface for left follower arm."""

    right_follower_port: int = 1234
    """Server port for right follower arm."""

    left_follower_port: int = 1235
    """Server port for left follower arm."""

    follower_gripper: Literal["crank_4310", "linear_3507", "linear_4310", "no_gripper", "v3"] = "v3"
    """Type of gripper on follower arms."""

    # Leader arms (only used if mode == 'full')
    right_leader_can: str = "can_leader_r"
    """CAN interface for right leader arm."""

    left_leader_can: str = "can_leader_l"
    """CAN interface for left leader arm."""

    right_leader_port: int = 5001
    """Server port for right leader arm."""

    left_leader_port: int = 5002
    """Server port for left leader arm."""

    leader_gripper: Literal["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "v3"] = (
        "yam_teaching_handle"
    )
    """Type of gripper/teaching handle on leader arms."""


def main(args: Args) -> None:
    """Main entry point for the bimanual Yam arm server."""
    # Map "v3" to the correct gripper type
    follower_gripper_name = "linear_4310" if args.follower_gripper == "v3" else args.follower_gripper
    leader_gripper_name = "linear_4310" if args.leader_gripper == "v3" else args.leader_gripper

    follower_gripper_type = GripperType.from_string_name(follower_gripper_name)
    leader_gripper_type = GripperType.from_string_name(leader_gripper_name)

    servers = []
    threads = []

    try:
        # Initialize and start follower arms
        print("connecting to follower arms...")
        right_follower_robot = get_yam_robot(
            channel=args.right_follower_can, gripper_type=follower_gripper_type
        )
        right_follower_server = ServerRobot(right_follower_robot, args.right_follower_port, "right_follower")
        servers.append(right_follower_server)

        left_follower_robot = get_yam_robot(
            channel=args.left_follower_can, gripper_type=follower_gripper_type
        )
        left_follower_server = ServerRobot(left_follower_robot, args.left_follower_port, "left_follower")
        servers.append(left_follower_server)

        # Initialize and start leader arms (if in full mode)
        if args.mode == "full":
            print("connecting to leader arms...")
            right_leader_robot = get_yam_robot(
                channel=args.right_leader_can, gripper_type=leader_gripper_type
            )
            right_leader_wrapped = YAMLeaderRobot(right_leader_robot, "right_leader")
            right_leader_server = ServerRobot(right_leader_wrapped, args.right_leader_port, "right_leader")
            servers.append(right_leader_server)

            left_leader_robot = get_yam_robot(channel=args.left_leader_can, gripper_type=leader_gripper_type)
            left_leader_wrapped = YAMLeaderRobot(left_leader_robot, "left_leader")
            left_leader_server = ServerRobot(left_leader_wrapped, args.left_leader_port, "left_leader")
            servers.append(left_leader_server)

        # Start all servers
        for server in servers:
            thread = server.serve_background()
            threads.append(thread)
            time.sleep(0.2)  # Small delay between server starts

        # Wait for all servers to be running
        time.sleep(1.0)
        all_running = all(server.is_running() for server in servers)

        if not all_running:
            print("error: some servers failed to start")
            sys.exit(1)

        # Print status
        print("\nbimanual yam server running:")
        print(f"  right follower: port {args.right_follower_port} (can: {args.right_follower_can})")
        print(f"  left follower:  port {args.left_follower_port} (can: {args.left_follower_can})")
        if args.mode == "full":
            print(f"  right leader:   port {args.right_leader_port} (can: {args.right_leader_can})")
            print(f"  left leader:    port {args.left_leader_port} (can: {args.left_leader_can})")
        print()

        # Keep main thread alive
        for thread in threads:
            thread.join()

    except KeyboardInterrupt:
        print("\nshutting down...")
    except Exception as e:
        print(f"error during startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
