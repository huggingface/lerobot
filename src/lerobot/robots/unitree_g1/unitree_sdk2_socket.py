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

import base64
import json
from typing import Any

import zmq

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

_ctx: zmq.Context | None = None
_lowcmd_sock: zmq.Socket | None = None
_lowstate_sock: zmq.Socket | None = None

LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001

# DDS topic names follow Unitree SDK naming conventions
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"


class LowStateMsg:
    """
    Wrapper class that mimics the Unitree SDK LowState_ message structure.

    Reconstructs the message from deserialized JSON data to maintain
    compatibility with existing code that expects SDK message objects.
    """

    class MotorState:
        """Motor state data for a single joint."""

        def __init__(self, data: dict[str, Any]) -> None:
            self.q: float = data.get("q", 0.0)
            self.dq: float = data.get("dq", 0.0)
            self.tau_est: float = data.get("tau_est", 0.0)
            self.temperature: float = data.get("temperature", 0.0)

    class IMUState:
        """IMU sensor data."""

        def __init__(self, data: dict[str, Any]) -> None:
            self.quaternion: list[float] = data.get("quaternion", [1.0, 0.0, 0.0, 0.0])
            self.gyroscope: list[float] = data.get("gyroscope", [0.0, 0.0, 0.0])
            self.accelerometer: list[float] = data.get("accelerometer", [0.0, 0.0, 0.0])
            self.rpy: list[float] = data.get("rpy", [0.0, 0.0, 0.0])
            self.temperature: float = data.get("temperature", 0.0)

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize from deserialized JSON data."""
        self.motor_state = [self.MotorState(m) for m in data.get("motor_state", [])]
        self.imu_state = self.IMUState(data.get("imu_state", {}))
        # Decode base64-encoded wireless_remote bytes
        wireless_b64 = data.get("wireless_remote", "")
        self.wireless_remote: bytes = base64.b64decode(wireless_b64) if wireless_b64 else b""
        self.mode_machine: int = data.get("mode_machine", 0)


def lowcmd_to_dict(topic: str, msg: Any) -> dict[str, Any]:
    """Convert LowCmd message to a JSON-serializable dictionary."""
    motor_cmds = []
    # Iterate over all motor commands in the message
    for i in range(len(msg.motor_cmd)):
        motor_cmds.append(
            {
                "mode": int(msg.motor_cmd[i].mode),
                "q": float(msg.motor_cmd[i].q),
                "dq": float(msg.motor_cmd[i].dq),
                "kp": float(msg.motor_cmd[i].kp),
                "kd": float(msg.motor_cmd[i].kd),
                "tau": float(msg.motor_cmd[i].tau),
            }
        )

    return {
        "topic": topic,
        "data": {
            "mode_pr": int(msg.mode_pr),
            "mode_machine": int(msg.mode_machine),
            "motor_cmd": motor_cmds,
        },
    }


def ChannelFactoryInitialize(*args: Any, **kwargs: Any) -> None:  # noqa: N802
    """
    Initialize ZMQ sockets for robot communication.

    This function mimics the Unitree SDK's ChannelFactoryInitialize but uses
    ZMQ sockets to connect to the robot server bridge instead of DDS.
    """
    global _ctx, _lowcmd_sock, _lowstate_sock

    # read socket config
    config = UnitreeG1Config()
    robot_ip = config.robot_ip

    ctx = zmq.Context.instance()
    _ctx = ctx

    # lowcmd: send robot commands
    lowcmd_sock = ctx.socket(zmq.PUSH)
    lowcmd_sock.setsockopt(zmq.CONFLATE, 1)  # keep only last message
    lowcmd_sock.connect(f"tcp://{robot_ip}:{LOWCMD_PORT}")
    _lowcmd_sock = lowcmd_sock

    # lowstate: receive robot observations
    lowstate_sock = ctx.socket(zmq.SUB)
    lowstate_sock.setsockopt(zmq.CONFLATE, 1)  # keep only last message
    lowstate_sock.connect(f"tcp://{robot_ip}:{LOWSTATE_PORT}")
    lowstate_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    _lowstate_sock = lowstate_sock


class ChannelPublisher:
    """ZMQ-based publisher that sends commands to the robot server."""

    def __init__(self, topic: str, msg_type: type) -> None:
        self.topic = topic
        self.msg_type = msg_type

    def Init(self) -> None:  # noqa: N802
        """Initialize the publisher (no-op for ZMQ)."""
        pass

    def Write(self, msg: Any) -> None:  # noqa: N802
        """Serialize and send a command message to the robot."""
        if _lowcmd_sock is None:
            raise RuntimeError("ChannelFactoryInitialize must be called first")

        payload = json.dumps(lowcmd_to_dict(self.topic, msg)).encode("utf-8")
        _lowcmd_sock.send(payload)


class ChannelSubscriber:
    """ZMQ-based subscriber that receives state from the robot server."""

    def __init__(self, topic: str, msg_type: type) -> None:
        self.topic = topic
        self.msg_type = msg_type

    def Init(self) -> None:  # noqa: N802
        """Initialize the subscriber (no-op for ZMQ)."""
        pass

    def Read(self) -> LowStateMsg:  # noqa: N802
        """Receive and deserialize a state message from the robot."""
        if _lowstate_sock is None:
            raise RuntimeError("ChannelFactoryInitialize must be called first")

        payload = _lowstate_sock.recv()
        msg_dict = json.loads(payload.decode("utf-8"))
        return LowStateMsg(msg_dict.get("data", {}))
