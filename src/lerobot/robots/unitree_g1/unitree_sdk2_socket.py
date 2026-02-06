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

"""
ZMQ-based SDK wrapper for Unitree G1 robot with Dex3 hands.

This module provides SDK-compatible classes that use ZMQ to communicate with
the robot server, avoiding the need to import from unitree_sdk2py (which has
circular import issues).
"""

import base64
import json
import threading
from typing import Any

import zmq

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config

# Global ZMQ context and sockets
_ctx: zmq.Context | None = None
_lowcmd_sock: zmq.Socket | None = None
_lowstate_sock: zmq.Socket | None = None
_handcmd_sock: zmq.Socket | None = None
_handstate_sock: zmq.Socket | None = None
_robot_ip: str = ""

# Latest hand states (updated by background thread)
_left_hand_state: "HandStateMsg | None" = None
_right_hand_state: "HandStateMsg | None" = None
_hand_state_lock = threading.Lock()
_hand_thread: threading.Thread | None = None
_hand_shutdown_event: threading.Event | None = None

# ZMQ ports (must match server)
LOWCMD_PORT = 6000
LOWSTATE_PORT = 6001
HANDSTATE_PORT = 6002
HANDCMD_PORT = 6003

# DDS topic names (for protocol compatibility)
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


# ==============================================================================
# SDK-free wrapper classes for ZMQ mode
# ==============================================================================

class MotorCmd:
    """Motor command data for a single joint (SDK-free wrapper)."""
    def __init__(self):
        self.mode: int = 0
        self.q: float = 0.0
        self.dq: float = 0.0
        self.kp: float = 0.0
        self.kd: float = 0.0
        self.tau: float = 0.0


class LowCmdMsg:
    """Low-level command message (SDK-free wrapper for unitree_hg_msg_dds__LowCmd_)."""
    def __init__(self, num_motors: int = 35):
        self.mode_pr: int = 0
        self.mode_machine: int = 0
        self.motor_cmd: list[MotorCmd] = [MotorCmd() for _ in range(num_motors)]
        self.crc: int = 0


class HandCmdMsg:
    """Hand command message (SDK-free wrapper for unitree_hg_msg_dds__HandCmd_)."""
    def __init__(self, num_motors: int = 7):
        self.motor_cmd: list[MotorCmd] = [MotorCmd() for _ in range(num_motors)]


class CRC:
    """No-op CRC class for ZMQ mode (CRC is handled by the robot server)."""
    def Crc(self, msg: Any) -> int:  # noqa: N802
        return 0  # CRC not needed for ZMQ mode


# Placeholder types for channel initialization (not actually used in ZMQ mode)
hg_LowCmd = None  # noqa: N816
hg_LowState = None  # noqa: N816


class MotorState:
    """Motor state data for a single joint."""
    def __init__(self, data: dict[str, Any]) -> None:
        self.q: float = data.get("q", 0.0)
        self.dq: float = data.get("dq", 0.0)
        self.tau_est: float = data.get("tau_est", 0.0)
        self.temperature: float = data.get("temperature", 0.0)


class LowStateMsg:
    """Wrapper class that mimics the Unitree SDK LowState_ message structure."""

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
        self.motor_state = [MotorState(m) for m in data.get("motor_state", [])]
        self.imu_state = self.IMUState(data.get("imu_state", {}))
        # Decode base64-encoded wireless_remote bytes
        wireless_b64 = data.get("wireless_remote", "")
        self.wireless_remote: bytes = base64.b64decode(wireless_b64) if wireless_b64 else b""
        self.mode_machine: int = data.get("mode_machine", 0)


class HandStateMsg:
    """Wrapper class that mimics the Unitree SDK HandState_ message structure."""
    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize from deserialized JSON data."""
        self.side: str = data.get("side", "")
        self.motor_state = [MotorState(m) for m in data.get("motor_state", [])]


# ==============================================================================
# Serialization helpers
# ==============================================================================

def lowcmd_to_dict(topic: str, msg: Any) -> dict[str, Any]:
    """Convert LowCmd message to a JSON-serializable dictionary."""
    motor_cmds = []
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
            "mode_pr": int(msg.mode_pr) if hasattr(msg, "mode_pr") else 0,
            "mode_machine": int(msg.mode_machine) if hasattr(msg, "mode_machine") else 0,
            "motor_cmd": motor_cmds,
        },
    }


def handcmd_to_dict(topic: str, msg: HandCmdMsg) -> dict[str, Any]:
    """Convert HandCmd message to a JSON-serializable dictionary."""
    motor_cmds = []
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
        "data": {"motor_cmd": motor_cmds},
    }


# ==============================================================================
# Hand state subscription thread
# ==============================================================================

def _hand_state_thread_fn(sock: zmq.Socket, shutdown_event: threading.Event) -> None:
    """Background thread to receive hand state updates."""
    global _left_hand_state, _right_hand_state
    
    while not shutdown_event.is_set():
        try:
            payload = sock.recv(zmq.NOBLOCK)
            msg_dict = json.loads(payload.decode("utf-8"))
            topic = msg_dict.get("topic", "")
            data = msg_dict.get("data", {})
            
            with _hand_state_lock:
                if topic == kTopicDex3LeftState:
                    _left_hand_state = HandStateMsg(data)
                elif topic == kTopicDex3RightState:
                    _right_hand_state = HandStateMsg(data)
        except zmq.Again:
            pass  # No data available
        except Exception:
            pass  # Ignore errors in background thread
        
        shutdown_event.wait(0.001)  # Small sleep


# ==============================================================================
# ZMQ Channel classes (SDK-compatible interface)
# ==============================================================================

def ChannelFactoryInitialize(*args: Any, **kwargs: Any) -> None:  # noqa: N802
    """Initialize ZMQ sockets for robot communication."""
    global _ctx, _lowcmd_sock, _lowstate_sock, _handcmd_sock, _handstate_sock
    global _robot_ip, _hand_thread, _hand_shutdown_event

    if _ctx is not None:
        return

    # Read socket config
    config = UnitreeG1Config()
    _robot_ip = config.robot_ip

    ctx = zmq.Context.instance()
    _ctx = ctx

    # Body lowcmd: send robot commands
    lowcmd_sock = ctx.socket(zmq.PUSH)
    lowcmd_sock.setsockopt(zmq.CONFLATE, 1)
    lowcmd_sock.connect(f"tcp://{_robot_ip}:{LOWCMD_PORT}")
    _lowcmd_sock = lowcmd_sock

    # Body lowstate: receive robot observations
    lowstate_sock = ctx.socket(zmq.SUB)
    lowstate_sock.setsockopt(zmq.CONFLATE, 1)
    lowstate_sock.connect(f"tcp://{_robot_ip}:{LOWSTATE_PORT}")
    lowstate_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    _lowstate_sock = lowstate_sock

    # Hand command: send hand commands
    handcmd_sock = ctx.socket(zmq.PUSH)
    handcmd_sock.setsockopt(zmq.CONFLATE, 1)
    handcmd_sock.connect(f"tcp://{_robot_ip}:{HANDCMD_PORT}")
    _handcmd_sock = handcmd_sock

    # Hand state: receive hand observations
    handstate_sock = ctx.socket(zmq.SUB)
    handstate_sock.setsockopt(zmq.CONFLATE, 1)
    handstate_sock.connect(f"tcp://{_robot_ip}:{HANDSTATE_PORT}")
    handstate_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    _handstate_sock = handstate_sock

    # Start hand state subscription thread
    _hand_shutdown_event = threading.Event()
    _hand_thread = threading.Thread(
        target=_hand_state_thread_fn,
        args=(handstate_sock, _hand_shutdown_event),
        daemon=True,
        name="ZMQHandStateSubscriber",
    )
    _hand_thread.start()


class ChannelPublisher:
    """ZMQ-based publisher that sends commands to the robot server."""

    def __init__(self, topic: str, msg_type: type | None) -> None:
        self.topic = topic
        self.msg_type = msg_type

    def Init(self) -> None:  # noqa: N802
        """Initialize the publisher (no-op for ZMQ)."""
        pass

    def Write(self, msg: Any) -> None:  # noqa: N802
        """Serialize and send a command message to the robot."""
        # Select appropriate socket based on topic
        if self.topic == kTopicLowCommand_Debug:
            if _lowcmd_sock is None:
                raise RuntimeError("ChannelFactoryInitialize must be called first")
            payload = json.dumps(lowcmd_to_dict(self.topic, msg)).encode("utf-8")
            _lowcmd_sock.send(payload)
        elif self.topic in (kTopicDex3LeftCommand, kTopicDex3RightCommand):
            if _handcmd_sock is None:
                raise RuntimeError("ChannelFactoryInitialize must be called first")
            payload = json.dumps(handcmd_to_dict(self.topic, msg)).encode("utf-8")
            _handcmd_sock.send(payload)


class ChannelSubscriber:
    """ZMQ-based subscriber that receives state from the robot server."""

    def __init__(self, topic: str, msg_type: type | None) -> None:
        self.topic = topic
        self.msg_type = msg_type

    def Init(self) -> None:  # noqa: N802
        """Initialize the subscriber (no-op for ZMQ)."""
        pass

    def Read(self) -> LowStateMsg | HandStateMsg | None:  # noqa: N802
        """Receive and deserialize a state message from the robot."""
        if self.topic == "rt/lowstate":
            if _lowstate_sock is None:
                raise RuntimeError("ChannelFactoryInitialize must be called first")
            payload = _lowstate_sock.recv()
            msg_dict = json.loads(payload.decode("utf-8"))
            return LowStateMsg(msg_dict.get("data", {}))
        elif self.topic == kTopicDex3LeftState:
            with _hand_state_lock:
                return _left_hand_state
        elif self.topic == kTopicDex3RightState:
            with _hand_state_lock:
                return _right_hand_state
        return None

