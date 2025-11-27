"""
ZMQ socket wrapper that mimics the Unitree SDK Channel interface.

This module provides a drop-in replacement for the Unitree SDK's DDS-based
ChannelPublisher and ChannelSubscriber, using ZMQ sockets instead. This allows
remote communication with the robot over WiFi via the robot_server bridge.

Uses JSON for secure serialization instead of pickle.
"""

import base64
import json
from typing import Any

import zmq

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import (
    DATA,
    IMU_ACCELEROMETER,
    IMU_GYROSCOPE,
    IMU_QUATERNION,
    IMU_RPY,
    IMU_STATE,
    IMU_TEMPERATURE,
    MODE_MACHINE,
    MODE_PR,
    MOTOR_CMD,
    MOTOR_DQ,
    MOTOR_KD,
    MOTOR_KP,
    MOTOR_MODE,
    MOTOR_Q,
    MOTOR_STATE,
    MOTOR_TAU,
    MOTOR_TAU_EST,
    MOTOR_TEMPERATURE,
    TOPIC,
    WIRELESS_REMOTE,
)

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
            self.q: float = data.get(MOTOR_Q, 0.0)
            self.dq: float = data.get(MOTOR_DQ, 0.0)
            self.tau_est: float = data.get(MOTOR_TAU_EST, 0.0)
            self.temperature: float = data.get(MOTOR_TEMPERATURE, 0.0)

    class IMUState:
        """IMU sensor data."""

        def __init__(self, data: dict[str, Any]) -> None:
            self.quaternion: list[float] = data.get(IMU_QUATERNION, [1.0, 0.0, 0.0, 0.0])
            self.gyroscope: list[float] = data.get(IMU_GYROSCOPE, [0.0, 0.0, 0.0])
            self.accelerometer: list[float] = data.get(IMU_ACCELEROMETER, [0.0, 0.0, 0.0])
            self.rpy: list[float] = data.get(IMU_RPY, [0.0, 0.0, 0.0])
            self.temperature: float = data.get(IMU_TEMPERATURE, 0.0)

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize from deserialized JSON data."""
        self.motor_state = [self.MotorState(m) for m in data.get(MOTOR_STATE, [])]
        self.imu_state = self.IMUState(data.get(IMU_STATE, {}))
        # Decode base64-encoded wireless_remote bytes
        wireless_b64 = data.get(WIRELESS_REMOTE, "")
        self.wireless_remote: bytes = base64.b64decode(wireless_b64) if wireless_b64 else b""
        self.mode_machine: int = data.get(MODE_MACHINE, 0)


def lowcmd_to_dict(topic: str, msg: Any) -> dict[str, Any]:
    """Convert LowCmd message to a JSON-serializable dictionary."""
    motor_cmds = []
    # Iterate over all motor commands in the message
    for i in range(len(msg.motor_cmd)):
        motor_cmds.append(
            {
                MOTOR_MODE: msg.motor_cmd[i].mode,
                MOTOR_Q: msg.motor_cmd[i].q,
                MOTOR_DQ: msg.motor_cmd[i].dq,
                MOTOR_KP: msg.motor_cmd[i].kp,
                MOTOR_KD: msg.motor_cmd[i].kd,
                MOTOR_TAU: msg.motor_cmd[i].tau,
            }
        )

    return {
        TOPIC: topic,
        DATA: {
            MODE_PR: msg.mode_pr,
            MODE_MACHINE: msg.mode_machine,
            MOTOR_CMD: motor_cmds,
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
        return LowStateMsg(msg_dict.get(DATA, {}))
