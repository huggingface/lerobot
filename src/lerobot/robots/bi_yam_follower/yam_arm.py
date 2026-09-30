#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Native YAM motor mapping, unit conversion, and per-arm impedance targets."""

import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.damiao import MotorState
from lerobot.utils.import_utils import _mujoco_available, require_package

from .config_bi_yam_follower import JOINT_LIMITS, MOTOR_NAMES, YamArmConfig

if TYPE_CHECKING or _mujoco_available:
    import mujoco


def verify_adapter(config: YamArmConfig) -> None:
    """Check the USB ancestor of a SocketCAN interface before opening either arm."""
    if config.expected_adapter_serial is None:
        return
    device = (Path("/sys/class/net") / config.port / "device").resolve()
    for parent in (device, *device.parents):
        serial = parent / "serial"
        if serial.is_file():
            actual = serial.read_text().strip()
            if actual != config.expected_adapter_serial:
                raise ValueError(f"{config.port} adapter serial {actual!r} does not match the configured arm")
            return
    raise ValueError(f"Cannot verify USB serial for {config.port}; check the adapter connection")


def make_yam_bus(config: YamArmConfig) -> DamiaoMotorsBus:
    """IDs 1..7 / feedback 17..23, classic CAN at 1 Mbit/s (not OpenArm CAN FD)."""
    return DamiaoMotorsBus(
        port=config.port,
        can_interface="socketcan",
        use_can_fd=False,
        bitrate=1_000_000,
        motors={
            name: Motor(
                i + 1,
                "dm4340" if i < 3 else "dm4310",
                MotorNormMode.DEGREES,
                motor_type_str="dm4340" if i < 3 else "dm4310",
                recv_id=i + 17,
            )
            for i, name in enumerate(MOTOR_NAMES)
        },
    )


def decode_positions(config: YamArmConfig, states: dict[str, MotorState]) -> np.ndarray:
    """Convert bus degrees to dataset radians, and gripper 0=closed / 1=open."""
    closed, opened = config.gripper_closed_rad, config.gripper_open_rad
    if closed is None or opened is None:
        raise ValueError("Measure and configure gripper_closed_rad and gripper_open_rad before connecting")
    raw = np.radians([states[name]["position"] for name in MOTOR_NAMES])
    if not np.isfinite(raw).all():
        raise ConnectionError("Non-finite YAM feedback")
    joints = raw[:6] * np.asarray(config.joint_signs) + np.asarray(config.joint_offsets_rad)
    gripper = (raw[6] - closed) / (opened - closed)
    if not -0.05 <= gripper <= 1.05:
        raise ValueError("Gripper feedback is outside the calibrated stroke; check endpoints")
    return np.r_[joints, np.clip(gripper, 0, 1)]


def encode_positions(config: YamArmConfig, positions: np.ndarray) -> np.ndarray:
    """Inverse of decode_positions, returning the motor's native degrees."""
    closed, opened = config.gripper_closed_rad, config.gripper_open_rad
    if closed is None or opened is None:
        raise ValueError("Missing gripper calibration")
    raw = (positions[:6] - np.asarray(config.joint_offsets_rad)) / np.asarray(config.joint_signs)
    return np.degrees(np.r_[raw, closed + float(positions[6]) * (opened - closed)])


def validate_target(values: np.ndarray, *, feedback: bool = False) -> None:
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("YAM requires six finite joint radians and one normalized gripper position")
    for i, (value, (lower, upper)) in enumerate(zip(values, (*JOINT_LIMITS, (0, 1)), strict=True)):
        tolerance = 0.03 if feedback and i < 6 else 0.0
        if not lower - tolerance <= value <= upper + tolerance:
            raise ValueError(f"YAM joint/gripper {i} target {value} outside [{lower}, {upper}]")


class GravityCompensation:
    def __init__(self) -> None:
        require_package("mujoco", extra="yam")
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "assets/yam_linear.xml"))
        self.data = mujoco.MjData(self.model)

    def torque(self, positions: np.ndarray) -> np.ndarray:
        self.data.qpos[:6] = positions[:6]
        self.data.qpos[6:] = positions[6] * 0.0475
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[:6].copy()


class YamArm:
    def __init__(self, config: YamArmConfig) -> None:
        self.config = config
        self.bus = make_yam_bus(config)
        self.position = np.zeros(7)
        self.target = np.zeros(7)
        self.command = np.zeros(7)
        self.updated_at = 0.0
        self.commanded_at = 0.0
        self.command_timed_out = False
        self.gravity: GravityCompensation | None = None
        self.thread: threading.Thread | None = None
        self.ready = threading.Event()
        self.control_ready = threading.Event()
        self.enabled = False

    def command_packet(
        self, position: np.ndarray, dt: float
    ) -> dict[str, tuple[float, float, float, float, float]]:
        cfg = self.config
        speeds = np.r_[np.full(6, cfg.max_joint_speed_rad_s), cfg.max_gripper_speed_s]
        self.command += np.clip(self.target - self.command, -speeds * dt, speeds * dt)
        self.command[:6] = np.clip(
            self.command[:6],
            position[:6] - cfg.max_tracking_error_rad,
            position[:6] + cfg.max_tracking_error_rad,
        )
        raw_goal = encode_positions(cfg, self.command)
        raw_position = encode_positions(cfg, position)
        # Limit proportional closing/opening torque even on a blocked gripper.
        gripper_error_deg = math.degrees(cfg.gripper_torque_limit / cfg.gripper_kp)
        raw_goal[6] = np.clip(
            raw_goal[6], raw_position[6] - gripper_error_deg, raw_position[6] + gripper_error_deg
        )
        gravity = np.zeros(6) if self.gravity is None else self.gravity.torque(position)
        gravity *= np.asarray(cfg.gravity_factors) * np.asarray(cfg.joint_signs)
        gravity = np.clip(gravity, -10.0, 10.0)
        kp, kd = [*cfg.kp, cfg.gripper_kp], [*cfg.kd, cfg.gripper_kd]
        return {
            name: (kp[i], kd[i], float(raw_goal[i]), 0.0, float(gravity[i]) if i < 6 else 0.0)
            for i, name in enumerate(MOTOR_NAMES)
        }
