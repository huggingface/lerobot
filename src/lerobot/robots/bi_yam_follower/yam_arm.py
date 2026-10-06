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

"""Motorbridge YAM motor mapping, unit conversion, and per-arm impedance targets."""

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.import_utils import (
    _can_available,
    _motorbridge_available,
    _mujoco_available,
    require_package,
)

from .config_bi_yam_follower import JOINT_LIMITS, MOTOR_NAMES, YamArmConfig

if TYPE_CHECKING or _motorbridge_available:
    from motorbridge import Controller, Mode

if TYPE_CHECKING or _can_available:
    import can

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


def decode_positions(config: YamArmConfig, states: dict[str, Any]) -> np.ndarray:
    """Convert motorbridge radians to dataset radians, and gripper 0=closed / 1=open."""
    closed, opened = config.gripper_closed_rad, config.gripper_open_rad
    if closed is None or opened is None:
        raise ValueError("Measure and configure gripper_closed_rad and gripper_open_rad before connecting")
    raw = np.asarray([states[name].pos for name in MOTOR_NAMES])
    if not np.isfinite(raw).all():
        raise ConnectionError("Non-finite YAM feedback")
    joints = raw[:6] * np.asarray(config.joint_signs) + np.asarray(config.joint_offsets_rad)
    gripper = (raw[6] - closed) / (opened - closed)
    if not -0.05 <= gripper <= 1.05:
        raise ValueError("Gripper feedback is outside the calibrated stroke; check endpoints")
    return np.r_[joints, np.clip(gripper, 0, 1)]


def encode_positions(config: YamArmConfig, positions: np.ndarray) -> np.ndarray:
    """Inverse of decode_positions, returning motorbridge radians."""
    closed, opened = config.gripper_closed_rad, config.gripper_open_rad
    if closed is None or opened is None:
        raise ValueError("Missing gripper calibration")
    raw = (positions[:6] - np.asarray(config.joint_offsets_rad)) / np.asarray(config.joint_signs)
    return np.r_[raw, closed + float(positions[6]) * (opened - closed)]


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
    """One motorbridge controller; the robot owns the servo thread and its lifetime."""

    def __init__(self, config: YamArmConfig) -> None:
        self.config = config
        self.bus: Controller | None = None
        self.monitor: can.BusABC | None = None
        self.motors: dict[str, Any] = {}
        self.position = np.zeros(7)
        self.target = np.zeros(7)
        self.command = np.zeros(7)
        self.updated_at = 0.0
        self.commanded_at = 0.0
        self.command_timed_out = False
        self.gravity: GravityCompensation | None = None
        self.enabled = False
        self.last_feedback: dict[int, float] = {}

    def connect(self) -> None:
        # motorbridge 0.5 exposes cached states but not their timestamps. A passive
        # SocketCAN subscriber checks receipt times; all motor I/O stays in motorbridge.
        self.monitor = can.Bus(
            channel=self.config.port,
            interface="socketcan",
            can_filters=[{"can_id": i + 17, "can_mask": 0x7FF, "extended": False} for i in range(7)],
        )
        self.bus = Controller(channel=self.config.port)
        self.motors = {
            name: self.bus.add_damiao_motor(i + 1, i + 17, "4340" if i < 3 else "4310")
            for i, name in enumerate(MOTOR_NAMES)
        }
        self.last_feedback.clear()

    def read(self, timeout: float, *, wait: bool = False) -> dict[str, Any]:
        assert self.bus is not None and self.monitor is not None
        for motor in self.motors.values():
            motor.request_feedback()
        deadline = time.monotonic() + timeout if wait else time.monotonic()
        while True:
            # Bound draining in case unrelated traffic saturates the interface.
            for _ in range(256):
                msg = self.monitor.recv(timeout=0)
                if msg is None:
                    break
                if (
                    msg.is_error_frame
                    or msg.is_remote_frame
                    or msg.is_extended_id
                    or msg.dlc != 8
                    or msg.arbitration_id not in range(17, 24)
                ):
                    continue
                if msg.data[0] & 0x0F != msg.arbitration_id - 16:
                    continue
                status = msg.data[0] >> 4
                if status not in (0, 1):
                    raise ConnectionError(
                        f"{self.config.port}: motor {msg.arbitration_id - 16} fault {status:#x}"
                    )
                self.last_feedback[msg.arbitration_id] = msg.timestamp
            self.bus.poll_feedback_once()
            states = {name: motor.get_state() for name, motor in self.motors.items()}
            now = time.time()  # python-can SocketCAN timestamps use the kernel's wall clock.
            fresh = all(0 <= now - self.last_feedback.get(i + 17, 0) <= timeout for i in range(7))
            if fresh and all(state is not None for state in states.values()):
                for name, state in states.items():
                    if state.status_code not in (0, 1) or not math.isfinite(state.pos):
                        raise ConnectionError(f"{self.config.port}: invalid {name} feedback")
                return states
            if time.monotonic() >= deadline:
                ages = ", ".join(
                    f"{i}: {(now - self.last_feedback[i + 16]) * 1000:.1f} ms"
                    if i + 16 in self.last_feedback
                    else f"{i}: missing"
                    for i in range(1, 8)
                )
                raise ConnectionError(f"{self.config.port}: missing or stale motor feedback ({ages})")
            time.sleep(0.001)

    def configure(self) -> None:
        assert self.bus is not None
        self.bus.disable_all()
        for motor in self.motors.values():
            motor.ensure_mode(Mode.MIT)

    def enable(self) -> None:
        assert self.bus is not None
        # Seed the measured pose with zero gains before enabling; never send zero angles.
        for name, value in zip(MOTOR_NAMES, encode_positions(self.config, self.position), strict=True):
            self.motors[name].send_mit(float(value), 0.0, 0.0, 0.0, 0.0)
        self.enabled = True  # Partial enable also requires disable during cleanup.
        self.bus.enable_all()

    def close(self) -> None:
        try:
            if self.bus is not None and self.enabled:
                self.bus.disable_all()
        finally:
            self.enabled = False
            for motor in self.motors.values():
                motor.close()
            self.motors.clear()
            if self.bus is not None:
                self.bus.close()
                self.bus = None
            if self.monitor is not None:
                self.monitor.shutdown()
                self.monitor = None

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
        self.command[:6] = np.clip(self.command[:6], *np.asarray(JOINT_LIMITS).T)
        raw_goal = encode_positions(cfg, self.command)
        raw_position = encode_positions(cfg, position)
        # Limit proportional closing/opening torque even on a blocked gripper.
        gripper_error_rad = cfg.gripper_torque_limit / cfg.gripper_kp
        raw_goal[6] = np.clip(
            raw_goal[6], raw_position[6] - gripper_error_rad, raw_position[6] + gripper_error_rad
        )
        gravity = np.zeros(6) if self.gravity is None else self.gravity.torque(position)
        gravity *= np.asarray(cfg.gravity_factors) * np.asarray(cfg.joint_signs)
        gravity = np.clip(gravity, -10.0, 10.0)
        kp, kd = [*cfg.kp, cfg.gripper_kp], [*cfg.kd, cfg.gripper_kd]
        return {
            name: (float(raw_goal[i]), 0.0, kp[i], kd[i], float(gravity[i]) if i < 6 else 0.0)
            for i, name in enumerate(MOTOR_NAMES)
        }
