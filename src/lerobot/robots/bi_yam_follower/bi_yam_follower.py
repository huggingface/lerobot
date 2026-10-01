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

"""Bimanual YAM v1 with motorbridge and a gravity-compensated impedance loop."""

import logging
import threading
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorCalibration
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..robot import Robot
from .config_bi_yam_follower import JOINT_LIMITS, MOTOR_NAMES, YAM_FEATURE_NAMES, BiYamFollowerConfig
from .yam_arm import GravityCompensation, YamArm, decode_positions, validate_target, verify_adapter

logger = logging.getLogger(__name__)


class BiYamFollower(Robot):
    """Left joints 0..5 + gripper, then right: radians and 0=closed / 1=open."""

    config_class = BiYamFollowerConfig
    name = "bi_yam_follower"

    def __init__(self, config: BiYamFollowerConfig) -> None:
        require_package("motorbridge", extra="yam")
        require_package("python-can", extra="yam")
        super().__init__(config)
        self.config = config
        self.arms = {"left": YamArm(config.left_arm), "right": YamArm(config.right_arm)}
        self._apply_gripper_calibration()
        self.cameras = make_cameras_from_configs(config.cameras)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: Exception | None = None
        self._connected = False
        self._calibration_session = False

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(YAM_FEATURE_NAMES, float)

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self.action_features,
            **{name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()},
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # lerobot-calibrate opens a strictly feedback-only session, even if a
        # calibration file exists and the user passed read_only=false.
        self._calibration_session = not calibrate
        self._failure = None
        self._stop.clear()
        if calibrate and not self.is_calibrated:
            raise ValueError("Run lerobot-calibrate with this robot.id to measure both gripper endpoints")
        try:
            for arm in self.arms.values():
                verify_adapter(arm.config)
            if calibrate:
                for camera in self.cameras.values():
                    camera.connect()
            for side, arm in self.arms.items():
                arm.connect()
                states = arm.read(self.config.feedback_timeout_s, wait=True)
                if not calibrate:
                    continue
                arm.position = decode_positions(arm.config, states)
                validate_target(arm.position, feedback=True)
                cfg = arm.config
                if not self.config.read_only:
                    if cfg.initial_position_rad is not None and np.any(
                        np.abs(arm.position[:6] - cfg.initial_position_rad) > cfg.initial_tolerance_rad
                    ):
                        raise ValueError(f"{side} arm is outside the configured initial pose tolerance")
                    if (
                        cfg.initial_gripper_position is not None
                        and abs(arm.position[6] - cfg.initial_gripper_position)
                        > cfg.initial_gripper_tolerance
                    ):
                        raise ValueError(f"{side} gripper is outside the initial pose tolerance")
                    if cfg.gravity_compensation:
                        arm.gravity = GravityCompensation()
                arm.target = arm.position.copy()
                arm.command = arm.position.copy()
                arm.updated_at = arm.commanded_at = time.monotonic()
                arm.command_timed_out = False
            if calibrate:
                # Validate both arms and all cameras before enabling either arm.
                self.configure()
                self._thread = threading.Thread(target=self._run, name="yam-servo", daemon=True)
                self._thread.start()
            self._connected = True
        except BaseException:
            self._close()
            raise

    def configure(self) -> None:
        if not self.config.read_only and not self._calibration_session:
            for arm in self.arms.values():
                arm.configure()
            for arm in self.arms.values():
                arm.position = decode_positions(
                    arm.config, arm.read(self.config.feedback_timeout_s, wait=True)
                )
                validate_target(arm.position, feedback=True)
                arm.target = arm.position.copy()
                arm.command = arm.position.copy()
                arm.updated_at = arm.commanded_at = time.monotonic()
                arm.enable()

    def _run(self) -> None:
        previous = time.monotonic()
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                for arm in self.arms.values():
                    states = arm.read(self.config.feedback_timeout_s)
                    position = decode_positions(arm.config, states)
                    validate_target(position, feedback=True)
                    with self._lock:
                        arm.position = position
                        arm.updated_at = time.monotonic()
                        if (
                            started - arm.commanded_at > self.config.command_timeout_s
                            and not arm.command_timed_out
                        ):
                            arm.target = position.copy()
                            arm.command = position.copy()
                            arm.command_timed_out = True
                        packet = (
                            arm.command_packet(position, min(started - previous, 0.05)) if arm.enabled else {}
                        )
                    for name, command in packet.items():
                        if self._stop.is_set():
                            break
                        arm.motors[name].send_mit(*command)
                previous = started
                self._stop.wait(max(0, 1 / self.config.control_frequency - (time.monotonic() - started)))
        except Exception as exc:
            self._failure = exc
            self._stop.set()
        finally:
            for arm in self.arms.values():
                if arm.enabled and arm.bus is not None:
                    try:
                        arm.bus.disable_all()
                        arm.enabled = False
                    except Exception:
                        logger.exception("Could not disable YAM torque; use the hardware e-stop")

    def _check_feedback(self) -> None:
        if self._failure is not None or self._stop.is_set():
            raise ConnectionError("YAM servo stopped after a motor/feedback error") from self._failure
        if any(
            time.monotonic() - arm.updated_at > self.config.feedback_timeout_s for arm in self.arms.values()
        ):
            self._stop.set()
            raise ConnectionError("YAM servo feedback is stale; reconnect before commanding motion")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self._calibration_session:
            raise RuntimeError("Reconnect after calibration before reading policy observations")
        with self._lock:
            self._check_feedback()
            result: dict[str, Any] = {
                f"{side}_{name}.pos": float(arm.position[i])
                for side, arm in self.arms.items()
                for i, name in enumerate(MOTOR_NAMES)
            }
        for name, camera in self.cameras.items():
            result[name] = camera.read_latest(max_age_ms=200)
        return result

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.read_only or self._calibration_session:
            raise RuntimeError("YAM read-only/calibration connection forbids motor commands")
        if set(action) != set(YAM_FEATURE_NAMES):
            raise ValueError("YAM requires all 14 absolute joint/gripper targets; Cartesian actions need IK")
        targets = {
            side: np.asarray([action[f"{side}_{name}.pos"] for name in MOTOR_NAMES], dtype=float)
            for side in self.arms
        }
        for values in targets.values():
            validate_target(values, feedback=True)
            # Encoder quantization can place a captured home pose just outside a bound.
            # Accept only the same small feedback tolerance, then clamp the actual target.
            values[:6] = np.clip(values[:6], *np.asarray(JOINT_LIMITS).T)
        with self._lock:
            self._check_feedback()
            for side, arm in self.arms.items():
                arm.target = targets[side]
                arm.commanded_at = time.monotonic()
                arm.command_timed_out = False
        return {
            f"{side}_{name}.pos": float(targets[side][i])
            for side in self.arms
            for i, name in enumerate(MOTOR_NAMES)
        }

    def _close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError("YAM servo did not stop; use the hardware e-stop")
            self._thread = None
        for arm in self.arms.values():
            try:
                arm.close()
            except Exception:
                logger.exception("Failed to close YAM arm")
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self._connected = False
        self._calibration_session = False

    @check_if_not_connected
    def disconnect(self) -> None:
        self._close()

    @property
    def is_calibrated(self) -> bool:
        return all(
            a.config.gripper_closed_rad is not None and a.config.gripper_open_rad is not None
            for a in self.arms.values()
        )

    def _apply_gripper_calibration(self, *, overwrite: bool = False) -> None:
        """Load endpoints from standard MotorCalibration records in native MIT encoder counts."""
        for side, arm in self.arms.items():
            calibration = self.calibration.get(f"{side}_gripper")
            if calibration is None:
                continue
            if not (
                calibration.id == 7
                and calibration.drive_mode in (0, 1)
                and calibration.homing_offset == 0
                and 0 <= calibration.range_min < calibration.range_max <= 65535
            ):
                raise ValueError(f"Invalid saved {side} gripper calibration")
            # DM4310 reports position as unsigned 16-bit counts spanning +/-12.5 rad.
            endpoints = np.asarray([calibration.range_min, calibration.range_max]) * (25.0 / 65535) - 12.5
            if calibration.drive_mode:
                endpoints = endpoints[::-1]
            if overwrite or arm.config.gripper_closed_rad is None:
                arm.config.gripper_closed_rad, arm.config.gripper_open_rad = map(float, endpoints)
                arm.config.__post_init__()

    def _save_calibration(self, fpath: Path | None = None) -> None:
        """Atomically save the standard calibration file without writing motor settings."""
        path = fpath if fpath is not None else self.calibration_fpath
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            super()._save_calibration(temporary_path)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    @check_if_not_connected
    def calibrate(self) -> None:
        """Measure both gripper stops by hand through lerobot-calibrate; never enable torque or reset zeros."""
        if not self._calibration_session:
            raise RuntimeError("Reconnect with calibrate=False before measuring gripper endpoints")
        measurements: dict[str, dict[str, float]] = {}
        print("Support both arms. Move only the grippers gently by hand; stop if either resists.")
        for endpoint in ("closed", "open"):
            input(f"Place BOTH grippers fully {endpoint}, release them, then press Enter: ")
            samples: dict[str, list[float]] = {side: [] for side in self.arms}
            for _ in range(10):
                for side, arm in self.arms.items():
                    state = arm.read(self.config.feedback_timeout_s, wait=True)["gripper"]
                    samples[side].append(state.pos)
                time.sleep(0.02)
            if any(not np.isfinite(values).all() or np.ptp(values) > 0.03 for values in samples.values()):
                raise ValueError("Grippers moved or returned invalid feedback; calibration was not saved")
            measurements[endpoint] = {side: float(np.median(values)) for side, values in samples.items()}
        calibration = {}
        for side in self.arms:
            closed, opened = measurements["closed"][side], measurements["open"][side]
            if not (abs(closed) <= 12.5 and abs(opened) <= 12.5 and 0.5 < abs(opened - closed) < 10):
                raise ValueError(f"Implausible {side} gripper stroke; calibration was not saved")
            counts = [round((value + 12.5) * 65535 / 25.0) for value in (closed, opened)]
            calibration[f"{side}_gripper"] = MotorCalibration(
                id=7,
                drive_mode=int(opened < closed),
                homing_offset=0,
                range_min=min(counts),
                range_max=max(counts),
            )
        previous = self.calibration
        self.calibration = calibration
        try:
            self._save_calibration()
        except Exception:
            self.calibration = previous
            raise
        self._apply_gripper_calibration(overwrite=True)
        print(f"Saved gripper endpoints to {self.calibration_fpath}. Joint zeros were not changed.")
