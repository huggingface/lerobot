# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Bimanual YAM: native MotorsBus transport and a policy-independent servo loop."""

import gc
import logging
import math
import threading
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.configs.policies import PreTrainedConfig
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.damiao import MotorState
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _mujoco_available, require_package

from ..robot import Robot
from .config_bi_yam_follower import (
    JOINT_LIMITS,
    MOTOR_NAMES,
    YAM_FEATURE_NAMES,
    BiYamFollowerConfig,
    YamArmConfig,
)

if TYPE_CHECKING or _mujoco_available:
    import mujoco

logger = logging.getLogger(__name__)


class _FeedbackTimeoutError(ConnectionError):
    """A freshness deadline, distinct from missing packets or hardware faults."""


class _ControlGC:
    """Keep the preloaded model heap out of cyclic scans while servo threads run.

    New objects still participate in normal collection. The process-wide freeze
    is shared by connected YAM instances and restored only after the last one
    disconnects. An existing caller-owned freeze is extended and left frozen;
    its owner must unfreeze it. Disabled collection is left alone.
    """

    _lock = threading.Lock()
    _users = 0
    _owns_freeze = False

    @classmethod
    def acquire(cls) -> None:
        with cls._lock:
            if cls._users == 0 and gc.isenabled():
                cls._owns_freeze = gc.get_freeze_count() == 0
                gc.collect()
                gc.freeze()
            cls._users += 1

    @classmethod
    def release(cls) -> None:
        with cls._lock:
            cls._users -= 1
            if cls._users == 0 and cls._owns_freeze:
                gc.unfreeze()
                cls._owns_freeze = False


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


class _Gravity:
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


class _Arm:
    def __init__(self, config: YamArmConfig) -> None:
        self.config = config
        self.bus = make_yam_bus(config)
        self.position = np.zeros(7)
        self.target = np.zeros(7)
        self.command = np.zeros(7)
        self.updated_at = 0.0
        self.commanded_at = 0.0
        self.command_timed_out = False
        self.gravity: _Gravity | None = None
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


class BiYamFollower(Robot):
    """Absolute joint control: left six joints + gripper, then right six + gripper.

    End-effector poses are not joint commands. Cartesian policies require an
    explicit, separately validated IK processor; this adapter does not guess.
    """

    config_class = BiYamFollowerConfig
    name = "bi_yam_follower"

    def __init__(self, config: BiYamFollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.arms = {"left": _Arm(config.left_arm), "right": _Arm(config.right_arm)}
        self._apply_gripper_calibration()
        self.cameras = make_cameras_from_configs(config.cameras)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._enable_requested = threading.Event()
        self._failure: Exception | None = None
        self._connected = False
        self._calibration_session = False
        self._gc_acquired = False
        self._control_started = False

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

    @property
    def is_control_enabled(self) -> bool:
        return all(arm.enabled for arm in self.arms.values())

    @property
    def has_started_control(self) -> bool:
        return self._control_started

    def _validate_initial_pose(self, side: str, arm: _Arm) -> None:
        if arm.config.initial_position_rad is not None and np.any(
            np.abs(arm.position[:6] - arm.config.initial_position_rad) > arm.config.initial_tolerance_rad
        ):
            raise ValueError(
                f"{side} arm is not in the configured initial pose; no automatic homing performed"
            )
        if (
            arm.config.initial_gripper_position is not None
            and abs(arm.position[6] - arm.config.initial_gripper_position)
            > arm.config.initial_gripper_tolerance
        ):
            raise ValueError(f"{side} gripper is not in the configured initial position")

    def _enable_arm(self, arm: _Arm, position: np.ndarray) -> None:
        """Called by the bus-owning worker, or at connect before workers exist."""
        if self._stop.is_set():
            raise ConnectionError("YAM feedback failed before torque enable") from self._failure
        with self._lock:
            arm.target = position.copy()
            arm.command = position.copy()
            arm.commanded_at = time.monotonic()
            arm.command_timed_out = False
        # Seed zero gains at the current measured pose, never an old startup target.
        arm.bus.sync_write_mit(
            {
                name: (0.0, 0.0, float(p), 0.0, 0.0)
                for name, p in zip(MOTOR_NAMES, encode_positions(arm.config, position), strict=True)
            }
        )
        arm.enabled = True  # Ensure failure cleanup also attempts to disable this arm.
        arm.bus.enable_torque()
        arm.control_ready.set()

    @check_if_not_connected
    def start_control(self) -> None:
        if self.config.read_only or self._calibration_session:
            raise RuntimeError("YAM read-only/calibration connection forbids torque enable")
        with self._lock:
            self._check_feedback()
            if self.is_control_enabled:
                return
            # The operator may have moved an unpowered arm since connect.
            for side, arm in self.arms.items():
                self._validate_initial_pose(side, arm)
        self._enable_requested.set()
        deadline = time.monotonic() + 2.0
        try:
            while not all(arm.control_ready.is_set() for arm in self.arms.values()):
                if self._stop.is_set() or self._failure is not None:
                    raise ConnectionError("YAM activation failed") from self._failure
                if time.monotonic() >= deadline:
                    raise TimeoutError("YAM motors did not enable before the activation deadline")
                self._stop.wait(0.005)
            with self._lock:
                self._check_feedback()
            self._control_started = True
        except Exception:
            self._stop.set()
            raise

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
                    state = arm.bus.sync_read_all_states(strict=True)["gripper"]
                    samples[side].append(math.radians(state["position"]))
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

    def configure(self) -> None:
        """No persistent motor writes: factory MIT mode, IDs, and zero positions are retained."""

    def validate_policy_config(self, policy: PreTrainedConfig) -> None:
        action = (policy.output_features or {}).get(ACTION)
        if action is None or tuple(action.shape) != (14,):
            raise ValueError("Bimanual YAM requires a 14-dimensional absolute-joint action")
        names = (policy.dataset_feature_names or {}).get(ACTION)
        if names is not None and list(names) != list(YAM_FEATURE_NAMES):
            raise ValueError(
                "Policy action order differs from left joints/gripper, then right joints/gripper"
            )
        if policy.type == "molmoact2":
            if getattr(policy, "control_mode", None) != "absolute joint pose":
                raise ValueError("YAM cannot execute end-effector actions without an explicit IK processor")
            state = (policy.input_features or {}).get(OBS_STATE)
            if state is None or tuple(state.shape) != (14,):
                raise ValueError("MolmoAct2 YAM state must contain 14 joint/gripper values")
            state_names = (policy.dataset_feature_names or {}).get(OBS_STATE)
            if state_names is not None and list(state_names) != list(YAM_FEATURE_NAMES):
                raise ValueError("MolmoAct2 YAM state joint order is incorrect")
            if getattr(policy, "normalize_gripper", None) is not False:
                raise ValueError("MolmoAct2 YAM requires unnormalized continuous grippers")
            expected = [f"observation.images.{name}" for name in ("top", "left", "right")]
            if list(getattr(policy, "image_keys", [])) != expected:
                raise ValueError("MolmoAct2 YAM camera order must be top, left, right")
            if not {"top", "left", "right"}.issubset(self.cameras):
                raise ValueError("Configure all three YAM cameras before rollout")

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not calibrate:
            # lerobot-calibrate connects first with calibrate=False. Open only
            # raw feedback here, even if read_only=False or a prior file exists.
            try:
                for arm in self.arms.values():
                    verify_adapter(arm.config)
                for arm in self.arms.values():
                    arm.bus.connect(handshake=False)
                    arm.bus.sync_read_all_states(strict=True)
                self._calibration_session = True
                self._connected = True
                return
            except Exception:
                self._close()
                raise
        if not self.is_calibrated:
            raise ValueError(
                "Run lerobot-calibrate with the same robot.id and CAN ports to measure both YAM grippers"
            )
        self._calibration_session = False
        self._stop.clear()
        self._enable_requested.clear()
        self._failure = None
        self._control_started = False
        try:
            # Full collections of a loaded VLM can hold the GIL past the motor
            # feedback deadline. Prepare GC before any hardware or worker starts.
            _ControlGC.acquire()
            self._gc_acquired = True
            for arm in self.arms.values():
                verify_adapter(arm.config)
            self._connect_cameras()
            # Validate BOTH arms before enabling either. handshake=True enables
            # torque in DamiaoMotorsBus, so use the strictly read-only refresh.
            for side, arm in self.arms.items():
                arm.bus.connect(handshake=False)
                arm.position = decode_positions(arm.config, arm.bus.sync_read_all_states(strict=True))
                validate_target(arm.position, feedback=True)
                if not self.config.read_only:
                    self._validate_initial_pose(side, arm)
                arm.target = arm.position.copy()
                arm.command = arm.position.copy()
                arm.updated_at = arm.commanded_at = time.monotonic()
                if arm.config.gravity_compensation and not self.config.read_only:
                    arm.gravity = _Gravity()
            self.configure()
            for side, arm in self.arms.items():
                if self._stop.is_set():
                    raise ConnectionError("YAM feedback failed during startup") from self._failure
                arm.ready.clear()
                arm.control_ready.clear()
                if not self.config.read_only and not self.config.defer_torque_enable:
                    self._enable_arm(arm, arm.position)
                arm.thread = threading.Thread(target=self._run, args=(arm,), name=f"yam-{side}", daemon=True)
                arm.thread.start()
                if not arm.ready.wait(timeout=self.config.feedback_timeout_s) or self._failure is not None:
                    raise ConnectionError(f"{side} YAM servo failed to start") from self._failure
            self._connected = True
            self._check_feedback()
            self._control_started = self.is_control_enabled
        except Exception:
            self._close()
            raise

    def _connect_cameras(self) -> None:
        """Retry a first-frame timeout once, before opening either motor bus."""
        for name in self.cameras:
            for attempt in range(2):
                camera = self.cameras[name]
                try:
                    camera.connect()
                    break
                except TimeoutError:
                    if camera.is_connected:
                        camera.disconnect()
                    if attempt:
                        raise
                    logger.warning(
                        "Camera %s produced no initial frame; reopening once before motor startup", name
                    )
                    # Use a fresh object so a delayed reader from the failed
                    # connection cannot publish into the replacement's buffer.
                    self.cameras[name] = make_cameras_from_configs({name: self.config.cameras[name]})[name]
                    time.sleep(0.5)

    def _run(self, arm: _Arm) -> None:
        previous = time.monotonic()
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                states = arm.bus.sync_read_all_states(strict=True)
                if time.monotonic() - started > self.config.feedback_timeout_s:
                    raise _FeedbackTimeoutError(
                        f"{arm.config.port}: YAM feedback exceeded freshness deadline"
                    )
                position = decode_positions(arm.config, states)
                validate_target(position, feedback=True)
                if self._enable_requested.is_set() and not arm.enabled:
                    self._enable_arm(arm, position)
                with self._lock:
                    arm.position = position
                    arm.updated_at = time.monotonic()
                    arm.ready.set()
                    if (
                        started - arm.commanded_at > self.config.command_timeout_s
                        and not arm.command_timed_out
                    ):
                        # Capture once on expiry. Repeatedly following measured
                        # sag removes position stiffness while the policy is idle.
                        arm.target = position.copy()
                        arm.command = position.copy()
                        arm.command_timed_out = True
                    packet = (
                        arm.command_packet(position, min(started - previous, 0.05)) if arm.enabled else None
                    )
                if packet is not None and not self._stop.is_set():
                    arm.bus.sync_write_mit(packet)
                previous = started
                self._stop.wait(max(0, 1 / self.config.control_frequency - (time.monotonic() - started)))
        except Exception as exc:
            with self._lock:
                # Never replace a motor/disable fault with a recoverable timeout
                # from the other arm as the workers stop concurrently.
                if self._failure is None or isinstance(self._failure, _FeedbackTimeoutError):
                    self._failure = exc
            self._stop.set()
        finally:
            if arm.enabled:
                try:
                    arm.bus.disable_torque()
                except Exception as exc:
                    with self._lock:
                        self._failure = exc
                    logger.exception("Could not disable YAM torque; use the hardware e-stop")
                arm.enabled = False

    def _check_feedback(self) -> None:
        if self._failure is not None:
            raise ConnectionError("YAM servo loop stopped after a motor/feedback error") from self._failure
        now = time.monotonic()
        stale = [
            f"{arm.config.port}: {(now - arm.updated_at) * 1000:.1f} ms"
            for arm in self.arms.values()
            if now - arm.updated_at > self.config.feedback_timeout_s
        ]
        if stale:
            self._stop.set()
            self._failure = _FeedbackTimeoutError(
                f"YAM feedback is stale ({', '.join(stale)}; "
                f"deadline {self.config.feedback_timeout_s * 1000:.1f} ms)"
            )
            raise self._failure

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self._calibration_session:
            raise RuntimeError("Reconnect after calibration before reading policy observations")
        with self._lock:
            self._check_feedback()
            result = {
                f"{side}_{name}.pos": float(arm.position[i])
                for side, arm in self.arms.items()
                for i, name in enumerate(MOTOR_NAMES)
            }
        for name, camera in self.cameras.items():
            result[name] = camera.async_read()
        return result

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self._calibration_session:
            raise RuntimeError("Motor commands are forbidden during calibration")
        if self.config.read_only:
            raise RuntimeError("YAM read_only=true forbids motor commands")
        if self.config.defer_torque_enable and not self.is_control_enabled:
            raise RuntimeError("YAM torque is disabled; use /start before sending motor commands")
        if set(action) != set(YAM_FEATURE_NAMES):
            raise ValueError(
                "YAM requires the complete 14 absolute-joint/gripper action; Cartesian actions need IK"
            )
        targets = {
            side: np.asarray([action[f"{side}_{name}.pos"] for name in MOTOR_NAMES], dtype=float)
            for side in self.arms
        }
        for target in targets.values():
            validate_target(target)
        with self._lock:
            self._check_feedback()
            now = time.monotonic()
            for side, arm in self.arms.items():
                arm.target = targets[side]
                arm.commanded_at = now
                arm.command_timed_out = False
        return dict(action)

    def _recover_feedback_for_return(self) -> None:
        """One attempt, with torque off until both buses have 0.5 s of healthy reads."""
        self._stop.set()
        self._enable_requested.clear()
        for arm in self.arms.values():
            if arm.thread is not None:
                arm.thread.join(timeout=2)
                if arm.thread.is_alive():
                    raise RuntimeError("YAM worker did not stop; cannot recover for home")
        # Check after joins: disable failures or late bus faults prohibit recovery.
        if not isinstance(self._failure, _FeedbackTimeoutError):
            raise ConnectionError(
                "YAM return recovery requires a freshness timeout, not a motor fault"
            ) from self._failure
        if any(arm.enabled for arm in self.arms.values()):
            raise RuntimeError("YAM torque is still enabled; cannot recover for home")
        logger.warning("YAM return-only recovery: checking both buses with torque disabled")
        self._failure = None
        self._stop.clear()
        try:
            for side, arm in self.arms.items():
                arm.ready.clear()
                arm.control_ready.clear()
                arm.thread = threading.Thread(target=self._run, args=(arm,), name=f"yam-{side}", daemon=True)
                arm.thread.start()
            for arm in self.arms.values():
                if not arm.ready.wait(timeout=self.config.feedback_timeout_s):
                    raise ConnectionError(
                        "YAM return recovery did not receive fresh feedback"
                    ) from self._failure
            deadline = time.monotonic() + 0.5
            while True:
                with self._lock:
                    self._check_feedback()
                if self._stop.is_set():
                    raise ConnectionError("YAM return recovery stopped") from self._failure
                if time.monotonic() >= deadline:
                    break
                self._stop.wait(0.005)
            # Each bus owner seeds its command from a new strict read before enabling.
            # Do not apply startup-pose assertions: this is a return from the current pose.
            self._enable_requested.set()
            for arm in self.arms.values():
                if not arm.control_ready.wait(timeout=2):
                    raise ConnectionError("YAM return recovery could not enable control") from self._failure
            with self._lock:
                self._check_feedback()
        except Exception:
            self._stop.set()
            raise

    @check_if_not_connected
    def return_to_position(self, position: RobotAction) -> bool:
        """Return using fresh motor feedback, independently of cameras or inference."""
        if self.config.read_only or self._calibration_session or not self.has_started_control:
            raise RuntimeError("YAM return requires previously activated motor control")
        # Validate the complete target before any possible reactivation.
        if set(position) != set(YAM_FEATURE_NAMES):
            raise ValueError("YAM return requires all 14 joint/gripper positions")
        # A captured observation may sit just beyond a joint bound because
        # feedback permits encoder/zero noise. Never send that as an out-of-range
        # command: validate with the feedback tolerance, then project onto the
        # unchanged command limits. Normal policy actions remain strictly checked.
        target = dict(position)
        for side in self.arms:
            values = np.asarray([position[f"{side}_{name}.pos"] for name in MOTOR_NAMES])
            validate_target(values, feedback=True)
            for i, (lower, upper) in enumerate(JOINT_LIMITS):
                target[f"{side}_{MOTOR_NAMES[i]}.pos"] = float(np.clip(values[i], lower, upper))
        try:
            with self._lock:
                self._check_feedback()
        except ConnectionError:
            if not self.config.recover_on_feedback_timeout:
                raise
            self._recover_feedback_for_return()
        self.wait_until_reached(target)
        return True

    @check_if_not_connected
    def wait_until_reached(self, position: RobotAction) -> None:
        """Maintain the return target until both arms settle, or fail within a bounded time.

        Use motor feedback directly: camera availability must not determine
        whether a completed return is reported. Servo speed and torque limits
        remain active throughout this phase.
        """
        self.send_action(position)
        deadline = time.monotonic() + self.config.return_timeout_s
        settled_since = None
        try:
            while True:
                self.send_action(position)
                with self._lock:
                    self._check_feedback()
                    reached = all(
                        np.max(np.abs(arm.position[:6] - arm.target[:6])) <= 0.03
                        and abs(arm.position[6] - arm.target[6]) <= 0.05
                        for arm in self.arms.values()
                    )
                now = time.monotonic()
                settled_since = (now if settled_since is None else settled_since) if reached else None
                if settled_since is not None and now - settled_since >= 0.2:
                    return
                if now >= deadline:
                    raise TimeoutError(
                        "YAM return did not reach the startup pose within its settling deadline"
                    )
                self._stop.wait(min(0.02, self.config.command_timeout_s / 3))
        except Exception:
            # A failed reset must not keep pursuing its old target while the
            # interactive session reports failure and waits for the operator.
            with self._lock:
                for arm in self.arms.values():
                    arm.target = arm.position.copy()
                    arm.command = arm.position.copy()
                    arm.commanded_at = time.monotonic()
            raise

    def _close(self) -> None:
        self._stop.set()
        for arm in self.arms.values():
            if arm.thread is not None:
                arm.thread.join(timeout=2)
                if arm.thread.is_alive():
                    raise RuntimeError("YAM servo did not stop; use the hardware e-stop")
        for arm in self.arms.values():
            if arm.bus.is_connected:
                try:
                    arm.bus.disconnect(disable_torque=arm.enabled)
                except Exception:
                    logger.exception("Failed to close YAM CAN bus")
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self._connected = False
        self._calibration_session = False
        if self._gc_acquired:
            _ControlGC.release()
            self._gc_acquired = False

    @check_if_not_connected
    def disconnect(self) -> None:
        self._close()
