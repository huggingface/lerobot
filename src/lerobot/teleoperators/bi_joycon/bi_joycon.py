#!/usr/bin/env python

import logging
import math
import time
from functools import cached_property
from typing import Dict, Sequence

import numpy as np

try:  # pragma: no cover - optional runtime dependency
    import hid
except ImportError:  # pragma: no cover - handled at runtime
    hid = None  # type: ignore[assignment]

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_bi_joycon import BiJoyconConfig

try:
    from pyjoycon import GyroTrackingJoyCon, get_L_id, get_R_id
except ImportError as exc:  # pragma: no cover - handled at runtime
    GyroTrackingJoyCon = None  # type: ignore[assignment]
    _JOYCON_IMPORT_ERROR = exc
else:  # pragma: no cover - import-time constant
    _JOYCON_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


def _ensure_joycon_available() -> None:
    if GyroTrackingJoyCon is None:
        raise ImportError(
            "pyjoycon is required for the JoyCon teleoperator. Install it with `pip install pyjoycon PyGLM hidapi`."
        ) from _JOYCON_IMPORT_ERROR


class BiJoycon(Teleoperator):
    """
    Bimanual SO-ARM controller driven by a pair of JoyCons.

    This teleoperator mirrors the bi-manual joint layout exposed by the SO-101 leader
    hardware while sourcing inputs from JoyCons. Stick, D-pad, and face button bindings
    match the `bi_gamepad` teleoperator, while the JoyCon IMUs provide wrist roll deltas.
    """

    config_class = BiJoyconConfig
    name = "bi_joycon"

    JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    JOINT_LIMITS = {
        "shoulder_pan": (-100.0, 100.0),
        "shoulder_lift": (-100.0, 100.0),
        "elbow_flex": (-100.0, 100.0),
        "wrist_flex": (-100.0, 100.0),
        "wrist_roll": (-100.0, 100.0),
        "gripper": (0.0, 100.0),
    }
    MAX_TIME_STEP = 0.25  # seconds

    def __init__(self, config: BiJoyconConfig):
        super().__init__(config)
        self.config = config
        self._left_joycon: GyroTrackingJoyCon | None = None
        self._right_joycon: GyroTrackingJoyCon | None = None
        self._joint_values: Dict[str, float] = self._init_joint_values()
        self._last_update: float | None = None
        self._stick_centers = [
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        ]
        self._orientation_maps: list[np.ndarray] = []
        self._neutral_quaternions: list[np.ndarray] = []
        self._prev_relative_quats: list[np.ndarray] = []
        self._button_prev: dict[str, bool] = {}
        self._stick_x_directions = self._expand_axis_directions(config.stick_x_directions)
        self._stick_y_directions = self._expand_axis_directions(config.stick_y_directions)

    @cached_property
    def action_features(self) -> Dict[str, type]:
        return {
            **{f"left_{joint}.pos": float for joint in self.JOINT_ORDER},
            **{f"right_{joint}.pos": float for joint in self.JOINT_ORDER},
        }

    @cached_property
    def feedback_features(self) -> Dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._left_joycon is not None and self._right_joycon is not None

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 - JoyCons self-calibrate via pyjoycon
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        _ensure_joycon_available()

        left_joycon: GyroTrackingJoyCon | None = None
        right_joycon: GyroTrackingJoyCon | None = None
        try:
            right_joycon = self._connect_joycon(get_R_id, "right", self.config.right_serial_hint)
            left_joycon = self._connect_joycon(get_L_id, "left", self.config.left_serial_hint)
            
        except Exception:
            # Clean up partially created connections.
            if left_joycon is not None:
                try:
                    left_joycon.disconnect()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
            if right_joycon is not None:
                try:
                    right_joycon.disconnect()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
            raise

        self._left_joycon = left_joycon
        self._right_joycon = right_joycon

        self._orientation_maps = self._build_orientation_maps()
        self._neutral_quaternions = [
            self._read_quaternion(self._left_joycon),
            self._read_quaternion(self._right_joycon),
        ]
        self._prev_relative_quats = [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ]

        # Give IMU threads a moment to stabilise, then capture stick centres.
        time.sleep(0.1)
        self._capture_stick_center(0)
        self._capture_stick_center(1)

        self._last_update = time.perf_counter()
        logger.info("JoyCon teleoperator connected for bi-manual teleoperation.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        # Recentring is handled interactively via stick press buttons.
        pass

    def configure(self) -> None:
        # No additional configuration is necessary for the JoyCon backend.
        pass

    def get_action(self) -> Dict[str, float]:
        if not self.is_connected or self._left_joycon is None or self._right_joycon is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        dt = self._compute_dt()
        self._update_inputs(dt)
        return dict(self._joint_values)

    def send_feedback(self, feedback: Dict[str, float]) -> None:  # noqa: ARG002 (unused)
        # No haptics or LEDs currently controlled through this interface.
        pass

    def disconnect(self) -> None:
        # if self._left_joycon is not None:
        #     try:
        #         self._left_joycon.disconnect()
        #     except Exception:  # pragma: no cover - hardware cleanup
        #         pass
        #     self._left_joycon = None

        # if self._right_joycon is not None:
        #     try:
        #         self._right_joycon.disconnect()
        #     except Exception:  # pragma: no cover - hardware cleanup
        #         pass
        #     self._right_joycon = None

        # self._last_update = None
        # logger.info("JoyCon teleoperator disconnected.")
        pass
    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _build_orientation_maps(self) -> list[np.ndarray]:
        left_map = self._to_orientation_map(self.config.orientation_map)
        right_map = self._to_orientation_map(self.config.right_orientation_map) if self.config.right_orientation_map else left_map.copy()
        return [left_map, right_map]

    def _compute_dt(self) -> float:
        now = time.perf_counter()
        if self._last_update is None:
            self._last_update = now
            return 0.0

        dt = now - self._last_update
        self._last_update = now
        return min(dt, self.MAX_TIME_STEP)

    def _update_inputs(self, dt: float) -> None:
        left = self._left_joycon
        right = self._right_joycon
        if left is None or right is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Stick-based joints (shoulder pan / lift)
        left_horizontal, left_vertical = self._read_stick(left, "stick_l", self._stick_centers[0])
        right_horizontal, right_vertical = self._read_stick(right, "stick_r", self._stick_centers[1])

        left_pan_delta = self._stick_x_directions[0] * left_horizontal * self.config.axis_speed * dt
        right_pan_delta = self._stick_x_directions[1] * right_horizontal * self.config.axis_speed * dt
        left_lift_delta = self._stick_y_directions[0] * left_vertical * self.config.axis_speed * dt
        right_lift_delta = self._stick_y_directions[1] * right_vertical * self.config.axis_speed * dt

        self._update_joint("left", "shoulder_pan", self._apply_joint_sensitivity("shoulder_pan", left_pan_delta))
        self._update_joint("right", "shoulder_pan", self._apply_joint_sensitivity("shoulder_pan", right_pan_delta))
        self._update_joint("left", "shoulder_lift", self._apply_joint_sensitivity("shoulder_lift", left_lift_delta))
        self._update_joint("right", "shoulder_lift", self._apply_joint_sensitivity("shoulder_lift", right_lift_delta))

        # Elbow flex mapping: D-pad up/down for left, B/X for right.
        hat_y = self._digital_axis(left, "up", "down")
        self._update_joint(
            "left",
            "elbow_flex",
            self._apply_joint_sensitivity("elbow_flex", -hat_y * self.config.button_speed * dt),
        )

        button_a = self._button_value(right, "a")
        button_b = self._button_value(right, "b")
        button_x = self._button_value(right, "x")
        button_y = self._button_value(right, "y")
        self._update_joint(
            "right",
            "elbow_flex",
            self._apply_joint_sensitivity("elbow_flex", (button_b - button_x) * self.config.button_speed * dt),
        )

        # Wrist flex mapping via side strap buttons.
        left_decrease, left_increase = self.config.left_wrist_flex_buttons
        right_decrease, right_increase = self.config.right_wrist_flex_buttons
        left_wrist_flex_delta = (self._button_value(left, left_increase) - self._button_value(left, left_decrease)) * self.config.button_speed * dt
        right_wrist_flex_delta = (self._button_value(right, right_increase) - self._button_value(right, right_decrease)) * self.config.button_speed * dt
        self._update_joint(
            "left",
            "wrist_flex",
            self._apply_joint_sensitivity("wrist_flex", left_wrist_flex_delta),
        )
        self._update_joint(
            "right",
            "wrist_flex",
            self._apply_joint_sensitivity("wrist_flex", right_wrist_flex_delta),
        )

        # Wrist roll from D-pad left/right and A/Y, plus gyro roll deltas.
        hat_x = self._digital_axis(left, "right", "left")
        self._update_joint(
            "left",
            "wrist_roll",
            self._apply_joint_sensitivity("wrist_roll", -hat_x * self.config.button_speed * dt),
        )

        self._update_joint(
            "right",
            "wrist_roll",
            self._apply_joint_sensitivity("wrist_roll", (button_a - button_y) * self.config.button_speed * dt),
        )

        if self.config.use_gyro_roll:
            left_roll_delta = self._gyro_roll_delta(left, 0)
            if left_roll_delta:
                self._update_joint(
                    "left",
                    "wrist_roll",
                    self._apply_joint_sensitivity("wrist_roll", left_roll_delta),
                )
            right_roll_delta = self._gyro_roll_delta(right, 1)
            if right_roll_delta:
                self._update_joint(
                    "right",
                    "wrist_roll",
                    self._apply_joint_sensitivity("wrist_roll", right_roll_delta),
                )

        # Gripper open / close using shoulder buttons.
        left_open_btn, left_close_btn = self.config.left_gripper_buttons
        right_open_btn, right_close_btn = self.config.right_gripper_buttons
        left_gripper_delta = (self._button_value(left, left_open_btn) - self._button_value(left, left_close_btn)) * self.config.gripper_speed * dt
        right_gripper_delta = (self._button_value(right, right_open_btn) - self._button_value(right, right_close_btn)) * self.config.gripper_speed * dt
        self._update_joint(
            "left",
            "gripper",
            self._apply_joint_sensitivity("gripper", left_gripper_delta),
        )
        self._update_joint(
            "right",
            "gripper",
            self._apply_joint_sensitivity("gripper", right_gripper_delta),
        )

        # Allow stick presses to recentre orientation and stick baselines.
        if self._button_edge("left_recenter", bool(getattr(left, "stick_l_btn", False))):
            self._neutral_quaternions[0] = self._read_quaternion(left)
            self._prev_relative_quats[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._capture_stick_center(0)
        if self._button_edge("right_recenter", bool(getattr(right, "stick_r_btn", False))):
            self._neutral_quaternions[1] = self._read_quaternion(right)
            self._prev_relative_quats[1] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self._capture_stick_center(1)

    def _connect_joycon(self, id_getter, label: str, serial_hint: str | None) -> GyroTrackingJoyCon:
        deadline = time.perf_counter() + max(0.0, self.config.discovery_timeout)
        poll_interval = max(0.1, self.config.discovery_poll_interval)
        joycon_id = (None, None, None)

        while time.perf_counter() <= deadline:
            joycon_id = id_getter()
            if joycon_id and joycon_id[0] is not None:
                break

            if serial_hint:
                joycon_id = self._find_joycon_by_serial(serial_hint, label)
                if joycon_id[0] is not None:
                    break

            time.sleep(poll_interval)

        if joycon_id[0] is None:
            detected = self._format_detected_joycons()
            hint_msg = f" with serial '{serial_hint}'" if serial_hint else ""
            if detected:
                msg_suffix = f" Detected JoyCons: {detected}. Ensure the desired controller is paired, not claimed by another process, and listed separately as Joy-Con (L) / Joy-Con (R) in your Bluetooth settings."
            else:
                msg_suffix = " Ensure it is paired, powered on, and connected via Bluetooth."
            raise RuntimeError(f"Could not find {label} JoyCon{hint_msg}.{msg_suffix}")

        joycon = GyroTrackingJoyCon(*joycon_id)
        joycon.calibrate(seconds=1.0)
        return joycon

    def _find_joycon_by_serial(self, serial_hint: str, label: str) -> tuple[int | None, int | None, str | None]:
        for device in self._enumerate_joycon_devices():
            serial = device.get("serial_number")
            if serial is None:
                continue
            if serial.lower() != serial_hint.lower():
                continue
            if not self._hid_matches_label(device, label):
                continue
            return device["vendor_id"], device["product_id"], serial
        return (None, None, None)

    def _format_detected_joycons(self) -> str:
        devices = []
        for device in self._enumerate_joycon_devices():
            product = device.get("product_string") or "Joy-Con"
            serial = device.get("serial_number") or "unknown"
            devices.append(f"{product} (serial {serial})")
        return ", ".join(devices)

    def _enumerate_joycon_devices(self) -> list[dict]:
        if hid is None:  # pragma: no cover - hidapi optional at runtime
            return []
        try:
            all_devices = hid.enumerate()  # type: ignore[no-untyped-call]
        except Exception:  # pragma: no cover - best effort diagnostics
            logger.debug("Failed to enumerate HID devices while searching for JoyCons.", exc_info=True)
            return []
        joycons: list[dict] = []
        for device in all_devices:
            if not isinstance(device, dict):
                continue
            if not self._is_joycon_device(device):
                continue
            joycons.append(device)
        return joycons

    @staticmethod
    def _is_joycon_device(device: dict) -> bool:
        product = (device.get("product_string") or "").lower()
        vendor = device.get("vendor_id")
        return vendor == 0x057E and "joy-con" in product

    @staticmethod
    def _hid_matches_label(device: dict, label: str) -> bool:
        product = (device.get("product_string") or "").lower()
        if label == "left":
            return "joy-con (l" in product or "joy-con (left" in product
        if label == "right":
            return "joy-con (r" in product or "joy-con (right" in product
        return True

    def _expand_axis_directions(self, dirs: Sequence[float]) -> tuple[float, float]:
        values = tuple(float(v) for v in dirs)
        if not values:
            return (1.0, 1.0)
        if len(values) == 1:
            return (values[0], values[0])
        return (values[0], values[1])

    def _init_joint_values(self) -> Dict[str, float]:
        joint_values: Dict[str, float] = {}
        for prefix in ("left", "right"):
            for joint in self.JOINT_ORDER:
                default = 0.0
                if joint == "gripper":
                    default = self._apply_joint_sensitivity("gripper", self.config.gripper_open_value)
                joint_values[f"{prefix}_{joint}.pos"] = default
        return joint_values

    def _apply_joint_sensitivity(self, joint: str, value: float) -> float:
        sensitivity = self.config.joint_sensitivity.get(joint, 1.0)
        return value * sensitivity

    def _update_joint(self, arm: str, joint: str, delta: float) -> None:
        if delta == 0.0:
            return
        key = f"{arm}_{joint}.pos"
        current = self._joint_values[key]
        self._joint_values[key] = self._clip_joint(joint, current + delta)

    def _set_joint(self, arm: str, joint: str, value: float) -> None:
        key = f"{arm}_{joint}.pos"
        self._joint_values[key] = self._clip_joint(joint, value)

    def _clip_joint(self, joint: str, value: float) -> float:
        lower, upper = self.JOINT_LIMITS[joint]
        return max(min(value, upper), lower)

    def _button_edge(self, key: str, pressed: bool) -> bool:
        prev = self._button_prev.get(key, False)
        self._button_prev[key] = pressed
        return pressed and not prev

    def _button_value(self, joycon: GyroTrackingJoyCon, attr: str) -> float:
        return 1.0 if bool(getattr(joycon, attr, False)) else 0.0

    def _digital_axis(self, joycon: GyroTrackingJoyCon, positive_attr: str, negative_attr: str) -> float:
        return self._button_value(joycon, positive_attr) - self._button_value(joycon, negative_attr)

    def _read_stick(self, joycon: GyroTrackingJoyCon, attr: str, center: np.ndarray) -> tuple[float, float]:
        axes = getattr(joycon, attr, None)
        if axes is None:
            return 0.0, 0.0
        values = np.asarray(axes, dtype=np.float32)
        if values.shape[0] < 2:
            return 0.0, 0.0
        horizontal = self._apply_deadzone(self._normalize_axis(values[0], center[0]))
        vertical = self._apply_deadzone(self._normalize_axis(values[1], center[1]))
        return float(horizontal), float(vertical)

    def _capture_stick_center(self, arm_idx: int) -> None:
        joycon: GyroTrackingJoyCon | None
        attr: str
        if arm_idx == 0:
            joycon = self._left_joycon
            attr = "stick_l"
        else:
            joycon = self._right_joycon
            attr = "stick_r"

        if joycon is None:
            return

        samples: list[np.ndarray] = []
        for _ in range(20):
            axes = getattr(joycon, attr, None)
            if axes is not None:
                values = np.asarray(axes, dtype=np.float32)
                if values.shape[0] >= 2:
                    samples.append(values[:2])
            time.sleep(0.01)
        if samples:
            self._stick_centers[arm_idx] = np.mean(samples, axis=0)
        else:
            self._stick_centers[arm_idx] = np.zeros(2, dtype=np.float32)

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.config.deadzone else value

    @staticmethod
    def _normalize_axis(value: float, center: float) -> float:
        norm = (value - center) / 2048.0
        return float(np.clip(norm, -1.0, 1.0))

    @staticmethod
    def _read_quaternion(joycon: GyroTrackingJoyCon) -> np.ndarray:
        quat = joycon.direction_Q
        return np.array([float(quat.w), float(quat.x), float(quat.y), float(quat.z)], dtype=np.float32)

    def _gyro_roll_delta(self, joycon: GyroTrackingJoyCon, arm_idx: int) -> float:
        rotvec = self._update_orientation(joycon, arm_idx)
        axis_idx = self.config.gyro_roll_axis_index
        if axis_idx < 0 or axis_idx >= rotvec.shape[0]:
            return 0.0
        roll = float(rotvec[axis_idx])
        if abs(roll) < self.config.gyro_deadzone:
            return 0.0
        return roll * self.config.gyro_roll_gain

    def _update_orientation(self, joycon: GyroTrackingJoyCon, arm_idx: int) -> np.ndarray:
        quat = self._read_quaternion(joycon)
        neutral = self._neutral_quaternions[arm_idx]
        relative = self._quat_multiply(self._quat_conjugate(neutral), quat)
        norm = float(np.linalg.norm(relative))
        if norm == 0.0:
            relative = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            relative = (relative / norm).astype(np.float32)

        prev_relative = self._prev_relative_quats[arm_idx]
        delta_quat = self._quat_multiply(self._quat_conjugate(prev_relative), relative)
        delta_vec_world = self._quat_to_rotvec(delta_quat)
        prev_matrix = self._quat_to_matrix(prev_relative)
        delta_vec = prev_matrix.T @ delta_vec_world
        self._prev_relative_quats[arm_idx] = relative
        orientation_map = self._orientation_maps[arm_idx]
        mapped = orientation_map @ delta_vec
        return mapped

    @staticmethod
    def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in quat]
        sin_half = math.sqrt(x * x + y * y + z * z)
        if sin_half < 1e-9:
            return np.zeros(3, dtype=np.float32)
        clamped_w = max(min(w, 1.0), -1.0)
        angle = 2.0 * math.atan2(sin_half, clamped_w)
        if angle > math.pi:
            angle -= 2.0 * math.pi
        axis = np.array([x, y, z], dtype=np.float32) / sin_half
        return axis * angle

    @staticmethod
    def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in quat]
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        return np.array(
            [
                [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _to_orientation_map(matrix: Sequence[Sequence[float]] | None) -> np.ndarray:
        if matrix is None:
            return np.eye(3, dtype=np.float32)
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.shape != (3, 3):
            raise ValueError("Orientation map must be a 3x3 matrix.")
        return arr
