# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""NexArm motor bus driver.

NexArm uses a custom CommProtocol over USB serial at 1 Mbps.
Frame format: [0xFF][0xFF][ID][LEN][CMD][ARGS...][CHECKSUM]

The leader (master) ESP32 directly drives HX-30HM servos.
The follower (slave) ESP32 forwards commands through an AT32F421 co-processor
to the same servos.

The bus only exposes Present_Position / Goal_Position on the standard
``MotorsBusBase`` interface; it does not use Dynamixel/Feetech SDK port handlers
because the wire protocol is custom. Calibration (homing_offset / range_min /
range_max) and normalisation (DEGREES, RANGE_0_100, RANGE_M100_100) are
implemented directly so that follower/leader code matches the official
SO-100/SO-101 flow.
"""

from __future__ import annotations

import contextlib
import logging
import struct
import threading
import time
from collections.abc import Sequence
from pprint import pformat
from typing import TYPE_CHECKING

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _serial_available, require_package
from lerobot.utils.utils import enter_pressed, move_cursor_up

if TYPE_CHECKING or _serial_available:
    import serial
else:
    serial = None  # type: ignore[assignment]

from ..motors_bus import (
    Motor,
    MotorCalibration,
    MotorNormMode,
    MotorsBusBase,
    NameOrID,
    Value,
)

logger = logging.getLogger(__name__)

SYSTEM_ID = 0xFF

CMD_LEROBOT_MODE = 68
CMD_READ_POS = 96
CMD_WRITE_POS = 97
CMD_TORQUE = 98
CMD_SET_MOTION_PARAMS = 56

JOINT_COUNT = 6
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

MODEL = "hx30hm"
MODEL_RESOLUTION = 4096
POSITION_MIN = 0
POSITION_MAX = MODEL_RESOLUTION - 1
HALF_TURN = MODEL_RESOLUTION // 2

DEFAULT_BAUDRATE = 1_000_000
DEFAULT_TIMEOUT = 0.05
REPLY_TIMEOUT = 0.15


def build_frame(device_id: int, cmd: int, args: bytes = b"") -> bytes:
    length = len(args) + 2
    data_raw = bytes([device_id & 0xFF, length & 0xFF, cmd & 0xFF]) + args
    checksum = (~sum(data_raw)) & 0xFF
    return b"\xff\xff" + data_raw + bytes([checksum])


def parse_frame(data: bytes) -> tuple[int, int, bytes] | None:
    """Parse a CommProtocol frame. Returns (id, cmd, args) or None.

    Accepts both the correct full-range checksum and the short checksum from
    the leader firmware (``tx_packet_complete`` uses ``rx_packet.elements.length``
    instead of ``tx_packet.elements.length``).
    """
    if len(data) < 6 or data[0] != 0xFF or data[1] != 0xFF:
        return None
    device_id = data[2]
    length = data[3]
    total = 4 + length
    if len(data) < total:
        return None
    cmd = data[4]
    n = length - 2
    args = data[5 : 5 + n]
    checksum_byte = data[total - 1]
    expected_full = (~sum(data[2 : total - 1])) & 0xFF
    expected_short = (~sum(data[2:5])) & 0xFF
    if checksum_byte != expected_full and checksum_byte != expected_short:
        return None
    return (device_id, cmd, bytes(args))


class NexArmMotorsBus(MotorsBusBase):
    """Bus driver for NexArm 6-DOF arms (leader or follower).

    Implements the subset of the official ``MotorsBusBase`` API needed for
    teleoperation and policy inference: ``sync_read("Present_Position")``,
    ``sync_write("Goal_Position", ...)``, torque enable/disable, calibration
    with ``set_half_turn_homings`` and ``record_ranges_of_motion``.

    Unlike Dynamixel/Feetech buses, NexArm has no per-register access; the
    embedded firmware exposes only a small command set. ``read``/``write`` and
    ``sync_read``/``sync_write`` therefore only accept ``"Present_Position"``
    and ``"Goal_Position"`` data names.
    """

    model_resolution_table = {MODEL: MODEL_RESOLUTION}
    apply_drive_mode = True

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor] | None = None,
        calibration: dict[str, MotorCalibration] | None = None,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        require_package("pyserial", extra="nexarm", import_name="serial")
        if motors is None:
            motors = self.default_motors()
        super().__init__(port, motors, calibration)
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial: serial.Serial | None = None
        self._lock = threading.Lock()
        self._id_to_name_dict = {m.id: name for name, m in self.motors.items()}
        self._name_to_id_dict = {name: m.id for name, m in self.motors.items()}
        self._ordered_names: list[str] = sorted(self.motors, key=lambda n: self.motors[n].id)
        self._validate_motors()

    @staticmethod
    def default_motors() -> dict[str, Motor]:
        norm = MotorNormMode.RANGE_M100_100
        gripper_norm = MotorNormMode.RANGE_0_100
        return {
            "shoulder_pan": Motor(id=1, model=MODEL, norm_mode=norm),
            "shoulder_lift": Motor(id=2, model=MODEL, norm_mode=norm),
            "elbow_flex": Motor(id=3, model=MODEL, norm_mode=norm),
            "wrist_flex": Motor(id=4, model=MODEL, norm_mode=norm),
            "wrist_roll": Motor(id=5, model=MODEL, norm_mode=norm),
            "gripper": Motor(id=6, model=MODEL, norm_mode=gripper_norm),
        }

    def _validate_motors(self) -> None:
        ids = [m.id for m in self.motors.values()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Some motors have the same id: {ids}")
        for name in self._ordered_names:
            if self.motors[name].model != MODEL:
                raise ValueError(
                    f"NexArm only supports model '{MODEL}', got '{self.motors[name].model}' for {name!r}"
                )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    Port: '{self.port}',\n"
            f"    Motors: \n{pformat(self.motors, indent=8, sort_dicts=False)},\n"
            ")',\n"
        )

    @property
    def is_connected(self) -> bool:
        return self._serial is not None and self._serial.is_open

    @property
    def is_calibrated(self) -> bool:
        if not self.calibration:
            return False
        for name in self.motors:
            c = self.calibration.get(name)
            if c is None or c.range_min == c.range_max:
                return False
        return True

    def connect(self, handshake: bool = True) -> None:
        if self.is_connected:
            return
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
        except (FileNotFoundError, OSError, serial.SerialException) as e:
            raise ConnectionError(
                f"Could not connect on port '{self.port}'. "
                "Make sure you are using the correct port (try `lerobot-find-port`)."
            ) from e
        time.sleep(0.1)
        self._serial.reset_input_buffer()
        logger.info(f"NexArmMotorsBus connected on {self.port}")
        if handshake:
            self._handshake()

    def _handshake(self) -> None:
        try:
            self._raw_read_positions()
        except TimeoutError as e:
            raise ConnectionError(
                f"NexArm handshake failed on '{self.port}': no position reply. "
                "Check that the arm is powered and in LeRobot bridge mode."
            ) from e

    def handshake(self) -> None:
        self._handshake()

    def disconnect(self, disable_torque: bool = True) -> None:
        if self._serial is None:
            return
        if disable_torque:
            with contextlib.suppress(Exception):
                self.disable_torque()
        with contextlib.suppress(Exception):
            self._serial.close()
        self._serial = None
        logger.info(f"NexArmMotorsBus disconnected from {self.port}")

    # ── Low-level frame I/O ────────────────────────────────────────────

    def _send(
        self,
        frame: bytes,
        expect_reply: bool = True,
        reply_timeout: float = REPLY_TIMEOUT,
    ) -> tuple[int, bytes] | None:
        if self._serial is None:
            raise ConnectionError("Serial port not open")
        with self._lock:
            self._serial.reset_input_buffer()
            self._serial.write(frame)
            if not expect_reply:
                return None
            return self._read_reply(reply_timeout)

    def _read_reply(self, timeout: float) -> tuple[int, bytes] | None:
        assert self._serial is not None
        buf = bytearray()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            avail = self._serial.in_waiting
            if avail:
                buf.extend(self._serial.read(avail))
                result = self._try_parse(buf)
                if result is not None:
                    return result
            else:
                time.sleep(0.001)
        return None

    def _try_parse(self, buf: bytearray) -> tuple[int, bytes] | None:
        while len(buf) >= 6:
            idx = bytes(buf).find(b"\xff\xff")
            if idx < 0:
                buf.clear()
                return None
            if idx > 0:
                del buf[:idx]
            if len(buf) < 4:
                return None
            length = buf[3]
            total = 4 + length
            if len(buf) < total:
                return None
            parsed = parse_frame(bytes(buf[:total]))
            if parsed is not None:
                del buf[:total]
                return (parsed[1], parsed[2])
            else:
                del buf[:2]
        return None

    def _raw_read_positions(self, retries: int = 3) -> list[int]:
        """Read 6 raw servo positions via CMD 96, retrying on garbled replies.

        The leader firmware's short-checksum bug leaves the 12 position bytes
        unverified, so a bit flip on the 1 Mbps bus can slip past parse_frame.
        HX30HM is a 12-bit encoder, so any value outside [0, 4095] is
        physically impossible — reject the frame and retry instead of
        clamping (which would poison record_ranges_of_motion's min/max).
        """
        for attempt in range(retries):
            frame = build_frame(SYSTEM_ID, CMD_READ_POS)
            result = self._send(frame, reply_timeout=REPLY_TIMEOUT)
            if result is not None:
                _, args = result
                if len(args) >= JOINT_COUNT * 2:
                    raw = [struct.unpack_from("<h", args, i * 2)[0] for i in range(JOINT_COUNT)]
                    if all(POSITION_MIN <= v <= POSITION_MAX for v in raw):
                        return raw
            if attempt < retries - 1:
                time.sleep(0.005)
        raise TimeoutError("No valid position reply from NexArm")

    def _raw_write_positions(self, positions_by_id: dict[int, int]) -> None:
        """Write 6 raw servo positions via CMD 97 (sync write, no reply).

        Missing IDs default to the last known target (or mid-range on first call).
        """
        if not hasattr(self, "_last_goal_raw"):
            self._last_goal_raw = [HALF_TURN] * JOINT_COUNT
        for i in range(JOINT_COUNT):
            id_ = i + 1
            if id_ in positions_by_id:
                self._last_goal_raw[i] = max(
                    POSITION_MIN, min(POSITION_MAX, int(round(positions_by_id[id_])))
                )
        args = b"".join(struct.pack("<h", self._last_goal_raw[i]) for i in range(JOINT_COUNT))
        frame = build_frame(SYSTEM_ID, CMD_WRITE_POS, args)
        self._send(frame, expect_reply=False)

    # ── LeRobot-mode control ──────────────────────────────────────────

    def enter_lerobot_mode(self) -> None:
        frame = build_frame(SYSTEM_ID, CMD_LEROBOT_MODE, bytes([1]))
        self._send(frame, expect_reply=False)
        time.sleep(0.1)

    def exit_lerobot_mode(self) -> None:
        frame = build_frame(SYSTEM_ID, CMD_LEROBOT_MODE, bytes([0]))
        self._send(frame, expect_reply=False)
        time.sleep(0.1)

    def write_motion_params(self, acc: int, speed: int = 0) -> None:
        """Set per-frame servo acceleration and max speed (CMD 56).

        acc:   0-254. 0 = max acceleration, ~100 is a smooth default.
        speed: 0-3400 raw units/s. 0 = no limit (full speed).
        """
        acc = max(0, min(254, int(acc)))
        speed = max(0, min(3400, int(speed)))
        args = bytes([acc & 0xFF, speed & 0xFF, (speed >> 8) & 0xFF])
        frame = build_frame(SYSTEM_ID, CMD_SET_MOTION_PARAMS, args)
        self._send(frame, expect_reply=False)

    # ── Torque ────────────────────────────────────────────────────────

    # CMD 98 is fanned out by the AT32 co-processor to all 6 servos with no
    # per-servo retry. On a noisy 1 Mbps half-duplex bus one servo (usually
    # 5/6 — they sit at the end of the loop) can drop the write packet and
    # stay enabled. Resend the frame ``_TORQUE_RESEND_TIMES`` times with a
    # short gap so any single-frame loss is covered.
    _TORQUE_RESEND_TIMES = 5
    _TORQUE_RESEND_GAP = 0.02

    def _send_torque(self, enable: bool) -> None:
        frame = build_frame(SYSTEM_ID, CMD_TORQUE, bytes([1 if enable else 0]))
        for _ in range(self._TORQUE_RESEND_TIMES):
            self._send(frame, expect_reply=False)
            time.sleep(self._TORQUE_RESEND_GAP)

    @check_if_not_connected
    def enable_torque(
        self, motors: str | list[str] | None = None, num_retry: int = 0
    ) -> None:
        # NexArm only supports torque on/off for the whole bus.
        _ = motors, num_retry
        self._send_torque(True)

    @check_if_not_connected
    def disable_torque(
        self, motors: str | list[str] | None = None, num_retry: int = 0
    ) -> None:
        _ = motors, num_retry
        self._send_torque(False)

    @contextlib.contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None):
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    # ── Standard read/write API ───────────────────────────────────────

    def _get_motors_list(self, motors: NameOrID | Sequence[NameOrID] | None) -> list[str]:
        if motors is None:
            return list(self._ordered_names)
        if isinstance(motors, str):
            return [motors]
        if isinstance(motors, int):
            return [self._id_to_name_dict[motors]]
        return [m if isinstance(m, str) else self._id_to_name_dict[m] for m in motors]

    def _check_data_name(self, data_name: str) -> None:
        if data_name not in ("Present_Position", "Goal_Position"):
            raise NotImplementedError(
                f"NexArmMotorsBus only supports 'Present_Position' and 'Goal_Position', got {data_name!r}"
            )

    @check_if_not_connected
    def read(self, data_name: str, motor: str) -> Value:
        self._check_data_name(data_name)
        if data_name != "Present_Position":
            raise NotImplementedError(f"NexArmMotorsBus.read does not support {data_name!r}")
        values = self.sync_read("Present_Position", [motor])
        return values[motor]

    @check_if_not_connected
    def write(self, data_name: str, motor: str, value: Value) -> None:
        self._check_data_name(data_name)
        if data_name != "Goal_Position":
            raise NotImplementedError(f"NexArmMotorsBus.write does not support {data_name!r}")
        self.sync_write("Goal_Position", {motor: value})

    @check_if_not_connected
    def sync_read(
        self,
        data_name: str,
        motors: NameOrID | Sequence[NameOrID] | None = None,
        *,
        normalize: bool = True,
    ) -> dict[str, Value]:
        self._check_data_name(data_name)
        if data_name != "Present_Position":
            raise NotImplementedError(f"NexArmMotorsBus.sync_read does not support {data_name!r}")
        names = self._get_motors_list(motors)
        raw_all = self._raw_read_positions()
        raw_by_name: dict[str, int] = {}
        for name in names:
            motor_id = self.motors[name].id
            raw_by_name[name] = raw_all[motor_id - 1]
        if not normalize or not self.calibration:
            return {name: float(raw_by_name[name]) for name in names}
        ids_values = {self.motors[name].id: raw_by_name[name] for name in names}
        normalized = self._normalize(ids_values)
        return {self._id_to_name_dict[id_]: normalized[id_] for id_ in ids_values}

    @check_if_not_connected
    def sync_write(
        self,
        data_name: str,
        values: dict[str, Value],
        *,
        normalize: bool = True,
    ) -> None:
        self._check_data_name(data_name)
        if data_name != "Goal_Position":
            raise NotImplementedError(f"NexArmMotorsBus.sync_write does not support {data_name!r}")
        if normalize and self.calibration:
            ids_values = {self.motors[name].id: float(v) for name, v in values.items()}
            raw_by_id = self._unnormalize(ids_values)
        else:
            raw_by_id = {
                self.motors[name].id: max(POSITION_MIN, min(POSITION_MAX, int(round(float(v)))))
                for name, v in values.items()
            }
        # Apply range_min/max clamp in raw space for safety.
        for id_ in list(raw_by_id):
            name = self._id_to_name_dict[id_]
            cal = self.calibration.get(name)
            if cal is not None and cal.range_min != cal.range_max:
                raw_by_id[id_] = max(cal.range_min, min(cal.range_max, raw_by_id[id_]))
            else:
                raw_by_id[id_] = max(POSITION_MIN, min(POSITION_MAX, raw_by_id[id_]))
        self._raw_write_positions(raw_by_id)

    # ── Normalisation (mirrors SerialMotorsBus) ──────────────────────

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")
        normalized: dict[int, float] = {}
        for id_, val in ids_values.items():
            name = self._id_to_name_dict[id_]
            cal = self.calibration[name]
            drive_mode = self.apply_drive_mode and cal.drive_mode
            min_, max_ = cal.range_min, cal.range_max
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{name}': min == max")
            bounded = min(max_, max(min_, val))
            mode = self.motors[name].norm_mode
            if mode is MotorNormMode.RANGE_M100_100:
                norm = (((bounded - min_) / (max_ - min_)) * 200) - 100
                normalized[id_] = -norm if drive_mode else norm
            elif mode is MotorNormMode.RANGE_0_100:
                norm = ((bounded - min_) / (max_ - min_)) * 100
                normalized[id_] = 100 - norm if drive_mode else norm
            elif mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self.motors[name].model] - 1
                deg = (val - mid) * 360 / max_res
                normalized[id_] = -deg if drive_mode else deg
            else:
                raise NotImplementedError(mode)
        return normalized

    def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")
        unnormalized: dict[int, int] = {}
        for id_, val in ids_values.items():
            name = self._id_to_name_dict[id_]
            cal = self.calibration[name]
            drive_mode = self.apply_drive_mode and cal.drive_mode
            min_, max_ = cal.range_min, cal.range_max
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{name}': min == max")
            mode = self.motors[name].norm_mode
            if mode is MotorNormMode.RANGE_M100_100:
                v = -val if drive_mode else val
                bounded = min(100.0, max(-100.0, v))
                unnormalized[id_] = int(((bounded + 100) / 200) * (max_ - min_) + min_)
            elif mode is MotorNormMode.RANGE_0_100:
                v = 100 - val if drive_mode else val
                bounded = min(100.0, max(0.0, v))
                unnormalized[id_] = int((bounded / 100) * (max_ - min_) + min_)
            elif mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self.motors[name].model] - 1
                v = -val if drive_mode else val
                unnormalized[id_] = int((v * max_res / 360) + mid)
            else:
                raise NotImplementedError(mode)
        return unnormalized

    # ── Calibration helpers ───────────────────────────────────────────

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """NexArm servos do not expose calibration registers; return cache."""
        return dict(self.calibration)

    def write_calibration(
        self, calibration_dict: dict[str, MotorCalibration], cache: bool = True
    ) -> None:
        """NexArm servos do not expose calibration registers; only update cache."""
        if cache:
            self.calibration = dict(calibration_dict)

    def reset_calibration(
        self, motors: NameOrID | Sequence[NameOrID] | None = None
    ) -> None:
        names = self._get_motors_list(motors)
        for name in names:
            self.calibration[name] = MotorCalibration(
                id=self.motors[name].id,
                drive_mode=0,
                homing_offset=0,
                range_min=POSITION_MIN,
                range_max=POSITION_MAX,
            )

    def set_half_turn_homings(
        self, motors: NameOrID | Sequence[NameOrID] | None = None
    ) -> dict[str, int]:
        """Record the current position of each motor as its half-turn (mid) point.

        The recorded value is stored as ``homing_offset = current_raw - HALF_TURN``.
        ``range_min``/``range_max`` are left at full resolution until
        :pymeth:`record_ranges_of_motion` runs.
        """
        names = self._get_motors_list(motors)
        present = self.sync_read("Present_Position", names, normalize=False)
        offsets: dict[str, int] = {}
        for name in names:
            raw = int(present[name])
            offset = raw - HALF_TURN
            existing = self.calibration.get(name)
            drive_mode = existing.drive_mode if existing else 0
            range_min = existing.range_min if existing else POSITION_MIN
            range_max = existing.range_max if existing else POSITION_MAX
            if range_min == range_max:
                range_min, range_max = POSITION_MIN, POSITION_MAX
            self.calibration[name] = MotorCalibration(
                id=self.motors[name].id,
                drive_mode=drive_mode,
                homing_offset=offset,
                range_min=range_min,
                range_max=range_max,
            )
            offsets[name] = offset
        return offsets

    def record_ranges_of_motion(
        self,
        motors: NameOrID | Sequence[NameOrID] | None = None,
        display_values: bool = True,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Stream live positions while the operator walks each joint through its full range.

        Press ENTER to finish. Returns (min_dict, max_dict).
        """
        names = self._get_motors_list(motors)
        start = self.sync_read("Present_Position", names, normalize=False)
        mins = {name: int(start[name]) for name in names}
        maxes = dict(mins)

        done = False
        while not done:
            positions = self.sync_read("Present_Position", names, normalize=False)
            for name in names:
                p = int(positions[name])
                mins[name] = min(mins[name], p)
                maxes[name] = max(maxes[name], p)
            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>6} | {'POS':>6} | {'MAX':>6}")
                for name in names:
                    print(
                        f"{name:<15} | {mins[name]:>6} | {int(positions[name]):>6} | {maxes[name]:>6}"
                    )
            if enter_pressed():
                done = True
            if display_values and not done:
                move_cursor_up(len(names) + 3)

        same = [n for n in names if mins[n] == maxes[n]]
        if same:
            raise ValueError(
                f"Some motors have identical min and max — did the joint move? {pformat(same)}"
            )

        for name in names:
            existing = self.calibration.get(name)
            self.calibration[name] = MotorCalibration(
                id=self.motors[name].id,
                drive_mode=existing.drive_mode if existing else 0,
                homing_offset=existing.homing_offset if existing else 0,
                range_min=mins[name],
                range_max=maxes[name],
            )
        return mins, maxes
