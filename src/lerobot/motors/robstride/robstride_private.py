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

"""Robstride *private protocol* (factory default) CAN bus.

Robstride motors ship speaking their vendor "private" protocol: 29-bit extended CAN IDs
packed as ``comm_type[28:24] | data16[23:8] | target_id[7:0]``, parameter-driven control,
and per-command feedback acknowledgments. :class:`~lerobot.motors.robstride.RobstrideMotorsBus`
instead targets motors that were re-flashed into the MIT communication mode (11-bit standard
IDs); a motor speaks exactly one of the two at a time, selected by a persisted setting plus a
power cycle.

This bus works with motors out of the box (no protocol switch) and exposes the modes the MIT
frame format cannot: Position (``run_mode`` 1, driven by ``loc_ref`` parameter writes),
Velocity (``run_mode`` 2, ``spd_ref``) and Current (``run_mode`` 3, ``iq_ref``), alongside
operation/MIT control frames (``run_mode`` 0). See issues #3547 and #3488 for the design
discussion.

Implementation notes learned from real RS-series hardware:

- Live positions are read as *parameter reads* of ``mech_pos`` (0x7019), which return the
  exact float32 in radians and work whether the motor is enabled, stopped or streaming.
  The type-0x02 feedback frames are only harvested opportunistically: they are command
  acknowledgments, not a reliable telemetry stream (on tested firmware, enabling the active
  report yields a single frame per motor, then goes quiet).
- Motors stream compact type-0x18 report frames by default. They are not documented and are
  skipped (with an absolute deadline so a report flood cannot starve a read of its timeout
  budget).
- ``run_mode`` writes are ignored by some firmware revisions while torque is enabled, so mode
  switches always stop the motor first and read the mode back to verify.
- ``mech_vel`` is parameter 0x701B; its neighbor 0x701A is ``iqf``, the filtered q-axis
  current in amperes. Hardware-verified on rs00 and rs02: 0x701B tracks the numerical
  derivative of ``mech_pos`` sample by sample, while 0x701A reads exactly 0.0 with torque
  disabled.
- A read of an index the firmware does not support is answered with an *error echo*: the
  reply sets data16's high byte and carries a zeroed value payload. Verified on rs00, rs03
  and rs06. Reply filters must reject those frames or a bad index silently reads as 0.0.
"""

import logging
import struct
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _can_available, require_package

if TYPE_CHECKING or _can_available:
    import can
else:
    can = SimpleNamespace(Message=object, interface=None, BusABC=object)

from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..motors_bus import Motor, MotorCalibration, MotorsBusBase, NameOrID, Value
from .robstride import MotorState
from .tables import (
    AVAILABLE_BAUDRATES,
    DEFAULT_BAUDRATE,
    DEFAULT_PRIVATE_HOST_ID,
    DEFAULT_TIMEOUT_MS,
    PRIVATE_ACK_TIMEOUT_S,
    PRIVATE_MODE_SWITCH_RETRIES,
    PRIVATE_PARAM_TIMEOUT_S,
    PRIVATE_PARAMS,
    PRIVATE_RECV_POLL_S,
    RS_MODEL_LIMITS,
    PrivateCommType,
    PrivateControlMode,
    PrivateParam,
    RSModelLimits,
)

logger = logging.getLogger(__name__)

_RAD_TO_DEG = 180.0 / 3.141592653589793
_DEG_TO_RAD = 3.141592653589793 / 180.0

# Feedback (type-0x02) frames carry fault bits 21:16 and a 2-bit status in bits 23:22 of the
# extended ID. Status codes: 0 = reset, 1 = calibration, 2 = run.
_FEEDBACK_FAULT_MASK = 0x3F
_FEEDBACK_FAULT_SHIFT = 16
_FEEDBACK_STATUS_SHIFT = 22


def _normalize_model(motor_type_str: str) -> str:
    """Normalize an RS model string ("RS-02", "rs02", "RS02") to the table key "rs02"."""
    return motor_type_str.lower().replace("-", "").replace("_", "").strip()


class RobstridePrivateMotorsBus(MotorsBusBase):
    """
    Robstride bus speaking the vendor *private* protocol (the factory default).

    Unlike :class:`RobstrideMotorsBus` (MIT communication mode), this bus requires no
    protocol switch on the motors and supports parameter access plus the Position,
    Velocity and Current control modes in addition to operation/MIT control frames.

    Positions are exposed in degrees, velocities in degrees/second, torques in N·m and
    currents in amperes at the API boundary, matching the other CAN buses.
    """

    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        can_interface: str = "auto",
        bitrate: int = 1_000_000,
        host_id: int = DEFAULT_PRIVATE_HOST_ID,
    ):
        """
        Args:
            port: CAN channel (e.g. "can0" for socketcan, "/dev/ttyACM0" for slcan).
            motors: Mapping of motor name to :class:`Motor`. ``Motor.id`` is the CAN device
                id (e.g. 1..7 on the reBot B601-RS; ids are configurable) and
                ``Motor.motor_type_str`` must name an RS-series model ("rs00".."rs06"),
                which selects the MIT packing and feedback scalings.
            calibration: Optional calibration mapping (kept in memory; Robstride motors do
                not store LeRobot calibration internally).
            can_interface: "auto" (default), "socketcan" or "slcan". Auto selects slcan for
                "/dev/..." ports, socketcan otherwise.
            bitrate: CAN bitrate in bps. Robstride private protocol uses classic CAN 2.0 at
                1 Mbps (no CAN FD).
            host_id: Host id placed in outgoing frames and expected in replies. 0xFD matches
                the vendor tools. Only one master may use the bus at a time.
        """
        require_package("python-can", extra="robstride", import_name="can")
        super().__init__(port, motors, calibration)
        self.can_interface = can_interface
        self.bitrate = bitrate
        self.host_id = host_id
        self.canbus: can.BusABC | None = None
        self._is_connected = False

        self._model_limits: dict[str, RSModelLimits] = {}
        self._gains: dict[str, dict[str, float]] = {}
        for name, motor in self.motors.items():
            if motor.motor_type_str is None:
                raise ValueError(
                    f"Motor '{name}' has no motor_type_str. The private-protocol bus needs the "
                    f"RS model name (one of {sorted(RS_MODEL_LIMITS)}) for frame scaling."
                )
            model = _normalize_model(motor.motor_type_str)
            if model not in RS_MODEL_LIMITS:
                raise ValueError(
                    f"Motor '{name}' has unknown motor_type_str '{motor.motor_type_str}'. "
                    f"Expected one of {sorted(RS_MODEL_LIMITS)}."
                )
            self._model_limits[name] = RS_MODEL_LIMITS[model]
            # MIT-mode gains used when writing Goal_Position in MIT control mode.
            self._gains[name] = {"kp": 10.0, "kd": 0.5}

        self._id_to_name: dict[int, str] = {motor.id: name for name, motor in self.motors.items()}
        if len(self._id_to_name) != len(self.motors):
            raise ValueError("Duplicate motor CAN ids in the motors mapping.")

        self.enabled: dict[str, bool] = dict.fromkeys(self.motors, False)
        self.control_mode: dict[str, PrivateControlMode] = dict.fromkeys(
            self.motors, PrivateControlMode.POSITION
        )
        self.fault_bits: dict[str, int] = dict.fromkeys(self.motors, 0)
        self.status_code: dict[str, int] = dict.fromkeys(self.motors, 0)
        self.last_feedback_time: dict[str, float | None] = dict.fromkeys(self.motors, None)
        self._last_known_states: dict[str, MotorState] = {
            name: {"position": 0.0, "velocity": 0.0, "torque": 0.0, "temp_mos": 0.0, "temp_rotor": 0.0}
            for name in self.motors
        }
        self._consecutive_read_failures: dict[str, int] = dict.fromkeys(self.motors, 0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.canbus is not None

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def _bus(self) -> can.BusABC:
        if self.canbus is None:
            raise DeviceNotConnectedError(f"{self.__class__.__name__}('{self.port}') is not connected.")
        return self.canbus

    @check_if_already_connected
    def connect(self, handshake: bool = True) -> None:
        """Open the CAN channel and optionally verify every motor answers a parameter read."""
        try:
            if self.can_interface == "auto":
                self.can_interface = "slcan" if self.port.startswith("/dev/") else "socketcan"
                logger.info(f"Auto-detected {self.can_interface} interface for port {self.port}")

            self.canbus = can.interface.Bus(
                channel=self.port, bitrate=self.bitrate, interface=self.can_interface
            )
            self._is_connected = True

            if handshake:
                self._handshake()

            logger.debug(f"{self.__class__.__name__} connected via {self.can_interface}.")
        except Exception as e:
            self._is_connected = False
            if self.canbus is not None:
                try:
                    self.canbus.shutdown()
                except Exception:
                    logger.debug("CAN bus shutdown failed during connect() rollback", exc_info=True)
                self.canbus = None
            raise ConnectionError(f"Failed to connect to CAN bus: {e}") from e

    def _handshake(self) -> None:
        """Verify every configured motor answers a ``mech_pos`` parameter read."""
        missing: list[str] = []
        for name in self.motors:
            value = self._read_param(name, "mech_pos", num_retry=2)
            if value is None:
                missing.append(f"{name} (id={self.motors[name].id})")
        if missing:
            raise ConnectionError(
                f"Missing motors on '{self.port}': {', '.join(missing)}. Check power, wiring, "
                f"CAN ids and that no other master (vendor tools) is using the bus."
            )

    @check_if_not_connected
    def disconnect(self, disable_torque: bool = True) -> None:
        """Close the CAN channel, optionally stopping every motor first."""
        if disable_torque:
            try:
                self.disable_torque()
            except Exception:
                logger.warning("Failed to disable torque during disconnect", exc_info=True)
        bus = self.canbus
        self.canbus = None
        self._is_connected = False
        if bus is not None:
            bus.shutdown()
        logger.debug(f"{self.__class__.__name__} disconnected.")

    # ------------------------------------------------------------------
    # Low-level framing
    # ------------------------------------------------------------------

    def _ext_id(self, comm_type: int, data16: int, target_id: int) -> int:
        return ((comm_type & 0x1F) << 24) | ((data16 & 0xFFFF) << 8) | (target_id & 0xFF)

    def _send(self, comm_type: int, motor_id: int, data: bytes, data16: int | None = None) -> None:
        payload = bytes(data) + bytes(8 - len(data))
        msg = can.Message(
            arbitration_id=self._ext_id(comm_type, self.host_id if data16 is None else data16, motor_id),
            data=payload,
            is_extended_id=True,
        )
        self._bus().send(msg)

    def _harvest_feedback(self, msg: "can.Message") -> str | None:
        """Decode a type-0x02 feedback frame into the state cache. Returns the motor name."""
        motor_id = (msg.arbitration_id >> 8) & 0xFF
        name = self._id_to_name.get(motor_id)
        if name is None or len(msg.data) < 8:
            return None
        limits = self._model_limits[name]
        pos_u16, vel_u16, torque_u16, temp_u16 = struct.unpack(">HHHH", bytes(msg.data[:8]))
        self._last_known_states[name] = {
            "position": (pos_u16 / 32767.0 - 1.0) * limits.p_max * _RAD_TO_DEG,
            "velocity": (vel_u16 / 32767.0 - 1.0) * limits.v_max * _RAD_TO_DEG,
            "torque": (torque_u16 / 32767.0 - 1.0) * limits.t_max,
            "temp_mos": temp_u16 * 0.1,
            "temp_rotor": self._last_known_states[name]["temp_rotor"],
        }
        self.last_feedback_time[name] = time.monotonic()
        fault = (msg.arbitration_id >> _FEEDBACK_FAULT_SHIFT) & _FEEDBACK_FAULT_MASK
        if fault and fault != self.fault_bits[name]:
            logger.warning(f"Motor '{name}' reports fault bits 0b{fault:06b}")
        self.fault_bits[name] = fault
        self.status_code[name] = (msg.arbitration_id >> _FEEDBACK_STATUS_SHIFT) & 0x3
        return name

    def _dispatch_frame(self, msg: "can.Message") -> None:
        """Route one incoming frame: harvest feedback, ignore report frames and foreign traffic."""
        if not msg.is_extended_id:
            return
        comm_type = (msg.arbitration_id >> 24) & 0x1F
        if comm_type == PrivateCommType.FEEDBACK:
            self._harvest_feedback(msg)
        # Type-0x18 report frames (streamed by default on RS firmware) and any other
        # traffic are intentionally ignored.

    def _collect(
        self,
        deadline: float,
        predicate: "object | None" = None,
    ) -> "can.Message | None":
        """Poll the socket until ``predicate(msg)`` matches or the absolute deadline passes.

        Every frame seen along the way is dispatched (feedback harvested, reports skipped),
        so a report-frame flood consumes the deadline instead of spinning forever.
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            msg = self._bus().recv(timeout=min(PRIVATE_RECV_POLL_S, remaining))
            if msg is None:
                continue
            self._dispatch_frame(msg)
            if predicate is not None and callable(predicate) and predicate(msg):
                return msg

    def flush_rx_queue(self, poll_timeout_s: float = PRIVATE_RECV_POLL_S, max_messages: int = 4096) -> int:
        """Drain pending RX frames (harvesting any feedback), returning the number drained."""
        count = 0
        while count < max_messages:
            msg = self._bus().recv(timeout=poll_timeout_s)
            if msg is None:
                break
            self._dispatch_frame(msg)
            count += 1
        return count

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def _get_param(self, param: str | int) -> PrivateParam:
        if isinstance(param, str):
            try:
                return PRIVATE_PARAMS[param]
            except KeyError:
                raise KeyError(f"Unknown parameter '{param}'. Known: {sorted(PRIVATE_PARAMS)}") from None
        return PrivateParam(index=param, fmt="f")

    def _read_param_once(self, motor_id: int, param: PrivateParam, timeout: float) -> float | int | None:
        self._send(PrivateCommType.PARAM_READ, motor_id, struct.pack("<H", param.index))
        deadline = time.monotonic() + timeout

        def matches(msg: "can.Message") -> bool:
            if not msg.is_extended_id or len(msg.data) < 8:
                return False
            arb = msg.arbitration_id
            return (
                (arb >> 24) & 0x1F == PrivateCommType.PARAM_READ
                and arb & 0xFF == self.host_id
                and (arb >> 8) & 0xFF == motor_id
                and struct.unpack("<H", bytes(msg.data[:2]))[0] == param.index
            )

        msg = self._collect(deadline, predicate=matches)
        if msg is None:
            return None
        error_flag = (msg.arbitration_id >> 16) & 0xFF
        if error_flag:
            logger.warning(
                f"Motor id={motor_id} answered the read of param 0x{param.index:04X} with error "
                f"flag 0x{error_flag:02X} (unsupported index on this firmware?); ignoring the reply."
            )
            return None
        size = struct.calcsize(param.fmt)
        value: float | int = struct.unpack("<" + param.fmt, bytes(msg.data[4 : 4 + size]))[0]
        return value

    def read_param(self, motor: NameOrID, param: str | int, *, num_retry: int = 0) -> float | int | None:
        """Read a private-protocol parameter (by table name or raw index) from one motor.

        Returns the decoded value, or ``None`` when the motor did not answer within the
        timeout on any attempt.
        """
        return self._read_param(self._get_motor_name(motor), param, num_retry=num_retry)

    def _read_param(self, motor: str, param: str | int, *, num_retry: int = 0) -> float | int | None:
        motor_id = self.motors[motor].id
        entry = self._get_param(param)
        for _ in range(1 + num_retry):
            value = self._read_param_once(motor_id, entry, PRIVATE_PARAM_TIMEOUT_S)
            if value is not None:
                return value
        return None

    def write_param(
        self,
        motor: NameOrID,
        param: str | int,
        value: float | int,
        *,
        wait_ack: bool = False,
    ) -> None:
        """Write a private-protocol parameter on one motor.

        Setpoint writes in tight control loops should keep ``wait_ack=False`` (fire and
        forget; acknowledgments are harvested opportunistically). Configuration writes can
        set ``wait_ack=True`` to block until the motor acknowledges with a feedback frame.

        The read-only guard applies to named parameters; a raw integer index is an expert
        escape hatch and is written as-is (as float32).
        """
        name = self._get_motor_name(motor)
        entry = self._get_param(param)
        if isinstance(param, str) and not entry.writable:
            raise ValueError(f"Parameter '{param}' is read-only.")
        motor_id = self.motors[name].id
        payload = struct.pack("<H", entry.index) + b"\x00\x00" + self._pack_value(entry, value)
        self._send(PrivateCommType.PARAM_WRITE, motor_id, payload)
        if wait_ack:
            self._wait_feedback(name, PRIVATE_ACK_TIMEOUT_S)

    def _pack_value(self, entry: PrivateParam, value: float | int) -> bytes:
        raw = struct.pack("<" + entry.fmt, float(value) if entry.fmt == "f" else int(value))
        return raw + bytes(4 - len(raw))

    def _wait_feedback(self, motor: str, timeout: float) -> bool:
        """Wait until a type-0x02 feedback frame from ``motor`` is harvested."""
        motor_id = self.motors[motor].id
        deadline = time.monotonic() + timeout

        def matches(msg: "can.Message") -> bool:
            arb = msg.arbitration_id
            return (
                msg.is_extended_id
                and (arb >> 24) & 0x1F == PrivateCommType.FEEDBACK
                and (arb >> 8) & 0xFF == motor_id
            )

        return self._collect(deadline, predicate=matches) is not None

    # ------------------------------------------------------------------
    # Torque, faults, zeroing, persistence
    # ------------------------------------------------------------------

    @check_if_not_connected
    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Enable torque (type 0x03). Waits for each motor's acknowledgment."""
        for name in self._get_motors_list(motors):
            for attempt in range(1 + num_retry):
                self._send(PrivateCommType.ENABLE, self.motors[name].id, b"")
                if self._wait_feedback(name, PRIVATE_ACK_TIMEOUT_S):
                    self.enabled[name] = True
                    break
                if attempt == num_retry:
                    raise RuntimeError(f"Motor '{name}' did not acknowledge torque enable.")

    @check_if_not_connected
    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Disable torque / stop (type 0x04). The motor becomes back-drivable."""
        for name in self._get_motors_list(motors):
            for attempt in range(1 + num_retry):
                self._send(PrivateCommType.STOP, self.motors[name].id, b"")
                if self._wait_feedback(name, PRIVATE_ACK_TIMEOUT_S):
                    self.enabled[name] = False
                    break
                if attempt == num_retry:
                    # Do not raise: disable is used on teardown paths where the bus may
                    # already be degraded. The warning keeps the failure visible.
                    logger.warning(f"Motor '{name}' did not acknowledge torque disable.")
                    self.enabled[name] = False

    @contextmanager
    def torque_disabled(self, motors: str | list[str] | None = None) -> Iterator[None]:
        """Context manager that disables torque on entry and re-enables it on exit."""
        self.disable_torque(motors)
        try:
            yield
        finally:
            self.enable_torque(motors)

    @check_if_not_connected
    def clear_fault(self, motors: str | list[str] | None = None) -> None:
        """Clear latched faults (type 0x04 with data[0]=1)."""
        for name in self._get_motors_list(motors):
            self._send(PrivateCommType.STOP, self.motors[name].id, b"\x01")
            if not self._wait_feedback(name, PRIVATE_ACK_TIMEOUT_S):
                logger.warning(f"Motor '{name}' did not acknowledge the fault-clear command.")

    @check_if_not_connected
    def set_zero_position(self, motors: str | list[str] | None = None) -> None:
        """Set the current mechanical position as zero (motor must be stopped).

        Sends type 0x06 (data[0]=1) followed by a ``zero_sta`` parameter write, matching the
        vendor SDK sequence. The single-turn zero is stored by the motor; the multi-turn
        count is volatile, so joints with more than one turn of travel can wake wrapped
        after a power cycle and must be re-homed. :meth:`save_parameters` additionally
        flashes the parameter file (the vendor tools treat it as an optional follow-up).

        On some firmware revisions the new zero only takes effect after a full power cycle
        of the motor supply (reconnecting CAN is not enough). Read back ``Present_Position``
        to confirm; if it still reports the old frame, power-cycle the motors.

        Raises:
            RuntimeError: If a motor is torque-enabled or does not acknowledge the command.
        """
        for name in self._get_motors_list(motors):
            if self.enabled[name]:
                raise RuntimeError(f"Motor '{name}' must be torque-disabled before zeroing.")
            self._send(PrivateCommType.SET_ZERO, self.motors[name].id, b"\x01")
            if not self._wait_feedback(name, PRIVATE_ACK_TIMEOUT_S):
                raise RuntimeError(f"Motor '{name}' did not acknowledge the set-zero command.")
            self.write_param(name, "zero_sta", 1, wait_ack=True)

    @check_if_not_connected
    def save_parameters(self, motors: str | list[str] | None = None) -> None:
        """Persist the motor's parameters to flash (type 0x16).

        Raises:
            RuntimeError: If a motor does not acknowledge the save within its 500 ms window.
        """
        for name in self._get_motors_list(motors):
            self._send(PrivateCommType.SAVE_PARAMS, self.motors[name].id, bytes(range(1, 9)))
            if not self._wait_feedback(name, 0.5):
                raise RuntimeError(f"Motor '{name}' did not acknowledge the parameter save.")

    @check_if_not_connected
    def ping(self, motor: NameOrID) -> bool:
        """Type-0x00 ping: returns True when the motor answers."""
        name = self._get_motor_name(motor)
        motor_id = self.motors[name].id
        self._send(PrivateCommType.PING, motor_id, b"")
        deadline = time.monotonic() + PRIVATE_PARAM_TIMEOUT_S

        def matches(msg: "can.Message") -> bool:
            return (
                msg.is_extended_id
                and (msg.arbitration_id >> 24) & 0x1F == PrivateCommType.PING
                and (msg.arbitration_id >> 8) & 0xFF == motor_id
            )

        return self._collect(deadline, predicate=matches) is not None

    # ------------------------------------------------------------------
    # Control modes
    # ------------------------------------------------------------------

    @check_if_not_connected
    def set_control_mode(self, mode: PrivateControlMode, motors: str | list[str] | None = None) -> None:
        """Switch motors to ``mode``, stopping them first and verifying by read-back.

        Some firmware revisions silently ignore ``run_mode`` writes while torque is enabled,
        so this always sends a stop first. Motors are left torque-disabled; call
        :meth:`enable_torque` when ready to move.
        """
        for name in self._get_motors_list(motors):
            for attempt in range(PRIVATE_MODE_SWITCH_RETRIES):
                self.disable_torque(name)
                self.write_param(name, "run_mode", int(mode), wait_ack=True)
                read_back = self._read_param(name, "run_mode", num_retry=1)
                if read_back is not None and int(read_back) == int(mode):
                    self.control_mode[name] = mode
                    break
                if attempt == PRIVATE_MODE_SWITCH_RETRIES - 1:
                    raise RuntimeError(
                        f"Motor '{name}' did not switch to {mode.name} (run_mode reads back {read_back})."
                    )

    @check_if_not_connected
    def configure_motors(self, mode: PrivateControlMode = PrivateControlMode.POSITION) -> None:
        """Put every motor in ``mode`` (torque-disabled). Robots call this from ``configure()``."""
        self.set_control_mode(mode)

    # ------------------------------------------------------------------
    # MIT (operation mode) control frames
    # ------------------------------------------------------------------

    def _mit_control(
        self,
        motor: NameOrID,
        kp: float,
        kd: float,
        position: float,
        velocity: float,
        torque: float,
    ) -> None:
        """Send one type-0x01 operation-mode frame (position [deg], velocity [deg/s], torque [N·m])."""
        name = self._get_motor_name(motor)
        limits = self._model_limits[name]
        pos_rad = position * _DEG_TO_RAD
        vel_rad = velocity * _DEG_TO_RAD

        def to_u16(value: float, half_range: float) -> int:
            clamped = min(max(value, -half_range), half_range)
            return int(round((clamped / half_range + 1.0) * 32767.0))

        def gain_to_u16(value: float, full_scale: float) -> int:
            clamped = min(max(value, 0.0), full_scale)
            return int(round(clamped / full_scale * 65535.0))

        payload = struct.pack(
            ">HHHH",
            to_u16(pos_rad, limits.p_max),
            to_u16(vel_rad, limits.v_max),
            gain_to_u16(kp, limits.kp_max),
            gain_to_u16(kd, limits.kd_max),
        )
        torque_u16 = to_u16(torque, limits.t_max)
        self._send(PrivateCommType.MIT_CONTROL, self.motors[name].id, payload, data16=torque_u16)

    def _mit_control_batch(self, commands: dict[NameOrID, tuple[float, float, float, float, float]]) -> None:
        """Send MIT frames for several motors: ``{motor: (kp, kd, pos_deg, vel_deg_s, torque)}``."""
        for motor, (kp, kd, position, velocity, torque) in commands.items():
            self._mit_control(motor, kp, kd, position, velocity, torque)
        self.flush_rx_queue()

    # ------------------------------------------------------------------
    # Data API
    # ------------------------------------------------------------------

    @check_if_not_connected
    def read(self, data_name: str, motor: str) -> Value:
        """Read one value from one motor.

        Supported: ``Present_Position`` [deg], ``Present_Velocity`` [deg/s],
        ``Present_Torque`` [N·m], ``Temperature_MOS`` [°C], ``VBUS`` [V].
        """
        if data_name == "Present_Position":
            value = self._read_param(motor, "mech_pos", num_retry=1)
            if value is None:
                return self._stale_read(motor, "position")
            self._consecutive_read_failures[motor] = 0
            position = float(value) * _RAD_TO_DEG
            self._last_known_states[motor]["position"] = position
            return position
        if data_name == "Present_Velocity":
            value = self._read_param(motor, "mech_vel", num_retry=1)
            if value is None:
                return self._stale_read(motor, "velocity")
            self._consecutive_read_failures[motor] = 0
            velocity = float(value) * _RAD_TO_DEG
            self._last_known_states[motor]["velocity"] = velocity
            return velocity
        if data_name == "Present_Torque":
            return self._last_known_states[motor]["torque"]
        if data_name == "Temperature_MOS":
            return self._last_known_states[motor]["temp_mos"]
        if data_name == "VBUS":
            value = self._read_param(motor, "vbus", num_retry=1)
            return float(value) if value is not None else 0.0
        raise ValueError(f"Reading '{data_name}' is not supported by {self.__class__.__name__}.")

    def _stale_read(self, motor: str, field: str) -> float:
        self._consecutive_read_failures[motor] += 1
        level = logging.ERROR if self._consecutive_read_failures[motor] >= 3 else logging.WARNING
        logger.log(
            level,
            f"Motor '{motor}' missed {self._consecutive_read_failures[motor]} consecutive "
            f"{field} read(s); using last known value.",
        )
        value: float = self._last_known_states[motor][field]  # type: ignore[literal-required]
        return value

    @check_if_not_connected
    def write(self, data_name: str, motor: str, value: Value) -> None:
        """Write one value to one motor.

        - ``Goal_Position`` [deg]: MIT frame in MIT mode (with the stored Kp/Kd), ``loc_ref``
          parameter write in Position mode.
        - ``Goal_Velocity`` [deg/s]: ``spd_ref`` write (requires Velocity mode).
        - ``Goal_Current`` [A]: ``iq_ref`` write (requires Current mode).
        - ``Kp`` / ``Kd``: stored gains used by MIT-mode ``Goal_Position`` writes.
        """
        if data_name == "Kp":
            self._gains[motor]["kp"] = float(value)
            return
        if data_name == "Kd":
            self._gains[motor]["kd"] = float(value)
            return
        if data_name == "Goal_Position":
            mode = self.control_mode[motor]
            if mode == PrivateControlMode.MIT:
                gains = self._gains[motor]
                self._mit_control(motor, gains["kp"], gains["kd"], float(value), 0.0, 0.0)
                self.flush_rx_queue()
            elif mode == PrivateControlMode.POSITION:
                self.write_param(motor, "loc_ref", float(value) * _DEG_TO_RAD)
                self.flush_rx_queue()
            else:
                raise ValueError(
                    f"Goal_Position requires MIT or Position mode, but motor '{motor}' is in "
                    f"{mode.name} mode. Call set_control_mode() first."
                )
            return
        if data_name == "Goal_Velocity":
            self._require_mode(motor, PrivateControlMode.VELOCITY, data_name)
            self.write_param(motor, "spd_ref", float(value) * _DEG_TO_RAD)
            self.flush_rx_queue()
            return
        if data_name == "Goal_Current":
            self._require_mode(motor, PrivateControlMode.CURRENT, data_name)
            self.write_param(motor, "iq_ref", float(value))
            self.flush_rx_queue()
            return
        raise ValueError(f"Writing '{data_name}' is not supported by {self.__class__.__name__}.")

    def _require_mode(self, motor: str, mode: PrivateControlMode, data_name: str) -> None:
        if self.control_mode[motor] != mode:
            raise ValueError(
                f"{data_name} requires {mode.name} mode, but motor '{motor}' is in "
                f"{self.control_mode[motor].name} mode. Call set_control_mode() first."
            )

    @check_if_not_connected
    def sync_read(self, data_name: str, motors: str | list[str] | None = None) -> dict[str, Value]:
        """Read one value from several motors (batched request-then-collect for parameters)."""
        names = self._get_motors_list(motors)
        if data_name == "Present_Position":
            values_rad = self._sync_read_param(names, "mech_pos")
            result: dict[str, Value] = {}
            for name in names:
                raw = values_rad.get(name)
                if raw is None:
                    result[name] = self._stale_read(name, "position")
                else:
                    self._consecutive_read_failures[name] = 0
                    position = float(raw) * _RAD_TO_DEG
                    self._last_known_states[name]["position"] = position
                    result[name] = position
            return result
        if data_name == "Present_Velocity":
            values_rad = self._sync_read_param(names, "mech_vel")
            result = {}
            for name in names:
                raw = values_rad.get(name)
                if raw is None:
                    result[name] = self._stale_read(name, "velocity")
                else:
                    self._consecutive_read_failures[name] = 0
                    velocity = float(raw) * _RAD_TO_DEG
                    self._last_known_states[name]["velocity"] = velocity
                    result[name] = velocity
            return result
        if data_name in ("Present_Torque", "Temperature_MOS"):
            field = "torque" if data_name == "Present_Torque" else "temp_mos"
            return {name: self._last_known_states[name][field] for name in names}  # type: ignore[literal-required]
        raise ValueError(f"Reading '{data_name}' is not supported by {self.__class__.__name__}.")

    def _sync_read_param(self, names: list[str], param: str) -> dict[str, float | int]:
        """Send one parameter-read request per motor back to back, then collect the replies."""
        entry = self._get_param(param)
        pending = {self.motors[name].id: name for name in names}
        for motor_id in pending:
            self._send(PrivateCommType.PARAM_READ, motor_id, struct.pack("<H", entry.index))

        results: dict[str, float | int] = {}
        deadline = time.monotonic() + PRIVATE_PARAM_TIMEOUT_S
        size = struct.calcsize(entry.fmt)

        def matches(msg: "can.Message") -> bool:
            if not msg.is_extended_id or len(msg.data) < 8:
                return False
            arb = msg.arbitration_id
            if (arb >> 24) & 0x1F != PrivateCommType.PARAM_READ or arb & 0xFF != self.host_id:
                return False
            motor_id = (arb >> 8) & 0xFF
            if motor_id not in pending:
                return False
            if struct.unpack("<H", bytes(msg.data[:2]))[0] != entry.index:
                return False
            error_flag = (arb >> 16) & 0xFF
            if error_flag:
                name = pending.pop(motor_id)
                logger.warning(
                    f"Motor '{name}' answered the read of param 0x{entry.index:04X} with error "
                    f"flag 0x{error_flag:02X} (unsupported index on this firmware?); ignoring the reply."
                )
                return not pending
            name = pending.pop(motor_id)
            results[name] = struct.unpack("<" + entry.fmt, bytes(msg.data[4 : 4 + size]))[0]
            return not pending  # stop collecting once every motor answered

        self._collect(deadline, predicate=matches)

        # One retry pass for the stragglers.
        for motor_id, name in list(pending.items()):
            value = self._read_param_once(motor_id, entry, PRIVATE_PARAM_TIMEOUT_S)
            if value is not None:
                results[name] = value
        return results

    @check_if_not_connected
    def sync_write(self, data_name: str, values: dict[str, Value]) -> None:
        """Write one value to several motors."""
        if data_name in ("Kp", "Kd"):
            for motor, value in values.items():
                self._gains[motor][data_name.lower()] = float(value)
            return
        if data_name == "Goal_Position":
            mit_batch: dict[NameOrID, tuple[float, float, float, float, float]] = {}
            for motor, value in values.items():
                mode = self.control_mode[motor]
                if mode == PrivateControlMode.MIT:
                    gains = self._gains[motor]
                    mit_batch[motor] = (gains["kp"], gains["kd"], float(value), 0.0, 0.0)
                elif mode == PrivateControlMode.POSITION:
                    self.write_param(motor, "loc_ref", float(value) * _DEG_TO_RAD)
                else:
                    raise ValueError(
                        f"Goal_Position requires MIT or Position mode, but motor '{motor}' is in "
                        f"{mode.name} mode. Call set_control_mode() first."
                    )
            if mit_batch:
                self._mit_control_batch(mit_batch)
            else:
                self.flush_rx_queue()
            return
        if data_name in ("Goal_Velocity", "Goal_Current"):
            for motor, value in values.items():
                self.write(data_name, motor, value)
            return
        raise ValueError(f"Writing '{data_name}' is not supported by {self.__class__.__name__}.")

    @check_if_not_connected
    def sync_read_all_states(
        self, motors: str | list[str] | None = None, *, num_retry: int = 0
    ) -> dict[str, MotorState]:
        """Return position/velocity/torque/temperature per motor.

        Only positions are actively refreshed (one ``mech_pos`` parameter read per motor);
        velocity, torque and temperature are the last values harvested from type-0x02
        feedback acknowledgments and may be stale, or 0.0 if none has been seen yet.
        """
        names = self._get_motors_list(motors)
        self.sync_read("Present_Position", names)
        return {name: deepcopy(self._last_known_states[name]) for name in names}

    # ------------------------------------------------------------------
    # Position-mode helpers
    # ------------------------------------------------------------------

    @check_if_not_connected
    def set_position_speed_limit(self, motor: NameOrID, speed_deg_s: float) -> None:
        """Set the Position-mode speed limit (``limit_spd``) for one motor [deg/s]."""
        self.write_param(motor, "limit_spd", speed_deg_s * _DEG_TO_RAD, wait_ack=True)

    @check_if_not_connected
    def set_position_gains(self, motor: NameOrID, *, loc_kp: float | None = None) -> None:
        """Set the Position-mode loop gain(s) for one motor."""
        if loc_kp is not None:
            self.write_param(motor, "loc_kp", loc_kp, wait_ack=True)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def read_calibration(self) -> dict[str, MotorCalibration]:
        """Robstride motors don't store LeRobot calibration internally; returns the cache."""
        return self.calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        """Cache the calibration in memory (nothing is written to the motors)."""
        if cache:
            self.calibration = calibration_dict

    @check_if_not_connected
    def record_ranges_of_motion(
        self, motors: str | list[str] | None = None, display_values: bool = True
    ) -> tuple[dict[str, Value], dict[str, Value]]:
        """Interactively record the min/max position of each motor, in degrees.

        Move the joints through their full range and press Enter to stop.
        """
        names = self._get_motors_list(motors)
        start_positions = self.sync_read("Present_Position", names)
        mins = dict(start_positions)
        maxes = dict(start_positions)

        user_pressed_enter = False
        while not user_pressed_enter:
            positions = self.sync_read("Present_Position", names)
            for name in names:
                mins[name] = min(mins[name], positions[name])
                maxes[name] = max(maxes[name], positions[name])

            if display_values:
                print("\n-------------------------------------------")
                print(f"{'NAME':<15} | {'MIN':>10} | {'POS':>10} | {'MAX':>10}")
                for name in names:
                    print(
                        f"{name:<15} | {mins[name]:>10.2f} | {positions[name]:>10.2f} | {maxes[name]:>10.2f}"
                    )

            if enter_pressed():
                user_pressed_enter = True

            if display_values and not user_pressed_enter:
                move_cursor_up(len(names) + 3)

        same = [name for name in names if abs(maxes[name] - mins[name]) < 5.0]
        if same:
            raise ValueError(f"Some motors barely moved (<5°): {same}. Re-run and move every joint.")
        return mins, maxes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_motors_list(self, motors: str | list[str] | None) -> list[str]:
        if motors is None:
            return list(self.motors.keys())
        if isinstance(motors, str):
            return [motors]
        if isinstance(motors, list):
            return motors
        raise TypeError(f"Invalid motors type: {type(motors)}")

    def _get_motor_name(self, motor: NameOrID) -> str:
        if isinstance(motor, str):
            if motor not in self.motors:
                raise ValueError(f"Unknown motor: {motor}")
            return motor
        name = self._id_to_name.get(motor)
        if name is None:
            raise ValueError(f"Unknown motor ID: {motor}")
        return name
