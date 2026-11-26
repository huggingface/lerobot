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

"""`MitMotorsBus` over `DamiaoMotorsBus`, for the Metal arm's seven motors."""

import math
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.damiao import MEDIUM_TIMEOUT_SEC, SHORT_TIMEOUT_SEC
from lerobot.motors.damiao.tables import CAN_CMD_REFRESH, CAN_PARAM_ID, MotorType
from lerobot.utils.import_utils import _can_available

from ..yam_follower.mit_control import MitCommand, MotorStates
from .config_metal_follower import MOTOR_NAMES

if TYPE_CHECKING or _can_available:
    import can

# Send IDs 1 to 7; each motor answers on its ID + 0x10.
MOTOR_IDS = range(1, len(MOTOR_NAMES) + 1)
FEEDBACK_ID_OFFSET = 0x10
# Damiao motor variants with Metal's MIT scaling (see `motors/damiao/tables.py`).
MOTOR_TYPES = ("metal_jlo", "metal_j2", "metal_jlo", "metal_jlo", "metal_jhi", "metal_jhi", "metal_jhi")
# Damiao reply status, the high nibble of a reply's first byte: 0 disabled, 1 enabled, anything else a fault;
# -1 means no reply yet.
_STATUS_DISABLED = 0
_STATUS_ENABLED = 1
_STATUS_OK = (-1, 0, 1)
_STATUS_KNOWN = (0, 1, 8, 9, 10, 11, 12, 13, 14)
# Silence that marks the end of queued replies, before a command whose reply must be told apart from them.
_QUIET_S = 0.003


class MetalMotorsBus:
    """CAN interface of one Metal arm for `MitController`, in radians.

    `DamiaoMotorsBus` encodes and decodes the frames. This adapter sends refresh and MIT frames itself and
    collects the replies, so it records when each motor last answered and its status; `read_states` raises if a
    motor reports a fault or hasn't answered within `feedback_timeout_s`. Enabling and disabling torque wait
    until every motor's status confirms it, in replies to requests sent after the queued replies were drained.
    Commands are buffered and sent as one batch when the gripper's, the last of each control cycle, arrives.
    One lock serializes every bus access.

    Args:
        port (`str`):
            SocketCAN interface or slcan serial port.
        can_interface (`str`):
            python-can interface: `auto`, `socketcan` or `slcan`.
        feedback_timeout_s (`float`):
            Maximum age of each motor's feedback.
        hold_kp (`np.ndarray`):
            Position gains of the arm joints' hold sent before enabling torque.
        hold_kd (`np.ndarray`):
            Damping gains of that hold.
    """

    motor_names: tuple[str, ...] = MOTOR_NAMES

    def __init__(
        self,
        port: str,
        can_interface: str,
        feedback_timeout_s: float,
        hold_kp: np.ndarray,
        hold_kd: np.ndarray,
    ) -> None:
        self.port = port
        self.feedback_timeout_s = feedback_timeout_s
        self.hold_kp, self.hold_kd = hold_kp, hold_kd
        motors = {
            name: Motor(
                motor_id,
                motor_type,
                MotorNormMode.DEGREES,
                motor_type_str=motor_type,
                recv_id=motor_id + FEEDBACK_ID_OFFSET,
            )
            for name, motor_id, motor_type in zip(MOTOR_NAMES, MOTOR_IDS, MOTOR_TYPES, strict=True)
        }
        self.bus = DamiaoMotorsBus(
            port, motors, can_interface=can_interface, use_can_fd=False, data_bitrate=None
        )
        self.enabled = False
        self._enabled_at = 0.0
        self._pending: dict[str, MitCommand] = {}
        self._replied_at = dict.fromkeys(MOTOR_NAMES, -math.inf)
        self._status = dict.fromkeys(MOTOR_NAMES, -1)
        self._replies = dict.fromkeys(MOTOR_NAMES, 0)
        self._lock = threading.RLock()

    def connect(self) -> None:
        """Open the CAN interface without the handshake, which enables torque, and check that torque is off.

        Raises:
            ConnectionError: If a motor doesn't answer, reports a fault or already has torque on.
        """
        with self._lock:
            self.enabled = False
            self._replied_at = dict.fromkeys(MOTOR_NAMES, -math.inf)
            self._status = dict.fromkeys(MOTOR_NAMES, -1)
            self.bus.connect(handshake=False)
            self._refresh_until_fresh()
            holding = [
                motor_id
                for motor_id, name in zip(MOTOR_IDS, self.motor_names, strict=True)
                if self._status[name] == _STATUS_ENABLED
            ]
            if holding:
                raise ConnectionError(
                    f"{self.port}: motors {holding} already have torque on; nothing was sent to them. Support the "
                    "arm and cut motor power"
                )

    def read_states(self) -> MotorStates:
        """Refresh every motor's state, retrying until all are fresh or `feedback_timeout_s` passes.

        Raises:
            ConnectionError: If a motor reports a fault, has no fresh feedback or was disabled unexpectedly.
        """
        with self._lock:
            self._refresh_until_fresh()
            if self.enabled and time.monotonic() - self._enabled_at > self.feedback_timeout_s:
                for motor_id, name in zip(MOTOR_IDS, self.motor_names, strict=True):
                    if self._status[name] == _STATUS_DISABLED:
                        raise ConnectionError(f"{self.port}: motor {motor_id} unexpectedly disabled")
            return self._states()

    def read_fresh(self) -> tuple[MotorStates, np.ndarray, np.ndarray]:
        """Refresh once, without raising.

        Returns:
            `tuple[MotorStates, np.ndarray, np.ndarray]`: The latest states, a mask of the motors that answered
            within `feedback_timeout_s` without a fault, and a mask of those with torque on.
        """
        with self._lock:
            self._refresh()
            now = time.monotonic()
            fresh = np.array(
                [
                    now - self._replied_at[name] <= self.feedback_timeout_s
                    and self._status[name] in _STATUS_OK
                    for name in self.motor_names
                ]
            )
            enabled = fresh & np.array([self._status[name] == _STATUS_ENABLED for name in self.motor_names])
            return self._states(), fresh, enabled

    def sample(self) -> tuple[MotorStates, np.ndarray, dict[str, int]]:
        """Refresh once, without raising, and return one consistent snapshot.

        Returns:
            `tuple[MotorStates, np.ndarray, dict[str, int]]`: The latest states, a mask of the motors that answered
            within `feedback_timeout_s` without a fault, and the number of valid replies received from each motor,
            to tell a new reply from the cached one.
        """
        with self._lock:
            states, fresh, _ = self.read_fresh()
            return states, fresh, dict(self._replies)

    def enable_torque_at(self, hold: np.ndarray) -> None:
        """`preload` the `hold` motor positions, then `enable`."""
        with self._lock:
            self.preload(hold)
            self.enable()

    def preload(self, hold: np.ndarray) -> None:
        """Command, with torque still off, a stiff hold of the arm joints at the `hold` motor positions and a limp
        gripper; every motor must confirm. A link lost during the enable then leaves the arm holding.

        Raises:
            ConnectionError: If a motor doesn't confirm.
        """
        with self._lock:
            self._pending.clear()
            commands = {
                name: MitCommand(float(value), 0.0, float(kp), float(kd), 0.0)
                for name, value, kp, kd in zip(
                    self.motor_names[:-1], hold[:-1], self.hold_kp, self.hold_kd, strict=True
                )
            }
            commands[self.motor_names[-1]] = MitCommand(float(hold[-1]), 0.0, 0.0, 0.0, 0.0)
            if unconfirmed := self.send_confirmed(commands, enabled=False):
                raise ConnectionError(
                    f"{self.port}: {', '.join(unconfirmed)} did not confirm the hold before enabling"
                )

    def enable(self) -> None:
        """Enable torque and wait for every motor to confirm.

        Raises:
            ConnectionError: If a motor doesn't confirm.
        """
        with self._lock:
            # Set before enabling, so disable_torque also turns off motors enabled by a failed enable.
            self.enabled = True
            sent_at = self._drain()
            self.bus.enable_torque()
            self._confirm_torque(True, sent_at)
            self._enabled_at = time.monotonic()

    def send_mit(self, motor: str, command: MitCommand) -> None:
        """Buffer one MIT command, and send the batch once the gripper's command arrives.

        Raises:
            ValueError: If a command of the batch has a NaN or infinite field; nothing is sent then.
        """
        with self._lock:
            self._pending[motor] = command
            if motor != self.motor_names[-1]:
                return
            commands, self._pending = self._pending, {}
            self._send_batch(commands)

    def send_confirmed(self, commands: dict[str, MitCommand], enabled: bool) -> list[str]:
        """Send a batch of commands and return the motors that didn't reply to it with torque `enabled`."""
        with self._lock:
            self._pending.clear()
            self._drain()
            replied = self._send_batch(commands)
            expected = _STATUS_ENABLED if enabled else _STATUS_DISABLED
            return [name for name in commands if name not in replied or self._status[name] != expected]

    def disable_torque(self) -> None:
        """Disable torque if enabled, once every motor confirms it. Callers decide whether the arm may drop.

        Raises:
            ConnectionError: If a motor doesn't confirm; `enabled` then stays `True`.
        """
        with self._lock:
            self._pending.clear()
            if not self.enabled or not self.bus.is_connected:
                return
            sent_at = self._drain()
            self.bus.disable_torque()
            self._confirm_torque(False, sent_at)
            self.enabled = False

    def disconnect(self, release: bool = True) -> None:
        """Close the CAN interface, first disabling torque if `release`; CAN closes even if disabling fails.

        Args:
            release (`bool`, *optional*, defaults to `True`):
                Disable torque first. `False` leaves the motors on their last command.
        """
        with self._lock:
            try:
                if release:
                    self.disable_torque()
            finally:
                if self.bus.is_connected:
                    self.bus.disconnect(disable_torque=False)

    def _send(self, arbitration_id: int, data: list[int]) -> None:
        if self.bus.canbus is None:
            raise RuntimeError(f"{self.port}: CAN bus is not connected")
        self.bus.canbus.send(can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False))

    def _collect(self, names: list[str], timeout_s: float) -> set[str]:
        """Collect the replies of `names`, record when each arrived and its status, and update the bus cache."""
        recv_ids = {
            motor_id + FEEDBACK_ID_OFFSET: name
            for motor_id, name in zip(MOTOR_IDS, self.motor_names, strict=True)
            if name in names
        }
        replies = self.bus._recv_all_responses(list(recv_ids), timeout=timeout_s)
        now = time.monotonic()
        valid = set()
        for recv_id, msg in replies.items():
            name = recv_ids[recv_id]
            data = bytes(msg.data)
            # A state frame: eight bytes, the motor's ID in the low nibble and a known status in the high one.
            if (
                len(data) != 8
                or data[0] & 0x0F != recv_id - FEEDBACK_ID_OFFSET
                or data[0] >> 4 not in _STATUS_KNOWN
            ):
                continue
            # The bus logs and drops a frame it can't decode; it only counts once it decodes to finite values.
            motor_type = MotorType[MOTOR_TYPES[self.motor_names.index(name)].upper()]
            try:
                decoded = self.bus._decode_motor_state(data, motor_type)
            except ValueError:
                continue
            if not all(math.isfinite(value) for value in decoded):
                continue
            self._replied_at[name] = now
            self._replies[name] += 1
            self._status[name] = data[0] >> 4
            self.bus._process_response(name, msg)
            valid.add(name)
        return valid

    def _refresh(self) -> None:
        for motor_id in MOTOR_IDS:
            self._send(CAN_PARAM_ID, [motor_id, 0, CAN_CMD_REFRESH, 0, 0, 0, 0, 0])
        self._collect(list(self.motor_names), MEDIUM_TIMEOUT_SEC)

    def _refresh_until_fresh(self) -> None:
        """Refresh until every motor answered within `feedback_timeout_s`, without a fault."""
        deadline = time.monotonic() + self.feedback_timeout_s
        while True:
            self._refresh()
            now = time.monotonic()
            for motor_id, name in zip(MOTOR_IDS, self.motor_names, strict=True):
                if self._status[name] not in _STATUS_OK:
                    raise ConnectionError(f"{self.port}: motor {motor_id} fault {self._status[name]:#x}")
            ages = [now - self._replied_at[name] for name in self.motor_names]
            if all(age <= self.feedback_timeout_s for age in ages):
                return
            if now >= deadline:
                details = ", ".join(
                    f"{motor_id}: missing" if math.isinf(age) else f"{motor_id}: {age * 1000:.1f} ms"
                    for motor_id, age in zip(MOTOR_IDS, ages, strict=True)
                )
                raise ConnectionError(f"{self.port}: missing or stale motor feedback ({details})")

    def _send_batch(self, commands: dict[str, MitCommand]) -> set[str]:
        """Send MIT commands in one batch, after checking every field is finite; return the motors that replied."""
        for name, c in commands.items():
            if not all(math.isfinite(v) for v in (c.position, c.velocity, c.kp, c.kd, c.torque)):
                raise ValueError(f"{self.port}: non-finite MIT command for {name}: {c}")
        for name, c in commands.items():
            index = self.motor_names.index(name)
            motor_type = MotorType[MOTOR_TYPES[index].upper()]
            data = self.bus._encode_mit_packet(
                motor_type, c.kp, c.kd, math.degrees(c.position), math.degrees(c.velocity), c.torque
            )
            self._send(MOTOR_IDS[index], data)
        return self._collect(list(commands), SHORT_TIMEOUT_SEC)

    def _drain(self) -> float:
        """Discard queued replies until the bus is quiet, and return the time after which replies are new.

        Raises:
            ConnectionError: If the bus doesn't go quiet within `feedback_timeout_s`.
        """
        canbus = self.bus.canbus
        deadline = time.monotonic() + self.feedback_timeout_s
        while canbus is not None and canbus.recv(timeout=_QUIET_S) is not None:
            if time.monotonic() >= deadline:
                raise ConnectionError(f"{self.port}: the bus didn't go quiet; is another program sending?")
        return time.monotonic()

    def _confirm_torque(self, enabled: bool, sent_at: float) -> None:
        """Refresh until every motor has replied since `sent_at` with torque `enabled`; a fault counts as off.

        Raises:
            ConnectionError: If a motor doesn't confirm within `feedback_timeout_s`.
        """
        deadline = time.monotonic() + self.feedback_timeout_s
        while True:
            self._refresh()
            pending = [
                motor_id
                for motor_id, name in zip(MOTOR_IDS, self.motor_names, strict=True)
                if self._replied_at[name] < sent_at or (self._status[name] == _STATUS_ENABLED) != enabled
            ]
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise ConnectionError(
                    f"{self.port}: motors {pending} did not confirm torque {'on' if enabled else 'off'}"
                )

    def _states(self) -> MotorStates:
        states = self.bus._last_known_states
        return MotorStates(
            position=np.radians([states[name]["position"] for name in self.motor_names]),
            velocity=np.radians([states[name]["velocity"] for name in self.motor_names]),
            torque=np.asarray([states[name]["torque"] for name in self.motor_names], dtype=float),
        )
