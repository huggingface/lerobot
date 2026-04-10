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

"""Feetech auto-calibration and WritePosEx utilities (measure mechanical range, default range, single write position+speed+acceleration)."""

import contextlib
import logging
import time

import scservo_sdk as scs

from ..motors_bus import NameOrID
from .calibration_defaults import (
    DEFAULT_ACCELERATION,
    DEFAULT_POS_SPEED,
    DEFAULT_TIMEOUT,
    FULL_TURN,
    HOMING_OFFSET_MAX_MAG,
    MID_POS,
    OVERLOAD_SETTLE_TIME,
    SAFE_IO_INTERVAL,
    SAFE_IO_RETRIES,
    SO_STS_DEFAULT_RANGES,
    STALL_POSITION_DELTA_THRESHOLD,
    STALL_VELOCITY_THRESHOLD,
    UNFOLD_OVERLOAD_SETTLE,
    UNFOLD_TOLERANCE_DEG,
    motor_label,
)

COMM_ERR = (RuntimeError, ConnectionError)
"""Exception types that may be raised during servo communication: RuntimeError (Overload) and ConnectionError (no status packet)."""

logger = logging.getLogger(__name__)


class FeetechCalibrationMixin:
    """Provides auto mechanical range measurement, default range, write_pos_ex_and_wait/wait_until_stopped."""

    # Single instruction writes Acceleration(41) + Goal_Position(42-43) + Goal_Time(44-45) + Goal_Velocity(46-47), 7 bytes total (matches STServo WritePosEx)
    _POS_EX_START_ADDR = 41
    _POS_EX_LEN = 7
    # Min_Position_Limit(9,2) + Max_Position_Limit(11,2) contiguous 4 bytes, writable in a single instruction
    _POS_LIMITS_START_ADDR = 9
    _POS_LIMITS_LEN = 4

    def get_default_range(self, motor: NameOrID) -> tuple[int, int]:
        """Return the default (range_min, range_max) used during auto-calibration for the specified motor.

        If the motor name is found in calibration_defaults.SO_STS_DEFAULT_RANGES, uses the preset values;
        otherwise uses the full range (0, max_res) for that model's resolution.
        """
        motor_names = self._get_motors_list(motor)
        name = motor_names[0]
        if name in SO_STS_DEFAULT_RANGES:
            return SO_STS_DEFAULT_RANGES[name]
        model = self._get_motor_model(motor)
        max_res = self.model_resolution_table[model] - 1
        return (0, max_res)

    def _wait_for_stall(
        self,
        motor: str,
        stall_confirm_samples: int,
        timeout_s: float,
        sample_interval_s: float,
        *,
        velocity_threshold: int = STALL_VELOCITY_THRESHOLD,
        position_delta_threshold: int = STALL_POSITION_DELTA_THRESHOLD,
    ) -> str:
        """Poll for stall/limit detection, return a stop-reason string.

        Prioritizes the AND condition (near-zero velocity + stable position + Moving=0), returning immediately if met;
        otherwise checks Status register BIT5 (overload) or communication error, requiring stall_confirm_samples consecutive hits.
        """
        stall_count = 0
        stable_count = 0
        prev_position: int | None = None
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            try:
                vel = self.read("Present_Velocity", motor, normalize=False)
                pos = self.read("Present_Position", motor, normalize=False)
                moving = self.read("Moving", motor, normalize=False)
                status = self.read("Status", motor, normalize=False)
            except COMM_ERR:
                stable_count = 0
                stall_count += 1
                if stall_count >= stall_confirm_samples:
                    return f"stall confirmed({stall_confirm_samples}x): communication error"
                time.sleep(sample_interval_s)
                continue

            # Priority: near-zero velocity AND stable position AND Moving=0 (N consecutive times)
            vel_ok = abs(vel) < velocity_threshold
            pos_ok = prev_position is None or abs(pos - prev_position) < position_delta_threshold
            moving_ok = moving == 0
            if vel_ok and pos_ok and moving_ok:
                stable_count += 1
                if stable_count >= stall_confirm_samples:
                    return (
                        f"limit confirmed({stall_confirm_samples}x): "
                        "near-zero velocity + stable position + Moving=0"
                    )
            else:
                stable_count = 0
            prev_position = pos

            # Secondary: Status BIT5 overload
            if status & 0x20:
                stall_count += 1
                if stall_count >= stall_confirm_samples:
                    return f"stall confirmed({stall_confirm_samples}x): Status=0x{status:02X}(BIT5 overload)"
            else:
                stall_count = 0

            time.sleep(sample_interval_s)
        return f"timeout({timeout_s}s)"

    def _wait_for_stall_multi(
        self,
        motors: list[str],
        stall_confirm_samples: int,
        timeout_s: float,
        sample_interval_s: float,
        *,
        velocity_threshold: int = STALL_VELOCITY_THRESHOLD,
        position_delta_threshold: int = STALL_POSITION_DELTA_THRESHOLD,
    ) -> tuple[dict[str, str], dict[str, int]]:
        """Multi-motor stall polling: monitor all simultaneously, write Goal_Velocity=0 to each as it stalls, return after all stalled or timeout.

        Returns (reasons, positions): stop reason and stall Present_Position for each motor.
        """
        still_running = set(motors)
        reasons: dict[str, str] = {}
        positions: dict[str, int] = {}
        # Per-motor state: previous position, stable count, overload count
        prev_pos: dict[str, int | None] = dict.fromkeys(motors)
        stable_count: dict[str, int] = dict.fromkeys(motors, 0)
        stall_count: dict[str, int] = dict.fromkeys(motors, 0)
        t0 = time.monotonic()

        while still_running and (time.monotonic() - t0 < timeout_s):
            for m in list(still_running):
                try:
                    vel = self.read("Present_Velocity", m, normalize=False)
                    pos = self.read("Present_Position", m, normalize=False)
                    moving = self.read("Moving", m, normalize=False)
                    status = self.read("Status", m, normalize=False)
                except COMM_ERR:
                    stall_count[m] = stall_count.get(m, 0) + 1
                    if stall_count[m] >= stall_confirm_samples:
                        reasons[m] = f"stall confirmed({stall_confirm_samples}x): communication error"
                        positions[m] = self._read_with_retry("Present_Position", m)
                        self.write("Goal_Velocity", m, 0)
                        still_running.discard(m)
                    continue

                vel_ok = abs(vel) < velocity_threshold
                pos_ok = prev_pos[m] is None or abs(pos - prev_pos[m]) < position_delta_threshold
                moving_ok = moving == 0
                if vel_ok and pos_ok and moving_ok:
                    stable_count[m] = stable_count.get(m, 0) + 1
                    if stable_count[m] >= stall_confirm_samples:
                        reasons[m] = (
                            f"limit confirmed({stall_confirm_samples}x): "
                            "near-zero velocity + stable position + Moving=0"
                        )
                        positions[m] = pos
                        self.write("Goal_Velocity", m, 0)
                        still_running.discard(m)
                        continue
                else:
                    stable_count[m] = 0
                prev_pos[m] = pos

                if status & 0x20:
                    stall_count[m] = stall_count.get(m, 0) + 1
                    if stall_count[m] >= stall_confirm_samples:
                        reasons[m] = (
                            f"stall confirmed({stall_confirm_samples}x): Status=0x{status:02X}(BIT5 overload)"
                        )
                        positions[m] = pos
                        self.write("Goal_Velocity", m, 0)
                        still_running.discard(m)
                else:
                    stall_count[m] = 0

            time.sleep(sample_interval_s)

        for m in still_running:
            reasons[m] = f"timeout({timeout_s}s)"
            try:
                positions[m] = self.read("Present_Position", m, normalize=False)
            except COMM_ERR:
                positions[m] = 0
            with contextlib.suppress(COMM_ERR):
                self.write("Goal_Velocity", m, 0)
        return reasons, positions

    def _prepare_motors_for_range_measure(self, motors: list[str]) -> None:
        """Prepare motors for range measurement: clear overload, disable torque, set Phase(BIT4=0), Homing_Offset=0, velocity mode, enable torque."""
        from .feetech import OperatingMode

        for m in motors:
            self._safe_stop_and_clear_overload(m)
        #  self.disable_torque(m)
        for m in motors:
            phase_raw = self.read("Phase", m, normalize=False)
            if phase_raw & 0x10:
                self.write("Phase", m, phase_raw & ~0x10, normalize=False)
                print(
                    f"  [{motor_label(m)}] Phase(reg18): 0x{phase_raw:02X} -> 0x{phase_raw & ~0x10:02X} (BIT4=0 single-turn)"
                )
            else:
                print(f"  [{motor_label(m)}] Phase(reg18): 0x{phase_raw:02X} (already single-turn)")
            self.write("Homing_Offset", m, 0, normalize=False)
            self.write("Operating_Mode", m, OperatingMode.VELOCITY.value)
            mode = self.read("Operating_Mode", m, normalize=False)
            if len(motors) == 1:
                print(f"  [{motor_label(m)}] Operating_Mode={mode} (expected 1, velocity mode)")
            self.enable_torque(m)
        if motors:
            time.sleep(0.1)

    def _run_direction_until_stall(
        self,
        motors: list[str],
        velocity: int | dict[str, int],
        *,
        stall_confirm_samples: int = 2,
        timeout_s: float = 10.0,
        sample_interval_s: float = 0.05,
        initial_move_delay_s: float = 0.5,
    ) -> tuple[dict[str, str], dict[str, int]]:
        """Start specified motors with a single command, poll for stall then stop and clear overload. velocity can be a single int or per-motor dict. Returns (per-motor stop reason, per-motor stall position)."""
        vel_dict = velocity if isinstance(velocity, dict) else dict.fromkeys(motors, velocity)
        self.sync_write(
            "Goal_Velocity",
            vel_dict,
            normalize=False,
        )
        time.sleep(initial_move_delay_s)
        if len(motors) == 1:
            m = motors[0]
            reason = self._wait_for_stall(m, stall_confirm_samples, timeout_s, sample_interval_s)
            reasons = {m: reason}
            positions = {m: self._read_with_retry("Present_Position", m)}
        else:
            reasons, positions = self._wait_for_stall_multi(
                motors, stall_confirm_samples, timeout_s, sample_interval_s
            )
        # for m in motors:
        #     self._safe_stop_and_clear_overload(m)
        return reasons, positions

    def _compute_mid_and_range_from_limits(
        self,
        motor: str,
        pos_cw: int,
        pos_ccw: int,
        *,
        move_timeout: float = 5.0,
        reference_pos: int | None = None,
    ) -> tuple[int, int, int, int, int, int]:
        """From CW/CCW stall positions, perform backoff, compute physical midpoint and range. Returns (range_min, range_max, mid, raw_min, raw_max, homing_offset).
        If reference_pos is provided (e.g. (Present_Position+Homing_Offset)%FULL_TURN sampled during lift), skips backoff and uses it directly for arc selection."""
        arc_ccw_to_cw = (pos_cw - pos_ccw) % FULL_TURN
        arc_cw_to_ccw = (pos_ccw - pos_cw) % FULL_TURN
        if reference_pos is not None:
            start_pos = reference_pos
            print(
                f"  [{motor_label(motor)}] Using pre-sampled reference position start_pos={start_pos}, skipping limit backoff"
            )
        else:
            print("Probing reference position...")
            shortest_arc = min(arc_ccw_to_cw, arc_cw_to_ccw)
            steps_back = max(1, shortest_arc // 3)
            back_deg = steps_back * 360.0 / FULL_TURN
            print(
                f"  [{motor_label(motor)}] long arc: {max(arc_ccw_to_cw, arc_cw_to_ccw)} steps, "
                f"short arc: {min(arc_ccw_to_cw, arc_cw_to_ccw)} steps, "
                f"backoff: {steps_back} steps ({back_deg:.1f}°)"
            )
            self.unfold_single_joint(motor, back_deg, move_timeout=move_timeout)
            time.sleep(0.1)
            present_raw = self._read_with_retry("Present_Position", motor)
            homing_raw = self._read_with_retry("Homing_Offset", motor)
            start_pos = (present_raw + homing_raw) % FULL_TURN
            print(
                f"  [{motor_label(motor)}] Backed off {steps_back} steps ({back_deg:.1f}°) from limit, "
                f"reference position: present={present_raw}, offset={homing_raw}, actual={start_pos}"
            )
        arc_ccw_to_cw = (pos_cw - pos_ccw) % FULL_TURN
        arc_cw_to_ccw = (pos_ccw - pos_cw) % FULL_TURN
        start_in_arc_a = (start_pos - pos_ccw) % FULL_TURN <= arc_ccw_to_cw
        if start_in_arc_a:
            physical_range = arc_ccw_to_cw
            mid = (pos_ccw + physical_range // 2) % FULL_TURN
        else:
            physical_range = arc_cw_to_ccw
            mid = (pos_cw + physical_range // 2) % FULL_TURN
        raw_min = min(pos_cw, pos_ccw)
        raw_max = max(pos_cw, pos_ccw)
        homing_offset = mid - MID_POS
        homing_offset = max(
            -HOMING_OFFSET_MAX_MAG,
            min(HOMING_OFFSET_MAX_MAG, homing_offset),
        )
        half = physical_range // 2
        range_min = max(0, min(FULL_TURN - 1, MID_POS - half))
        range_max = max(0, min(FULL_TURN - 1, MID_POS + half))
        crosses_zero = pos_ccw > pos_cw
        print(
            f"  [{motor_label(motor)}] CW={pos_cw}, CCW={pos_ccw}, computed reference={start_pos}, "
            f"range steps={physical_range}  ({physical_range * 360 / FULL_TURN:.1f}°), "
            f"physical midpoint={mid}, crosses zero={crosses_zero}"
        )
        return range_min, range_max, mid, raw_min, raw_max, homing_offset

    def _safe_stop_and_clear_overload(self, motor: str, settle_s: float = 0.5) -> None:
        """Safe stop after stall: write Goal_Velocity=0, disable torque, wait for overload/communication error to clear."""
        for _ in range(5):
            try:
                self.write("Goal_Velocity", motor, 0)
                break
            except COMM_ERR:
                time.sleep(0.1)
        for _ in range(5):
            try:
                self.disable_torque(motor)
                break
            except COMM_ERR:
                time.sleep(0.1)
        time.sleep(settle_s)

    def _read_with_retry(self, data_name: str, motor: str, retries: int = 5, interval_s: float = 0.2) -> int:
        """Read with retries, used for reading registers during overload/communication error recovery."""
        for i in range(retries):
            try:
                return self.read(data_name, motor, normalize=False)
            except COMM_ERR as e:
                if i < retries - 1:
                    time.sleep(interval_s)
                    continue
                raise RuntimeError(
                    f"_read_with_retry: all {retries} attempts failed for {data_name} on {motor}: {e}"
                ) from e
        raise RuntimeError(f"_read_with_retry: unable to read {data_name} on {motor}")

    def safe_read(
        self,
        reg: str,
        motor: NameOrID,
        *,
        retries: int = SAFE_IO_RETRIES,
        interval_s: float = SAFE_IO_INTERVAL,
    ) -> int:
        """Safe read with retries."""
        return self._read_with_retry(
            reg, self._get_motors_list(motor)[0], retries=retries, interval_s=interval_s
        )

    def safe_write(
        self,
        reg: str,
        motor: NameOrID,
        value: int,
        *,
        normalize: bool = True,
        retries: int = SAFE_IO_RETRIES,
        interval_s: float = SAFE_IO_INTERVAL,
    ) -> None:
        """Safe write with retries."""
        motor_name = self._get_motors_list(motor)[0]
        for i in range(retries):
            try:
                self.write(reg, motor_name, value, normalize=normalize)
                return
            except COMM_ERR as e:
                if i < retries - 1:
                    time.sleep(interval_s)
                    continue
                raise RuntimeError(
                    f"safe_write: all {retries} attempts failed for {reg}={value} on {motor_name}: {e}"
                ) from e
        raise RuntimeError(f"safe_write: unable to write {reg} on {motor_name}")

    def _write_torque_with_recovery(
        self, motor: str, value: int, retries: int = 3, interval_s: float = 0.5
    ) -> None:
        """Write Torque_Enable with type-specific error recovery and retry.

        - RuntimeError (Overload): disable torque to clear overload status, wait, then retry.
        - ConnectionError (no response): wait and retry directly.
        Raises if all retries are exhausted.
        """
        for attempt in range(retries):
            try:
                self.write("Torque_Enable", motor, value)
                return
            except RuntimeError as e:
                # Overload: disable torque to clear, wait, then retry
                if attempt < retries - 1:
                    print(
                        f"  [{motor_label(motor)}] Torque_Enable={value} Overload, clearing and retrying "
                        f"({attempt + 1}/{retries}): {e}"
                    )
                    with contextlib.suppress(COMM_ERR):
                        self.write("Torque_Enable", motor, 0)
                    time.sleep(interval_s)
                else:
                    raise RuntimeError(
                        f"_write_torque_with_recovery: all {retries} attempts failed for Torque_Enable={value} on {motor}: {e}"
                    ) from e
            except ConnectionError as e:
                # No response: wait and retry
                if attempt < retries - 1:
                    print(
                        f"  [{motor_label(motor)}] Torque_Enable={value} no response, waiting and retrying "
                        f"({attempt + 1}/{retries}): {e}"
                    )
                    time.sleep(interval_s)
                else:
                    raise RuntimeError(
                        f"_write_torque_with_recovery: all {retries} attempts failed for Torque_Enable={value} on {motor}: {e}"
                    ) from e

    def _clear_and_enable_torque(self, motor: str, settle_s: float = OVERLOAD_SETTLE_TIME) -> None:
        """Clear overload and re-enable torque: disable torque and wait, then enable torque with recovery retry.

        Used after stall stop when reverse motion is needed, replacing bare enable_torque calls.
        """
        # Disable torque to clear overload status (with retry logic)
        for _ in range(5):
            try:
                self.write("Torque_Enable", motor, 0)
                break
            except COMM_ERR:
                time.sleep(0.1)
        time.sleep(settle_s)
        # Enable torque with recovery retry
        self._write_torque_with_recovery(motor, 1)
        with contextlib.suppress(COMM_ERR):
            self.write("Lock", motor, 1)

    def safe_write_position_limits(
        self,
        motor: NameOrID,
        rmin: int,
        rmax: int,
        *,
        retries: int = SAFE_IO_RETRIES,
        interval_s: float = SAFE_IO_INTERVAL,
    ) -> None:
        """Safe write of Min/Max position limits with retries (single instruction)."""
        motor_name = self._get_motors_list(motor)[0]
        for i in range(retries):
            try:
                self.write_position_limits(motor_name, rmin, rmax)
                return
            except COMM_ERR as e:
                if i < retries - 1:
                    time.sleep(interval_s)
                    continue
                raise RuntimeError(
                    f"safe_write_position_limits: all {retries} attempts failed for rmin={rmin} rmax={rmax} on {motor_name}: {e}"
                ) from e
        raise RuntimeError(f"safe_write_position_limits: unable to write on {motor_name}")

    def safe_disable_all(
        self,
        motor_names: list[str] | None = None,
        *,
        num_try_per_motor: int = 3,
        interval_s: float = 0.1,
    ) -> None:
        """Safely disable torque on all motors, ignoring communication errors."""
        names = motor_names if motor_names is not None else list(self.motors.keys())
        for m in names:
            for _ in range(num_try_per_motor):
                try:
                    self.write("Torque_Enable", m, 0)
                    break
                except COMM_ERR:
                    time.sleep(interval_s)

    def go_to_mid(
        self,
        motor: NameOrID,
        *,
        timeout_s: float = DEFAULT_TIMEOUT,
        poll_interval_s: float = 0.05,
    ) -> bool:
        """Move motor to midpoint (servo mode), wait until reached. Returns True if reached, False if timed out."""
        motor_name = self._get_motors_list(motor)[0]
        ok = self.write_pos_ex_and_wait(
            motor_name,
            MID_POS,
            DEFAULT_POS_SPEED,
            DEFAULT_ACCELERATION,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
        )
        try:
            cur = self.read("Present_Position", motor_name, normalize=False)
        except COMM_ERR:
            cur = -1
        if not ok:
            logger.warning(
                "%s return to mid timed out (%.1fs), current position=%s",
                motor_label(motor_name),
                timeout_s,
                cur,
            )
        return ok

    def measure_ranges_of_motion(
        self,
        motor: NameOrID,
        *,
        velocity_limit: int = 1000,
        stall_confirm_samples: int = 2,
        timeout_s: float = 10.0,
        sample_interval_s: float = 0.05,
        initial_move_delay_s: float = 0.5,
    ) -> tuple[int, int, int, int, int, int]:
        """Automatically measure the mechanical limit range of a single motor (velocity mode).

        Only one motor may be passed. Torque (Max_Torque_Limit, Torque_Limit) and acceleration (Acceleration)
        should be initialized before calling; this method only drives via Goal_Velocity.

        Procedure:
        1. Set register 18 BIT4=0 (single-turn angle feedback, 0-4095)
        2. Zero out Homing_Offset
        3. Switch to Operating_Mode=1 (velocity mode), drive CW/CCW via Goal_Velocity
        4. Detect stall via Status register BIT5, obtain CW/CCW limits
        5. Independent of start position: back off from CCW limit in the reverse direction by some steps to get reference point P.
           Backoff steps = min(distance to 0, distance to 4095) / 2; if CCW limit is exactly 0 or 4095, use CW limit for calculation.

        Returns:
            (range_min, range_max, mid, raw_min, raw_max, homing_offset) - six integers:
            - range_min, range_max: limits in offset space (0~4095, non-wrapping), for writing Min/Max position limits;
            - mid: measured physical midpoint (raw encoding);
            - raw_min, raw_max: measured extremes (raw encoding, may wrap around zero);
            - homing_offset: offset value mid - MID_POS, for writing Homing_Offset.
        """
        motor = self._get_motors_list(motor)[0]
        self._prepare_motors_for_range_measure([motor])

        cw_reasons, pos_cw_dict = self._run_direction_until_stall(
            [motor],
            velocity_limit,
            stall_confirm_samples=stall_confirm_samples,
            timeout_s=timeout_s,
            sample_interval_s=sample_interval_s,
            initial_move_delay_s=initial_move_delay_s,
        )
        print(f"  [{motor_label(motor)}] CW stop reason: {cw_reasons[motor]}")
        print(f"  [{motor_label(motor)}] CW stall position: {pos_cw_dict[motor]}")

        self._clear_and_enable_torque(motor)
        time.sleep(0.05)
        ccw_reasons, pos_ccw_dict = self._run_direction_until_stall(
            [motor],
            -velocity_limit,
            stall_confirm_samples=stall_confirm_samples,
            timeout_s=timeout_s,
            sample_interval_s=sample_interval_s,
            initial_move_delay_s=initial_move_delay_s,
        )
        print(f"  [{motor_label(motor)}] CCW stop reason: {ccw_reasons[motor]}")
        print(f"  [{motor_label(motor)}] CCW stall position: {pos_ccw_dict[motor]}")

        return self._compute_mid_and_range_from_limits(motor, pos_cw_dict[motor], pos_ccw_dict[motor])

    def measure_ranges_of_motion_multi(
        self,
        motors: list[str],
        *,
        velocity_limit: int = 1000,
        stall_confirm_samples: int = 2,
        timeout_s: float = 10.0,
        sample_interval_s: float = 0.05,
        initial_move_delay_s: float = 0.5,
        ccw_first: bool | dict[str, bool] = False,
        reference_positions: dict[str, int] | None = None,
    ) -> dict[str, tuple[int, int, int, int, int, int]]:
        """Multi-motor simultaneous mechanical limit measurement: start all at once, poll for stall (write 0 to each as it stops), read positions after all stop; compute backoff and midpoint per motor individually.

        ccw_first: if True, that motor runs CCW then CW; if dict, specifies per motor (motor_name -> True/False).
        reference_positions: if a motor's reference position ((Present_Position+Homing_Offset)%FULL_TURN) is provided, skip limit backoff and use it directly for arc selection.
        Returns dict[motor_name, (range_min, range_max, mid, raw_min, raw_max, homing_offset)].
        """
        if not motors:
            return {}
        if len(motors) == 1:
            m = motors[0]
            t = self.measure_ranges_of_motion(
                m,
                velocity_limit=velocity_limit,
                stall_confirm_samples=stall_confirm_samples,
                timeout_s=timeout_s,
                sample_interval_s=sample_interval_s,
                initial_move_delay_s=initial_move_delay_s,
            )
            return {m: t}

        self._prepare_motors_for_range_measure(motors)

        # Per-motor direction: CCW first or CW first
        def _ccw_first(m: str) -> bool:
            return ccw_first.get(m, False) if isinstance(ccw_first, dict) else bool(ccw_first)

        first_vel_dict = {m: (-velocity_limit if _ccw_first(m) else velocity_limit) for m in motors}
        second_vel_dict = {m: (velocity_limit if _ccw_first(m) else -velocity_limit) for m in motors}

        first_reasons, first_pos = self._run_direction_until_stall(
            motors,
            first_vel_dict,
            stall_confirm_samples=stall_confirm_samples,
            timeout_s=timeout_s,
            sample_interval_s=sample_interval_s,
            initial_move_delay_s=initial_move_delay_s,
        )
        for m in motors:
            label = "CCW" if first_vel_dict[m] < 0 else "CW"
            print(
                f"  [{motor_label(m)}] {label} stop reason: {first_reasons[m]}, stall position: {first_pos[m]}"
            )
        for m in motors:
            self._clear_and_enable_torque(m)
        time.sleep(0.05)
        second_reasons, second_pos = self._run_direction_until_stall(
            motors,
            second_vel_dict,
            stall_confirm_samples=stall_confirm_samples,
            timeout_s=timeout_s,
            sample_interval_s=sample_interval_s,
            initial_move_delay_s=initial_move_delay_s,
        )
        for m in motors:
            label = "CCW" if second_vel_dict[m] < 0 else "CW"
            print(
                f"  [{motor_label(m)}] {label} stop reason: {second_reasons[m]}, stall position: {second_pos[m]}"
            )
        time.sleep(OVERLOAD_SETTLE_TIME)

        # Reconstruct pos_cw, pos_ccw per motor based on first/second direction (CW = positive velocity limit, CCW = negative velocity limit)
        pos_cw_dict = {
            m: (first_pos[m] if first_vel_dict[m] == velocity_limit else second_pos[m]) for m in motors
        }
        pos_ccw_dict = {
            m: (second_pos[m] if first_vel_dict[m] == velocity_limit else first_pos[m]) for m in motors
        }

        result: dict[str, tuple[int, int, int, int, int, int]] = {}
        for m in motors:
            ref_pos = reference_positions.get(m) if reference_positions else None
            result[m] = self._compute_mid_and_range_from_limits(
                m, pos_cw_dict[m], pos_ccw_dict[m], reference_pos=ref_pos
            )
        return result

    def _write_raw_bytes(
        self,
        addr: int,
        motor_id: int,
        data: list[int],
        *,
        num_retry: int = 0,
        raise_on_error: bool = True,
        err_msg: str = "",
    ) -> tuple[int, int]:
        """Write raw byte sequence to a register starting address (used by write_pos_ex_and_wait, sync_write_pos_ex, etc.)."""
        for n_try in range(1 + num_retry):
            comm, error = self.packet_handler.writeTxRx(self.port_handler, motor_id, addr, len(data), data)
            if self._is_comm_success(comm):
                break
            logger.debug(
                f"_write_raw_bytes @{addr} len={len(data)} id={motor_id} try={n_try}: "
                + self.packet_handler.getTxRxResult(comm)
            )
        if not self._is_comm_success(comm) and raise_on_error:
            raise ConnectionError(f"{err_msg} {self.packet_handler.getTxRxResult(comm)}")
        if self._is_error(error) and raise_on_error:
            raise RuntimeError(f"{err_msg} {self.packet_handler.getRxPacketError(error)}")
        return comm, error

    def write_position_limits(
        self,
        motor: NameOrID,
        rmin: int,
        rmax: int,
        *,
        num_retry: int = 0,
    ) -> None:
        """Write Min_Position_Limit(9) + Max_Position_Limit(11) in a single instruction, 4 bytes total."""
        id_ = self._get_motor_id(motor)
        data = self._split_into_byte_chunks(rmin, 2) + self._split_into_byte_chunks(rmax, 2)
        err_msg = f"write_position_limits(id={id_}, rmin={rmin}, rmax={rmax}) failed"
        self._write_raw_bytes(
            self._POS_LIMITS_START_ADDR,
            id_,
            data,
            num_retry=num_retry,
            raise_on_error=True,
            err_msg=err_msg,
        )

    def read_position_limits(self, motor: NameOrID) -> tuple[int, int]:
        """Read Min_Position_Limit and Max_Position_Limit, return (rmin, rmax)."""
        rmin = self.read("Min_Position_Limit", motor, normalize=False)
        rmax = self.read("Max_Position_Limit", motor, normalize=False)
        return (rmin, rmax)

    def wait_until_stopped(
        self,
        motor: NameOrID,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
    ) -> bool:
        """Poll the Moving register until it reads 0 or timeout (consistent with STServo read_write example).

        Returns True if stopped (Moving==0), False if timed out or read failed.
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            try:
                moving = self.read("Moving", motor, normalize=False)
            except COMM_ERR:
                logger.debug("wait_until_stopped: communication error reading Moving")
                time.sleep(poll_interval_s)
                continue
            if moving == 0:
                return True
            time.sleep(poll_interval_s)
        return False

    def write_pos_ex_and_wait(
        self,
        motor: NameOrID,
        position: int,
        speed: int,
        acc: int,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
        *,
        num_retry: int = 0,
    ) -> bool:
        """Write position+speed+acceleration in a single instruction, then poll Moving until stopped (consistent with STServo read_write example).

        Ensures servo mode (Operating_Mode=0) first, then writes Goal_Position/Goal_Velocity/Acceleration, then wait_until_stopped.
        Returns True if position reached, False if timed out or write failed.
        """
        try:
            self.write("Operating_Mode", motor, 0)  # Ensure servo mode, otherwise Goal_Position has no effect
            time.sleep(0.05)
            # Single instruction writes goal position, speed, acceleration (7 consecutive bytes from register 41), matching STServo WritePosEx
            id_ = self._get_motor_id(motor)
            pos_enc = self._encode_sign("Goal_Position", {id_: position})[id_]
            speed_enc = self._encode_sign("Goal_Velocity", {id_: speed})[id_]
            data = (
                [acc]
                + self._split_into_byte_chunks(pos_enc, 2)
                + [0, 0]
                + self._split_into_byte_chunks(speed_enc, 2)
            )
            self._write_raw_bytes(
                self._POS_EX_START_ADDR,
                id_,
                data,
                num_retry=num_retry,
                raise_on_error=True,
                err_msg=f"write_pos_ex_and_wait(id={id_}, pos={position}, speed={speed}, acc={acc}) failed",
            )
            time.sleep(0.3)
        except COMM_ERR:
            return False
        result = self.wait_until_stopped(motor, timeout_s=timeout_s, poll_interval_s=poll_interval_s)
        time.sleep(0.1)
        return result

    def sync_write_pos_ex(
        self,
        values: dict[str, tuple[int, int, int]],
        *,
        num_retry: int = 0,
    ) -> None:
        """Multi-motor RegWritePosEx to buffer, then RegAction to execute simultaneously (consistent with STServo reg_write example).

        values: motor_name -> (position, speed, acc). Multiple motors can share the same (position, speed, acc) values.
        """

        for motor_name, (position, speed, acc) in values.items():
            id_ = self._get_motor_id(motor_name)
            self._get_motor_model(motor_name)
            pos_enc = self._encode_sign("Goal_Position", {id_: position})[id_]
            speed_enc = self._encode_sign("Goal_Velocity", {id_: speed})[id_]
            data = (
                [acc]
                + self._split_into_byte_chunks(pos_enc, 2)
                + [0, 0]
                + self._split_into_byte_chunks(speed_enc, 2)
            )
            for _n_try in range(1 + num_retry):
                comm, error = self.packet_handler.regWriteTxRx(
                    self.port_handler, id_, self._POS_EX_START_ADDR, len(data), data
                )
                if self._is_comm_success(comm):
                    break
                logger.debug(
                    f"sync_write_pos_ex RegWrite id={id_}: {self.packet_handler.getTxRxResult(comm)}"
                )
            if self._is_error(error):
                logger.warning(
                    f"sync_write_pos_ex RegWrite id={id_}: {self.packet_handler.getRxPacketError(error)}"
                )
        comm = self.packet_handler.action(self.port_handler, scs.BROADCAST_ID)
        if not self._is_comm_success(comm):
            raise ConnectionError(
                f"sync_write_pos_ex RegAction failed: {self.packet_handler.getTxRxResult(comm)}"
            )

    def _unfold_move_and_wait(
        self,
        motor: str,
        goal: int,
        timeout_s: float,
        tolerance_deg: float = UNFOLD_TOLERANCE_DEG,
    ) -> tuple[bool, int, str]:
        """In servo mode, write goal via WritePosEx, poll Moving until stopped, then check position/Status to determine if reached or stalled. Error within tolerance_deg degrees is considered success (default 5 deg)."""
        goal = max(0, min(goal, FULL_TURN - 1))
        pos_now = self._read_with_retry("Present_Position", motor)
        print(f"    [{motor_label(motor)}] current position={pos_now}, goal position={goal}")
        ok = self.write_pos_ex_and_wait(
            motor,
            goal,
            DEFAULT_POS_SPEED,
            DEFAULT_ACCELERATION,
            timeout_s=timeout_s,
            poll_interval_s=0.05,
        )
        print(f"    [{motor_label(motor)}] move completed: {ok}, checking position")
        time.sleep(0.3)
        try:
            pos = self.read("Present_Position", motor, normalize=False)
        except COMM_ERR:
            self._clear_overload_unfold(motor)
            pos = self._read_with_retry("Present_Position", motor)
            err_deg = abs(pos - goal) * 360.0 / FULL_TURN
            print(
                f"    [{motor_label(motor)}] stall/error stop: pos={pos} ({abs(pos - goal)} steps from goal ≈ {err_deg:.1f}°)"
            )
            return False, pos, "stall(communication error)"
        if not ok:
            self._clear_overload_unfold(motor)
            err_deg = abs(pos - goal) * 360.0 / FULL_TURN
            print(
                f"    [{motor_label(motor)}] timeout: pos={pos} ({abs(pos - goal)} steps from goal ≈ {err_deg:.1f}°)"
            )
            return False, pos, "timeout"
        error_deg = abs(pos - goal) * 360.0 / FULL_TURN
        if error_deg <= tolerance_deg:
            print(
                f"    [{motor_label(motor)}] reached goal: pos={pos} (error {abs(pos - goal)} steps ≈ {error_deg:.1f}°, within {tolerance_deg}°)"
            )
            return True, pos, "reached"
        try:
            status = self.read("Status", motor, normalize=False)
        except COMM_ERR:
            status = 0
        if status & 0x20:
            self._clear_overload_unfold(motor)
            print(
                f"    [{motor_label(motor)}] stall stop: pos={pos} (Status=0x{status:02X} BIT5 overload, "
                f"{abs(pos - goal)} steps from goal ≈ {error_deg:.1f}°)"
            )
            return False, pos, f"stall(Status=0x{status:02X})"
        print(
            f"    [{motor_label(motor)}] stopped short: pos={pos} ({abs(pos - goal)} steps from goal ≈ {error_deg:.1f}°, exceeds {tolerance_deg}°)"
        )
        return False, pos, "not reached"

    def _clear_overload_unfold(self, motor: str) -> None:
        """Disable torque to clear overload status, wait for recovery, then re-enable torque (used by unfold logic)."""
        try:
            self.write("Torque_Enable", motor, 0)
            time.sleep(UNFOLD_OVERLOAD_SETTLE + 0.1)
            self.write("Torque_Enable", motor, 1)
        except COMM_ERR:
            pass

    def unfold_single_joint(
        self,
        motor: str,
        unfold_angle: float,
        move_timeout: float,
    ) -> tuple[str | None, int]:
        """Unfold a single joint: try forward direction first, then reverse if forward fails.

        PID/Acceleration/Operating_Mode=0 are already configured during initialization;
        in servo mode, only Goal_Position needs to be written to drive the motor.

        Returns:
            (direction, steps): direction is "forward"/"reverse", or (None, 0) on failure; steps is the target step count.
        """
        target_steps = int(unfold_angle / 360.0 * FULL_TURN)
        print(f"\n--- Unfold {motor_label(motor)} ({target_steps} steps ≈ {unfold_angle:.1f}°) ---")

        # Calibrate midpoint (Torque_Enable=128 sets current position to 2048)
        self._write_torque_with_recovery(motor, 128)
        self._write_torque_with_recovery(motor, 1)

        time.sleep(0.1)
        print(
            f"[{motor_label(motor)}] Set current position as midpoint: Present_Position={self._read_with_retry('Present_Position', motor)}"
        )
        time.sleep(0.1)
        # Restore servo mode (128 may have changed mode state) and enable torque
        self.write("Operating_Mode", motor, 0)
        self._write_torque_with_recovery(motor, 1)
        time.sleep(0.3)
        # Try forward direction
        # print(f"    [{motor_label(motor)}] Trying forward...")
        reached, pos_after, reason = self._unfold_move_and_wait(motor, MID_POS + target_steps, move_timeout)
        if reached:
            print(f"    [{motor_label(motor)}] forward direction succeeded")
            return "forward", target_steps

        abs(pos_after - MID_POS)
        # print(f"    [{motor_label(motor)}] forward failed ({reason}), moved {moved_fwd} steps")

        # Return to midpoint (after stall, _clear_overload_unfold has re-enabled torque; write midpoint with wait)
        self.write_pos_ex_and_wait(
            motor,
            MID_POS,
            DEFAULT_POS_SPEED,
            DEFAULT_ACCELERATION,
            timeout_s=5.0,
            poll_interval_s=0.05,
        )

        # Try reverse direction
        # print(f"    [{motor_label(motor)}] Trying reverse...")
        reached, pos_after, reason = self._unfold_move_and_wait(motor, MID_POS - target_steps, move_timeout)
        if reached:
            print(f"    [{motor_label(motor)}] reverse direction succeeded")
            return "reverse", target_steps
        abs(MID_POS - pos_after)
        # print(f"    [{motor_label(motor)}] reverse also failed ({reason}), moved {moved_rev} steps, keeping current position")
        print(f"    [{motor_label(motor)}] unfold failed, keeping current position")
        return None, 0
