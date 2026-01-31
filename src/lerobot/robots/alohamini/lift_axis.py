from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class BusLike(Protocol):
    motors: dict[str, object]

    def read(self, item: str, name: str) -> float: ...
    def write(self, item: str, name: str, value: float) -> None: ...
    def sync_write(self, item: str, values: dict[str, float]) -> None: ...


from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import OperatingMode


@dataclass
class LiftAxisConfig:
    enabled: bool = True
    name: str = "lift_axis"
    bus: str = "left"  # "left" or "right" (select which existing bus to use)
    motor_id: int = 11
    motor_model: str = "sts3215"

    # Mechanical conversion (1 rev = 360° = 4096 ticks); adjust for your leadscrew/gear ratio
    lead_mm_per_rev: float = 84  # Lead screw pitch (mm per revolution)
    output_gear_ratio: float = 1.0  # Servo angle → lead screw angle transmission ratio
    soft_min_mm: float = 0.0
    soft_max_mm: float = 600  # Lift travel range

    # Homing (drive downward to hard stop → rebound slightly)
    home_down_speed: int = 1300  # Downward target velocity in velocity mode
    home_stall_current_ma: int = 150  # Stall current threshold; used when no current feedback
    home_backoff_deg: float = 5.0

    # Velocity closed-loop gains
    kp_vel: float = 300  # (target speed units / mm)
    v_max: int = 1300  # Velocity limit (depends on motor)
    on_target_mm: int = 1.0  # Position tolerance (mm)

    dir_sign: int = -1  # +1 no inversion; -1 invert direction
    step_mm: float = 2  # Step per key press (mm)


class LiftAxis:
    """Z-axis controller merged into existing left/right bus (velocity mode + multi-turn counter + mm-level closed loop)"""

    def __init__(
        self,
        cfg: LiftAxisConfig,
        bus_left: BusLike | None,
        bus_right: BusLike | None,
    ):
        self.cfg = cfg
        self._bus = bus_left if cfg.bus == "left" else bus_right
        self.enabled = bool(cfg.enabled and self._bus is not None)
        self._ticks_per_rev = 4096.0
        self._deg_per_tick = 360.0 / self._ticks_per_rev
        self._mm_per_deg = (cfg.lead_mm_per_rev * cfg.output_gear_ratio) / 360.0

        # Multi-turn tick tracking
        self._last_tick: float = 0.0
        self._extended_ticks: float = 0.0  # cumulative total
        # Zero reference (extended angle)
        self._z0_deg: float = 0.0

        self._configured = False

    def attach(self) -> None:
        if not self.enabled:
            return
        if self.cfg.name not in self._bus.motors:
            self._bus.motors[self.cfg.name] = Motor(
                self.cfg.motor_id, self.cfg.motor_model, MotorNormMode.DEGREES
            )

    def configure(self) -> None:
        if not self.enabled:
            return
        if self._configured:
            return
        self._bus.write("Operating_Mode", self.cfg.name, OperatingMode.VELOCITY.value)
        self._last_tick = float(self._bus.read("Present_Position", self.cfg.name, normalize=False))
        self._extended_ticks = 0.0
        self._configured = True

    def _update_extended_ticks(self) -> None:
        if not self.enabled:
            return
        cur = float(self._bus.read("Present_Position", self.cfg.name, normalize=False))  # 0..4095
        delta = cur - self._last_tick
        half = self._ticks_per_rev * 0.5
        if delta > +half:
            delta -= self._ticks_per_rev
        elif delta < -half:
            delta += self._ticks_per_rev
        self._extended_ticks += delta
        self._last_tick = cur

    def _extended_deg(self) -> float:
        return self.cfg.dir_sign * self._extended_ticks * self._deg_per_tick

    def get_height_mm(self) -> float:
        if not self.enabled:
            return 0.0
        self._update_extended_ticks()
        raw_mm = (self._extended_deg() - self._z0_deg) * self._mm_per_deg
        # print(f"[lift_axis.get_height_mm] raw_mm={raw_mm:.2f}, extended_deg={self._extended_deg():.2f}, z0_deg={self._z0_deg:.2f}")  # debug
        return raw_mm

    # Homing (down to hard stop → rebound, set z=0mm)
    def home(self, use_current: bool = True) -> None:
        if not self.enabled:
            return
        self.configure()
        name = self.cfg.name
        # Move downward
        v_down = self.cfg.home_down_speed
        self._bus.write("Goal_Velocity", name, v_down)
        stuck = 0
        last_tick = int(self._bus.read("Present_Position", name, normalize=False))
        for _ in range(600):  # ~30s @50ms
            time.sleep(0.05)
            self._update_extended_ticks()
            now_tick = self._last_tick
            moved = abs(now_tick - last_tick) > 10
            last_tick = now_tick
            cur_ma = 0
            raw_cur_ma = 0
            if use_current:
                try:
                    raw_cur_ma = int(self._bus.read("Present_Current", name, normalize=False))
                    cur_ma = raw_cur_ma * 6.5
                    print(f"[lift_axis.home] Present_Current={cur_ma} mA")  # debug
                    print(f"[lift_axis.home] Present_Position={now_tick} ticks")  # debug

                except Exception:
                    cur_ma = 0
            if (use_current and cur_ma >= self.cfg.home_stall_current_ma) or (not moved):
                print(f"[lift_axis.home] Stalled at current={cur_ma} mA, moved={moved}")  # debug
                stuck += 1
            else:
                stuck = 0
            if stuck >= 2:
                break
        # self._bus.write("Goal_Velocity", name, 0)
        self._bus.write("Torque_Enable", name, 0)
        print("Disable torque output (motor will be released)")
        time.sleep(1)

        self._update_extended_ticks()
        self._z0_deg = self._extended_deg()
        print("Extended ticks after homing:", self._extended_ticks)
        h_now = self.get_height_mm()
        print(f"[home] set-zero z0_deg={self._z0_deg:.2f}, height_now={h_now:.2f} mm")  # should be ~0 here

    # Lightweight coupling with action/obs
    def contribute_observation(self, obs: dict[str, float]) -> None:
        """Export convenient observation fields: height_mm and velocity"""
        if not self.enabled:
            return
        obs[f"{self.cfg.name}.height_mm"] = self.get_height_mm()
        try:
            obs[f"{self.cfg.name}.vel"] = int(
                self._bus.read("Present_Velocity", self.cfg.name, normalize=False)
            )
        except Exception:
            pass

    def apply_action(self, action: dict[str, float]) -> None:
        """
        Supports two action keys:
        - f"{name}.height_mm": target height (mm)  (recommended)
        - f"{name}.vel"      : target velocity     (advanced)
        """
        # print(f"[lift_axis.apply_action] action={action}")  # debug
        if not self.enabled:
            return
        key_h = f"{self.cfg.name}.height_mm"
        key_v = f"{self.cfg.name}.vel"
        if key_h in action:
            target_mm = float(action[key_h])
            cur_mm = self.get_height_mm()
            err = target_mm - cur_mm
            if abs(err) <= self.cfg.on_target_mm:
                v_cmd = 0
            else:
                v_cmd = self.cfg.kp_vel * err
                if v_cmd > self.cfg.v_max:
                    v_cmd = self.cfg.v_max
                elif v_cmd < -self.cfg.v_max:
                    v_cmd = -self.cfg.v_max
            # Limit if already at boundary
            if (cur_mm >= self.cfg.soft_max_mm and v_cmd > 0) or (
                cur_mm <= self.cfg.soft_min_mm and v_cmd < 0
            ):
                v_cmd = 0
            self._bus.write("Goal_Velocity", self.cfg.name, int(self.cfg.dir_sign * v_cmd))
        if key_v in action:
            # Direct velocity clears height target
            v = int(action[key_v])
            v = max(-self.cfg.v_max, min(self.cfg.v_max, v))
            # Limit if already at boundary
            try:
                cur_mm = self.get_height_mm()
                if (cur_mm >= self.cfg.soft_max_mm and v > 0) or (cur_mm <= self.cfg.soft_min_mm and v < 0):
                    v = 0
            except Exception:
                pass
            self._bus.write("Goal_Velocity", self.cfg.name, v * self.cfg.dir_sign)

        # ticks = int(self._bus.read("Present_Position", self.cfg.name, normalize=False))
        # print(f"[lift_axis] Z-axis ticks: {ticks}")
        # print(f"[lift_axis] Z-axis height: {self.get_height_mm():.2f} mm")
