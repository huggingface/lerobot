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

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from math import isfinite
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from .motor_family import GRIPPER_MOTOR, MotorFamily, MotorFamilyProfile, profile_for

CAN_ADAPTERS = ("damiao", "socketcan")


def _broadcast_per_joint(
    name: str,
    value: float | list[float] | Mapping[str, float],
    joints: Iterable[str],
) -> dict[str, float]:
    """Normalize a legacy scalar/list or mapping into a complete joint mapping."""
    joints = tuple(joints)
    if isinstance(value, (int, float)):
        return dict.fromkeys(joints, float(value))
    if isinstance(value, list):
        if len(value) != len(joints):
            raise ValueError(f"`{name}` must contain exactly {len(joints)} values, got {len(value)}.")
        return {joint: float(item) for joint, item in zip(joints, value, strict=True)}

    _validate_exact_keys(name, value, joints)
    return {joint: float(value[joint]) for joint in joints}


def _validate_exact_keys(name: str, value: Mapping[str, object], joints: Iterable[str]) -> None:
    expected = set(joints)
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unknown: {', '.join(extra)}")
        raise ValueError(f"`{name}` must contain exactly the configured joints ({'; '.join(details)}).")


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"`{name}` must be finite.")
    return value


def public_joint_limits(
    raw_limits: Mapping[str, tuple[float, float]],
    directions: Mapping[str, float],
) -> dict[str, tuple[float, float]]:
    """Convert raw motor-frame limits into the public robot coordinate frame."""
    return {
        joint: tuple(sorted((lower / directions[joint], upper / directions[joint])))
        for joint, (lower, upper) in raw_limits.items()
    }


@dataclass
class RebotB601FollowerConfig:
    """Base configuration for the Seeed Studio reBot B601 follower arm.

    The B601 is a 6-DOF arm plus gripper sold in two builds with the same joint
    topology but different geometry and actuators, selected by `motor_family`:
    `"dm"` for the Damiao B601-DM and `"rs"` for the RobStride B601-RS. Motor
    communication goes through the ``motorbridge`` package over a CAN bus.

    Every per-joint field below defaults to `None`, meaning "use the value for my
    motor family". After `__post_init__` they are all fully populated dicts keyed
    by the joints declared in `motor_can_ids`, so consumers never need to handle a
    scalar, a partial mapping, or `None`.
    """

    # Communication port. For `can_adapter="damiao"` this is the Damiao serial
    # bridge device (e.g. "/dev/ttyACM0"); for `"socketcan"` it is the CAN channel
    # name (e.g. "can0").
    port: str

    # Omitted by legacy DM configs. RS must be selected explicitly.
    motor_family: MotorFamily = MotorFamily.DM

    # CAN transport: "damiao" for the Damiao-only USB-to-CAN serial bridge,
    # "socketcan" for SocketCAN adapters. SocketCAN supports both families.
    can_adapter: str | None = None

    # Baud rate of the Damiao serial bridge. Unused when can_adapter="socketcan".
    dm_serial_baud: int = 921600

    disable_torque_on_disconnect: bool = True

    # Caps the magnitude of the relative positional target vector (in degrees) for
    # safety. A scalar applies to every joint; a dict sets per-joint values. `None`
    # disables the check entirely — unlike the fields below, `None` is a real value
    # here and is not filled in from the motor family.
    max_relative_target: float | dict[str, float] | None = None
    feedback_cache_ttl_s: float = 0.1

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Maps joint names to their (send_can_id, recv_can_id) pair. Damiao motors use
    # a per-motor recv id; RobStride motors all answer on the host id.
    motor_can_ids: dict[str, tuple[int, int]] | None = None

    # Arm control mode. "mit" everywhere; "pos_vel" on Damiao only.
    control_mode: str | None = None

    # Gripper control mode. "mit_impedance" (force-limited, RobStride) or
    # "force_pos" (Damiao) or plain "mit" position control on either.
    gripper_control_mode: str | None = None

    # MIT gains per joint, gripper included. Note the MIT frame packs gains against
    # a model-dependent full scale, so RobStride gains are not comparable between
    # the proximal (rs-06) and distal (rs-00) joints.
    mit_kp: float | list[float] | dict[str, float] | None = None
    mit_kd: float | list[float] | dict[str, float] | None = None

    # Speed cap (deg/s) for POS_VEL arm joints and the FORCE_POS gripper.
    pos_vel_velocity: float | list[float] | dict[str, float] | None = None

    # FORCE_POS gripper: grip force as a fraction of peak torque, in [0, 1].
    gripper_torque_ratio: float | None = None

    # Legacy aliases for mit_kp["gripper"] and mit_kd["gripper"].
    gripper_mit_kp: float | None = None
    gripper_mit_kd: float | None = None

    # Impedance gripper: max |feedforward torque| (N.m) while moving, and the
    # gentler cap at near-zero speed that bounds grip force on a closed grasp.
    gripper_torque_limit: float | None = None
    gripper_hold_torque_limit: float | None = None

    # Soft joint limits in the raw motor frame (degrees), clipped against after
    # `joint_directions` maps the public action into that frame.
    joint_limits: dict[str, tuple[float, float]] | None = None

    # Sign converting positions and torques between the public robot frame and the
    # raw motor frame. Only +1 and -1 are valid; scaling belongs in a processor or
    # teleoperator config.
    joint_directions: float | dict[str, float] | None = None

    # Refuse to enable torque when a joint reads far outside its limits, which
    # means a multi-turn encoder woke up wrapped by a whole revolution after a
    # power cycle. Commanding such a joint would drive it into its hard stop.
    check_position_plausibility: bool = True
    wrap_guard_margin_deg: float = 90.0

    @property
    def profile(self) -> MotorFamilyProfile:
        """Hardware profile for this arm's motor family."""
        return profile_for(self.motor_family)

    def _resolve_motor_family_defaults(self) -> None:
        """Fill every unset per-joint field from this arm's motor family profile."""
        profile = profile_for(self.motor_family)
        self.motor_family = profile.family
        if not self.port:
            raise ValueError("`port` must not be empty.")
        if not isinstance(self.dm_serial_baud, int) or self.dm_serial_baud <= 0:
            raise ValueError("`dm_serial_baud` must be a positive integer.")

        if self.can_adapter is None:
            self.can_adapter = profile.can_adapter
        if self.can_adapter not in CAN_ADAPTERS:
            raise ValueError(
                f"Unsupported can_adapter '{self.can_adapter}'. Available: {', '.join(CAN_ADAPTERS)}."
            )
        if self.can_adapter not in profile.can_adapters:
            available = ", ".join(sorted(profile.can_adapters))
            raise ValueError(
                f"can_adapter '{self.can_adapter}' is not available for "
                f"{self.motor_family.value} motors. Available: {available}."
            )

        joints = tuple(profile.motor_models)
        if self.motor_can_ids is None:
            self.motor_can_ids = dict(profile.motor_can_ids)
        _validate_exact_keys("motor_can_ids", self.motor_can_ids, joints)
        normalized_ids: dict[str, tuple[int, int]] = {}
        for joint, ids in self.motor_can_ids.items():
            if not isinstance(ids, (tuple, list)) or len(ids) != 2:
                raise ValueError(f"`motor_can_ids[{joint}]` must be a (send_id, receive_id) pair.")
            send_id, receive_id = ids
            if (
                not isinstance(send_id, int)
                or isinstance(send_id, bool)
                or not 0 < send_id <= 0x7FF
                or not isinstance(receive_id, int)
                or isinstance(receive_id, bool)
                or not 0 <= receive_id <= 0x7FF
            ):
                raise ValueError(f"`motor_can_ids[{joint}]` contains an invalid classic-CAN identifier.")
            normalized_ids[joint] = (send_id, receive_id)
        self.motor_can_ids = normalized_ids
        send_ids = [send_id for send_id, _ in self.motor_can_ids.values()]
        if len(send_ids) != len(set(send_ids)):
            raise ValueError("Motor send CAN IDs must be unique.")
        receive_ids = [receive_id for _, receive_id in self.motor_can_ids.values()]
        if self.motor_family is MotorFamily.DM and len(receive_ids) != len(set(receive_ids)):
            raise ValueError("Damiao receive CAN IDs must be unique.")
        if self.motor_family is MotorFamily.RS and set(receive_ids) != {0xFD}:
            raise ValueError("RobStride receive CAN IDs must all use host ID 0xFD.")

        if self.control_mode is None:
            self.control_mode = profile.control_mode
        if self.control_mode not in profile.arm_modes:
            raise ValueError(
                f"control_mode '{self.control_mode}' is not available on {self.motor_family.value} "
                f"motors. Available: {', '.join(sorted(profile.arm_modes))}."
            )

        if self.gripper_control_mode is None:
            self.gripper_control_mode = profile.gripper_control_mode
        if self.gripper_control_mode not in profile.gripper_modes:
            raise ValueError(
                f"gripper_control_mode '{self.gripper_control_mode}' is not available on "
                f"{self.motor_family.value} motors. Available: "
                f"{', '.join(sorted(profile.gripper_modes))}."
            )

        gain_inputs = {"mit_kp": self.mit_kp, "mit_kd": self.mit_kd}
        for name in ("mit_kp", "mit_kd", "joint_directions"):
            value = getattr(self, name)
            default = getattr(profile, name)
            setattr(self, name, _broadcast_per_joint(name, value if value is not None else default, joints))

        for name, alias_name in (("mit_kp", "gripper_mit_kp"), ("mit_kd", "gripper_mit_kd")):
            alias = getattr(self, alias_name)
            values = getattr(self, name)
            original = gain_inputs[name]
            if alias is None:
                # Legacy DM scalar/list gains controlled arm joints only; the
                # gripper had independent defaults. A mapping is the new explicit
                # way to configure every joint, including the gripper.
                if (
                    self.motor_family is MotorFamily.DM
                    and original is not None
                    and not isinstance(original, Mapping)
                ):
                    values[GRIPPER_MOTOR] = getattr(profile, name)[GRIPPER_MOTOR]
                setattr(self, alias_name, values[GRIPPER_MOTOR])
                continue
            alias = _require_finite(alias_name, alias)
            if isinstance(original, Mapping) and values[GRIPPER_MOTOR] != alias:
                raise ValueError(
                    f'`{alias_name}` conflicts with `{name}["{GRIPPER_MOTOR}"]`; configure only one value.'
                )
            values[GRIPPER_MOTOR] = alias
            setattr(self, alias_name, alias)

        for name in ("mit_kp", "mit_kd", "joint_directions"):
            values = getattr(self, name)
            for joint, value in values.items():
                values[joint] = _require_finite(f"{name}[{joint}]", value)

        invalid_directions = {
            joint: direction
            for joint, direction in self.joint_directions.items()
            if direction not in (-1.0, 1.0)
        }
        if invalid_directions:
            raise ValueError(
                "`joint_directions` values must be +1 or -1. Invalid values: "
                + ", ".join(f"{joint}={value}" for joint, value in invalid_directions.items())
                + "."
            )

        if self.joint_limits is None:
            self.joint_limits = dict(profile.joint_limits)
        _validate_exact_keys("joint_limits", self.joint_limits, joints)
        normalized_limits: dict[str, tuple[float, float]] = {}
        for joint, limits in self.joint_limits.items():
            if not isinstance(limits, (tuple, list)) or len(limits) != 2:
                raise ValueError(f"`joint_limits[{joint}]` must contain exactly (min, max).")
            lower = _require_finite(f"joint_limits[{joint}][0]", limits[0])
            upper = _require_finite(f"joint_limits[{joint}][1]", limits[1])
            if lower >= upper:
                raise ValueError(f"`joint_limits[{joint}]` must satisfy min < max.")
            normalized_limits[joint] = (lower, upper)
        self.joint_limits = normalized_limits

        for joint in joints:
            kp_max, kd_max = profile.mit_gain_scale[joint]
            # Shipped DM configs use kd=12 on the proximal joints even though the
            # MIT frame saturates at 5. Keep accepting that established input
            # (and no larger) so upgrading does not invalidate existing configs.
            kd_config_max = max(kd_max, profile.mit_kd[joint])
            kp = self.mit_kp[joint]
            kd = self.mit_kd[joint]
            if not 0.0 <= kp <= kp_max:
                raise ValueError(f"`mit_kp[{joint}]` must be in [0, {kp_max}].")
            if not 0.0 <= kd <= kd_config_max:
                raise ValueError(f"`mit_kd[{joint}]` must be in [0, {kd_config_max}].")

        self.wrap_guard_margin_deg = _require_finite("wrap_guard_margin_deg", self.wrap_guard_margin_deg)
        if self.wrap_guard_margin_deg < 0.0:
            raise ValueError("`wrap_guard_margin_deg` must be non-negative.")
        self.feedback_cache_ttl_s = _require_finite("feedback_cache_ttl_s", self.feedback_cache_ttl_s)
        if self.feedback_cache_ttl_s <= 0.0:
            raise ValueError("`feedback_cache_ttl_s` must be positive.")

        if self.max_relative_target is not None:
            if isinstance(self.max_relative_target, Mapping):
                _validate_exact_keys("max_relative_target", self.max_relative_target, joints)
                self.max_relative_target = {
                    joint: _require_finite(f"max_relative_target[{joint}]", value)
                    for joint, value in self.max_relative_target.items()
                }
                if any(value <= 0.0 for value in self.max_relative_target.values()):
                    raise ValueError("Every `max_relative_target` value must be positive.")
            else:
                self.max_relative_target = _require_finite("max_relative_target", self.max_relative_target)
                if self.max_relative_target <= 0.0:
                    raise ValueError("`max_relative_target` must be positive.")

        self._resolve_mode_scoped_defaults(profile, joints)

    def _resolve_mode_scoped_defaults(self, profile: MotorFamilyProfile, joints: tuple[str, ...]) -> None:
        """Fill in the parameters that only exist for some control modes.

        A parameter the active family cannot use is rejected rather than ignored:
        silently dropping something like a gripper torque limit would leave the
        user believing a safety cap is in force when it is not.
        """
        uses_velocity_limit = self.control_mode == "pos_vel" or self.gripper_control_mode == "force_pos"
        if not uses_velocity_limit:
            if self.pos_vel_velocity is not None:
                raise ValueError(
                    "`pos_vel_velocity` has no effect unless the arm uses `pos_vel` "
                    "or the gripper uses `force_pos`."
                )
        else:
            value = self.pos_vel_velocity
            default_velocity = profile.pos_vel_velocity
            if value is None and default_velocity is None:
                raise ValueError(
                    f"No default `pos_vel_velocity` is defined for {self.motor_family.value} motors."
                )
            self.pos_vel_velocity = _broadcast_per_joint(
                "pos_vel_velocity",
                value if value is not None else default_velocity,
                joints,
            )
            for joint, velocity in self.pos_vel_velocity.items():
                velocity = _require_finite(f"pos_vel_velocity[{joint}]", velocity)
                if velocity <= 0.0:
                    raise ValueError(f"`pos_vel_velocity[{joint}]` must be positive.")
                self.pos_vel_velocity[joint] = velocity

        if self.gripper_control_mode == "force_pos":
            if self.gripper_torque_ratio is None:
                self.gripper_torque_ratio = profile.gripper_torque_ratio
            if self.gripper_torque_ratio is None:
                raise ValueError("`gripper_torque_ratio` is required in `force_pos` mode.")
            self.gripper_torque_ratio = _require_finite("gripper_torque_ratio", self.gripper_torque_ratio)
            if not 0.0 <= self.gripper_torque_ratio <= 1.0:
                raise ValueError("`gripper_torque_ratio` must be in [0, 1].")
        elif self.gripper_torque_ratio is not None:
            raise ValueError("`gripper_torque_ratio` only applies in `force_pos` mode.")

        impedance_fields = ("gripper_torque_limit", "gripper_hold_torque_limit")
        if self.gripper_control_mode == "mit_impedance":
            for name in impedance_fields:
                if getattr(self, name) is None:
                    setattr(self, name, getattr(profile, name))
                value = _require_finite(name, getattr(self, name))
                setattr(self, name, value)
                if value <= 0:
                    raise ValueError(f"`{name}` must be positive in `mit_impedance` mode.")
            if self.gripper_hold_torque_limit > self.gripper_torque_limit:
                raise ValueError("`gripper_hold_torque_limit` must not exceed `gripper_torque_limit`.")
            gripper_ceiling = profile.torque_ceiling[GRIPPER_MOTOR]
            if self.gripper_torque_limit > gripper_ceiling:
                raise ValueError(
                    f"`gripper_torque_limit` must not exceed the motor peak torque {gripper_ceiling}."
                )
        else:
            configured = [name for name in impedance_fields if getattr(self, name) is not None]
            if configured:
                raise ValueError(
                    f"{', '.join(f'`{name}`' for name in configured)} only applies in `mit_impedance` mode."
                )

    def __post_init__(self) -> None:
        self._resolve_motor_family_defaults()

    def as_robot_config(
        self,
        *,
        id: str | None = None,
        calibration_dir: Path | None = None,
        cameras: dict[str, CameraConfig] | None = None,
    ) -> "RebotB601FollowerRobotConfig":
        """Promote this arm config to the registered robot config.

        Used by the bimanual follower, whose per-arm configs are declared as the
        plain base type. Every field is forwarded by name so that adding an option
        never requires threading it through by hand.
        """
        values = {f.name: getattr(self, f.name) for f in fields(RebotB601FollowerConfig)}
        values = deepcopy(values)
        if cameras is not None:
            values["cameras"] = deepcopy(cameras)
        return RebotB601FollowerRobotConfig(id=id, calibration_dir=calibration_dir, **values)


@RobotConfig.register_subclass("rebot_b601_follower")
@dataclass
class RebotB601FollowerRobotConfig(RobotConfig, RebotB601FollowerConfig):
    """Registered configuration for the reBot B601 follower robot.

    Selects the Damiao or RobStride build with `--robot.motor_family={dm,rs}`.
    """

    def __post_init__(self) -> None:
        # `RobotConfig` comes first in the MRO, so its `__post_init__` shadows the
        # one on `RebotB601FollowerConfig`. Chain both explicitly.
        RobotConfig.__post_init__(self)
        RebotB601FollowerConfig.__post_init__(self)


__all__ = [
    "CAN_ADAPTERS",
    "RebotB601FollowerConfig",
    "RebotB601FollowerRobotConfig",
    "public_joint_limits",
]
