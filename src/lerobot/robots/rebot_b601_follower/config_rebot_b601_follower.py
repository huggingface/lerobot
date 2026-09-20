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

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from .motor_family import GRIPPER_MOTOR, MIT_MODE, MOTOR_PROFILES, MotorFamily


def _per_joint(value: float | list[float] | Mapping[str, float], joints: tuple[str, ...]) -> dict[str, float]:
    """Expand legacy scalar and list values into a joint-keyed mapping."""
    if isinstance(value, (int, float)):
        return dict.fromkeys(joints, float(value))
    if isinstance(value, list):
        return {joint: float(item) for joint, item in zip(joints, value, strict=True)}
    return dict(value)


@dataclass
class RebotB601FollowerConfig:
    """Base configuration for the Seeed Studio reBot B601 follower arm.

    The B601 is a 6-DOF arm plus gripper sold in two builds with the same joint
    topology but different geometry and actuators, selected by `motor_family`:
    `"dm"` for the Damiao B601-DM and `"rs"` for the RobStride B601-RS. Motor
    communication goes through the ``motorbridge`` package over a CAN bus.

    Family-specific defaults are selected in `__post_init__`. Legacy scalar and
    list tuning values are expanded into joint-keyed mappings for the runtime.
    Explicit values are otherwise passed through unchanged.
    """

    # Communication port. For `can_adapter="damiao"` this is the Damiao serial
    # bridge device (e.g. "/dev/ttyACM0"); for `"socketcan"` it is MotorBridge's
    # platform-specific CAN channel identifier (e.g. "can0").
    port: str

    # Omitted by legacy DM configs. RS must be selected explicitly.
    motor_family: MotorFamily = MotorFamily.DM

    # CAN transport: "damiao" for the Damiao-only USB-to-CAN serial bridge;
    # "socketcan" is the historical name for MotorBridge's native CAN transport,
    # whose backend is selected for the host platform. Native CAN supports both
    # motor families.
    can_adapter: str | None = None

    # Baud rate of the Damiao serial bridge. Unused by the native CAN transport.
    dm_serial_baud: int = 921600

    disable_torque_on_disconnect: bool = True

    # Caps the magnitude of the relative positional target vector (in degrees) for
    # safety. A scalar applies to every joint; a dict sets per-joint values. `None`
    # disables the check entirely — unlike the fields below, `None` is a real value
    # here and is not filled in from the motor family.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Maps joint names to their (send_can_id, recv_can_id) pair. Damiao motors use
    # a per-motor recv id; RobStride motors all answer on the host id.
    motor_can_ids: dict[str, tuple[int, int]] | None = None

    # Arm control mode. "mit" everywhere; "pos_vel" on Damiao only.
    control_mode: str = MIT_MODE

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

    def _resolve_motor_family_defaults(self) -> None:
        """Fill every unset per-joint field from this arm's motor family profile."""
        self.motor_family = MotorFamily(self.motor_family)
        profile = MOTOR_PROFILES[self.motor_family]
        joints = tuple(profile.motor_models)

        if self.can_adapter is None:
            self.can_adapter = profile.can_adapter

        if self.motor_can_ids is None:
            self.motor_can_ids = dict(profile.motor_can_ids)

        if self.gripper_control_mode is None:
            self.gripper_control_mode = profile.gripper_control_mode

        gain_inputs = {"mit_kp": self.mit_kp, "mit_kd": self.mit_kd}
        for name in ("mit_kp", "mit_kd"):
            value = getattr(self, name)
            default = getattr(profile, name)
            setattr(self, name, _per_joint(value if value is not None else default, joints))

        for name, alias_name in (("mit_kp", "gripper_mit_kp"), ("mit_kd", "gripper_mit_kd")):
            alias = getattr(self, alias_name)
            values = getattr(self, name)
            original = gain_inputs[name]
            if alias is None:
                # Legacy DM scalar/list gains controlled arm joints only; the
                # gripper had independent defaults. A mapping is the new explicit
                # way to configure every joint, including the gripper.
                if original is not None and not isinstance(original, Mapping):
                    values[GRIPPER_MOTOR] = getattr(profile, name)[GRIPPER_MOTOR]
            else:
                values[GRIPPER_MOTOR] = alias
            setattr(self, alias_name, values[GRIPPER_MOTOR])

        if self.joint_limits is None:
            self.joint_limits = dict(profile.joint_limits)

        velocity = self.pos_vel_velocity if self.pos_vel_velocity is not None else profile.pos_vel_velocity
        if velocity is not None:
            self.pos_vel_velocity = _per_joint(velocity, joints)

        if self.gripper_torque_ratio is None:
            self.gripper_torque_ratio = profile.gripper_torque_ratio

        for name in ("gripper_torque_limit", "gripper_hold_torque_limit"):
            if getattr(self, name) is None:
                setattr(self, name, getattr(profile, name))

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
