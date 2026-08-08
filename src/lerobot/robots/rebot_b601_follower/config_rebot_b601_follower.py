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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class RebotB601FollowerConfig:
    """Field definitions for the Seeed Studio reBot B601-DM follower arm.

    This class only carries the fields. The registered configuration users instantiate is
    [`RebotB601FollowerRobotConfig`], which documents them all in one place — doc-builder renders only a
    class's own docstring, never its bases'. It is also used directly as the per-arm config of
    [`~robots.bi_rebot_b601_follower.BiRebotB601FollowerConfig`].
    """

    # Communication port. For ``can_adapter="damiao"`` this is the Damiao serial
    # bridge device (e.g. "/dev/ttyACM0"); for ``can_adapter="socketcan"`` it is
    # the CAN channel name (e.g. "can0").
    port: str

    # CAN adapter type:
    #   "damiao"    - Damiao dedicated serial bridge (default)
    #   "socketcan" - SocketCAN based adapters (PCAN, slcan, embedded controllers, ...)
    can_adapter: str = "damiao"

    # Baud rate for the Damiao serial bridge (only used when can_adapter="damiao").
    dm_serial_baud: int = 921600

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target
    # vector for safety purposes (in degrees). Set to a positive scalar to apply the
    # same value to all motors, or to a dict mapping motor names to per-motor values.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Maps motor names to their (send_can_id, recv_can_id) pair.
    motor_can_ids: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0x01, 0x11),
            "shoulder_lift": (0x02, 0x12),
            "elbow_flex": (0x03, 0x13),
            "wrist_flex": (0x04, 0x14),
            "wrist_yaw": (0x05, 0x15),
            "wrist_roll": (0x06, 0x16),
            "gripper": (0x07, 0x17),
        }
    )

    # Max speed (deg/s) per joint for POS_VEL arms and FORCE_POS gripper (motor order).
    pos_vel_velocity: float | list[float] = field(
        default_factory=lambda: [150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 900.0]
    )

    # Arm control: "mit" or "pos_vel".
    control_mode: str = "mit"

    # MIT kp/kd per arm joint (motor order). Unused when control_mode="pos_vel".
    mit_kp: float | list[float] = field(default_factory=lambda: [45.0, 45.0, 45.0, 8.0, 9.0, 8.0, 8.0])
    mit_kd: float | list[float] = field(default_factory=lambda: [12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0])

    # Gripper control: "force_pos" or "mit".
    gripper_control_mode: str = "force_pos"

    # FORCE_POS only: max grip force, in [0, 1].
    gripper_torque_ratio: float = 0.07

    # MIT only.
    gripper_mit_kp: float = 8.0
    gripper_mit_kd: float = 0.3

    # Soft joint limits (degrees). These are clipped against on every action.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan": (-150.0, 150.0),
            "shoulder_lift": (-200.0, 1.0),
            "elbow_flex": (-200.0, 1.0),
            "wrist_flex": (-80.0, 90.0),
            "wrist_yaw": (-90.0, 90.0),
            "wrist_roll": (-90.0, 90.0),
            "gripper": (-270.0, 0.0),
        }
    )


@RobotConfig.register_subclass("rebot_b601_follower")
@dataclass
class RebotB601FollowerRobotConfig(RobotConfig, RebotB601FollowerConfig):
    """Configuration for the Seeed Studio reBot B601-DM follower arm.

    The B601-DM is a 6-DOF arm plus gripper on Damiao CAN motors, driven through the `motorbridge`
    package. What `port` means depends on `can_adapter`. Calibration follows the usual LeRobot flow and is
    stored per `id`.

    The arm and the gripper are controlled separately: `control_mode` governs the six arm joints and
    `gripper_control_mode` the gripper, and each mode uses a different subset of the gain fields.

    Per-joint lists hold 7 values in motor order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`,
    `wrist_flex`, `wrist_yaw`, `wrist_roll`, `gripper`.

    Args:
        port (`str`):
            Where the arm is reached. For `can_adapter="damiao"` this is the serial bridge device, e.g.
            `/dev/ttyACM0`; for `can_adapter="socketcan"` it is the CAN channel name, e.g. `can0`.
        can_adapter (`str`, *optional*, defaults to `"damiao"`):
            `"damiao"` for the dedicated Damiao serial bridge, or `"socketcan"` for SocketCAN adapters
            such as PCAN, slcan and embedded controllers.
        dm_serial_baud (`int`, *optional*, defaults to 921600):
            Baud rate of the Damiao serial bridge. Only used when `can_adapter="damiao"`.
        disable_torque_on_disconnect (`bool`, *optional*, defaults to `True`):
            Whether to release the motors on disconnect. Leave `True` unless the arm is holding a load it
            must not drop.
        max_relative_target (`float | dict[str, float]`, *optional*):
            Caps how far a single action may move the arm from its present position, in degrees. A scalar
            applies to every motor; a dict maps motor name to a per-motor cap. `None` disables clipping.
        cameras (`dict[str, CameraConfig]`, *optional*):
            Cameras to read alongside the arm's joint positions.
        motor_can_ids (`dict[str, tuple[int, int]]`, *optional*):
            Maps motor name to its `(send_can_id, recv_can_id)` pair. Change it only if you have
            re-addressed the motors.
        pos_vel_velocity (`float | list[float]`, *optional*):
            Maximum speed in deg/s per joint, used by the arm in `pos_vel` mode and by the gripper in
            `force_pos` mode.
        control_mode (`str`, *optional*, defaults to `"mit"`):
            How the six arm joints are driven: `"mit"` or `"pos_vel"`.
        mit_kp (`float | list[float]`, *optional*):
            MIT-mode proportional gains per arm joint. Unused when `control_mode="pos_vel"`.
        mit_kd (`float | list[float]`, *optional*):
            MIT-mode derivative gains per arm joint. Unused when `control_mode="pos_vel"`.
        gripper_control_mode (`str`, *optional*, defaults to `"force_pos"`):
            How the gripper is driven: `"force_pos"` or `"mit"`.
        gripper_torque_ratio (`float`, *optional*, defaults to 0.07):
            Maximum grip force as a fraction in `[0, 1]`. Only used when
            `gripper_control_mode="force_pos"`.
        gripper_mit_kp (`float`, *optional*, defaults to 8.0):
            Gripper MIT-mode proportional gain. Only used when `gripper_control_mode="mit"`.
        gripper_mit_kd (`float`, *optional*, defaults to 0.3):
            Gripper MIT-mode derivative gain. Only used when `gripper_control_mode="mit"`.
        joint_limits (`dict[str, tuple[float, float]]`, *optional*):
            Soft `(min, max)` limits in degrees per joint, clipped against on every action.
        id (`str`, *optional*):
            Identifier for this particular arm; also names its calibration file.
        calibration_dir (`Path`, *optional*):
            Where to read and write the calibration file. Defaults to the LeRobot calibration home.
    """

    pass
