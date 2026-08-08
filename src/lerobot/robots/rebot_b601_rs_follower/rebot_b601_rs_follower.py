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

import logging
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.robstride import RobstridePrivateMotorsBus
from lerobot.motors.robstride.tables import PrivateControlMode
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rebot_b601_rs_follower import RebotB601RSFollowerRobotConfig

logger = logging.getLogger(__name__)

# In MIT mode the gripper is driven with its own gains; every other joint uses mit_kp/mit_kd.
GRIPPER_MOTOR = "gripper"


class RebotB601RSFollower(Robot):
    """Seeed Studio reBot B601-RS follower arm (6-DOF + gripper, Robstride CAN motors).

    Motor communication uses the Robstride vendor ("private") CAN protocol through
    :class:`~lerobot.motors.robstride.RobstridePrivateMotorsBus`, over SocketCAN or an
    slcan bridge. Only ``python-can`` is required (``pip install 'lerobot[robstride]'``).

    Warning: on disconnect with ``disable_torque_on_disconnect=True`` (the default) the
    motors are stopped and the arm becomes back-drivable — it will fall under gravity.
    Hold the arm or park it in a stable rest pose before disconnecting.
    """

    config_class = RebotB601RSFollowerRobotConfig
    name = "rebot_b601_rs_follower"

    def __init__(self, config: RebotB601RSFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self.motor_names = list(config.motor_ids.keys())

        missing_models = [name for name in self.motor_names if name not in config.motor_models]
        if missing_models:
            raise ValueError(f"Missing motor_models entries for motors: {missing_models}")

        motors = {
            name: Motor(
                id=config.motor_ids[name],
                model="robstride",
                norm_mode=MotorNormMode.DEGREES,
                motor_type_str=config.motor_models[name],
            )
            for name in self.motor_names
        }
        self.bus = RobstridePrivateMotorsBus(
            port=config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=config.can_interface,
            host_id=config.host_id,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting {self} on {self.config.port} (interface={self.config.can_interface})...")
        self.bus.connect()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration)

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                "or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using calibration file associated with the id {self.id}")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        print(
            "\nCalibration: set zero position.\n"
            "Manually move the reBot B601 to its ZERO POSITION and close the gripper.\n"
            "See the B601 manual for the zero pose (the default sit-down position).\n"
        )
        input("Press ENTER when ready...")

        self.bus.set_zero_position()
        # Flash the parameter file so the zero survives power cycles on every firmware
        # revision (the vendor tools treat this store as optional but recommended).
        self.bus.save_parameters()
        logger.info("Arm zero position set.")

        self.calibration = {}
        for motor_name, motor_id in self.config.motor_ids.items():
            range_min, range_max = self.config.joint_limits[motor_name]
            self.calibration[motor_name] = MotorCalibration(
                id=motor_id,
                drive_mode=0,
                homing_offset=0,
                range_min=int(range_min),
                range_max=int(range_max),
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        if self.config.control_mode not in ("position", "mit"):
            raise ValueError(
                f"Unsupported control_mode '{self.config.control_mode}'. Use 'position' or 'mit'."
            )
        use_mit = self.config.control_mode == "mit"
        mode = PrivateControlMode.MIT if use_mit else PrivateControlMode.POSITION
        self.bus.configure_motors(mode)
        self._check_positions_plausible()
        if not use_mit:
            for idx, motor_name in enumerate(self.motor_names):
                speed_deg_s = (
                    self.config.position_speed_limit[idx]
                    if isinstance(self.config.position_speed_limit, list)
                    else self.config.position_speed_limit
                )
                self.bus.set_position_speed_limit(motor_name, speed_deg_s)
        self.bus.enable_torque()

    # Margin (degrees) beyond the soft joint limits that still counts as a plausible
    # reading. A multi-turn wrap shifts a reading by ~360 deg, so anything past this
    # margin means the encoder woke wrapped and must be re-homed, not commanded.
    _WRAP_GUARD_MARGIN_DEG = 90.0

    def _check_positions_plausible(self) -> None:
        """Refuse to enable torque on a joint whose reading is a multi-turn wrap.

        The RS motors' single-turn zero survives power cycles but the multi-turn count
        does not; geared joints with more than one turn of travel (the gripper) can wake
        reading ``physical + 360*k`` degrees. Commanding such a joint would slam it into
        its mechanical stop, so fail loudly instead.
        """
        positions = self.bus.sync_read("Present_Position")
        wrapped = []
        for motor_name, position in positions.items():
            range_min, range_max = self.config.joint_limits[motor_name]
            if not (
                range_min - self._WRAP_GUARD_MARGIN_DEG <= position <= range_max + self._WRAP_GUARD_MARGIN_DEG
            ):
                wrapped.append(f"{motor_name}={position:.1f} deg (limits {range_min}..{range_max} deg)")
        if wrapped:
            raise RuntimeError(
                "Implausible joint reading(s), most likely a multi-turn encoder wrap after a "
                f"power cycle: {', '.join(wrapped)}. Move the joint(s) back into range by hand "
                "(gripper: close it against the stop) and re-run calibration before enabling "
                "torque."
            )

    @check_if_not_connected
    def disable_torque(self) -> None:
        """Disable motor torque so the arm can be moved by hand (read-only debugging).

        The arm becomes back-drivable and will fall under gravity: hold it first.
        """
        self.bus.disable_torque()
        logger.info(f"{self} torque disabled.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        positions = self.bus.sync_read("Present_Position")
        obs_dict: RobotObservation = {f"{motor}.pos": pos for motor, pos in positions.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command the arm to a target joint configuration.

        Positions are expressed in degrees. The relative action magnitude may be
        clipped depending on `max_relative_target`, so the action actually sent is
        always returned.
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Clip against soft joint limits.
        for motor_name in list(goal_pos):
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped = max(min_limit, min(max_limit, goal_pos[motor_name]))
                if clipped != goal_pos[motor_name]:
                    logger.debug(f"Clipped {motor_name} from {goal_pos[motor_name]:.2f} to {clipped:.2f}")
                goal_pos[motor_name] = clipped

        # Tolerate 6-DOF leaders that have no wrist_yaw joint by holding it at zero.
        # This is intentional: it lets a 6-DOF leader such as the SO-100 / SO-101
        # (so100_leader / so101_leader) teleoperate this 7-DOF follower — the missing
        # wrist_yaw command is simply treated as 0.0 instead of raising.
        if "wrist_yaw" not in goal_pos:
            goal_pos["wrist_yaw"] = 0.0

        # Cap relative target when too far from the present position.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g, present_pos.get(key, g)) for key, g in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Unknown motor keys are silently skipped.
        known_goals = {name: pos for name, pos in goal_pos.items() if name in self.config.motor_ids}
        if self.config.control_mode == "mit":
            commands: dict[str, tuple[float, float, float, float, float]] = {}
            for motor_name, position_deg in known_goals.items():
                if motor_name == GRIPPER_MOTOR:
                    kp: float = self.config.gripper_mit_kp
                    kd: float = self.config.gripper_mit_kd
                else:
                    idx = self.motor_names.index(motor_name)
                    mit_kp, mit_kd = self.config.mit_kp, self.config.mit_kd
                    kp = mit_kp[idx] if isinstance(mit_kp, list) else mit_kp
                    kd = mit_kd[idx] if isinstance(mit_kd, list) else mit_kd
                commands[motor_name] = (kp, kd, position_deg, 0.0, 0.0)
            self.bus._mit_control_batch(commands)
        else:
            self.bus.sync_write("Goal_Position", known_goals)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        """Disconnect from the robot.

        With ``disable_torque_on_disconnect=True`` (the default) the motors are stopped
        and the arm becomes back-drivable: hold the arm or park it in a stable rest pose
        before disconnecting, otherwise it falls under gravity.
        """
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
