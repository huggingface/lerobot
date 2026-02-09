#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any, Dict

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.robstride import RobstrideMotorsBus,CommMode  # adapte le chemin exact
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_robstride import RobstrideTestConfig  # adapte le nom du fichier

logger = logging.getLogger(__name__)


class RobstrideTest(Robot):
    """
    Robot de test pour un ou plusieurs moteurs Robstride sur bus CAN.
    """

    config_class = RobstrideTestConfig
    name = "robstride_test"

    def __init__(self, config: RobstrideTestConfig):
        super().__init__(config)
        self.config = config

        norm_mode = MotorNormMode.DEGREES

        # Motors: {nom_joint: Motor}
        self.motors: Dict[str, Motor] = {}


        # config.motor_config: {name: (send_id, recv_id, motor_type_str)}
        for motor_name, (send_id, recv_id, motor_type_str) in self.config.motor_config.items():
            motor = Motor(
                send_id,
                motor_type_str,  # pour Robstride tu peux gérer ça en string dans le bus
                norm_mode,
            )
            # tu peux garder recv_id si ton bus l’utilise :
            motor.recv_id = recv_id
            self.motors[motor_name] = motor

        # Bus unique Robstride
        self.bus = RobstrideMotorsBus(
            port=self.config.port,
            motors=self.motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            # data_bitrate=None pour CAN classique, ou paramètre si ton bus le prend
        )

        # Caméras (optionnelles)
        self.cameras = make_cameras_from_configs(self.config.cameras)

    # --------- Features / spaces ---------

    @property
    def _motors_ft(self) -> Dict[str, type]:
        """Features moteurs (positions seulement pour commencer)."""
        return {f"{name}.pos": float for name in self.motors.keys()}

    @property
    def _cameras_ft(self) -> Dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> Dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> Dict[str, type]:
        return self._motors_ft

    # --------- Connection / calibration ---------

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        return getattr(self.bus, "is_calibrated", True)

    def connect(self, calibrate: bool = False) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        logger.info(f"Connecting Robstride bus on {self.config.port}...")
        self.bus.connect()

        if calibrate and not self.is_calibrated:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        if self.config.zero_position_on_connect and hasattr(self.bus, "set_zero_position"):
            logger.info("Setting current position as zero...")
            self.bus.set_zero_position()

        logger.info(f"{self} connected.")
        


    def calibrate(self) -> None:
        if self.calibration:
            # self.calibration is not empty here
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_zero_position()

        print(
            "Move all joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=0,  #we set the zero inside motor software
                range_min=range_mins[motor], #need int values here
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)
            

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        if hasattr(self.bus, "torque_disabled"):
            with self.bus.torque_disabled():
                self.bus.configure_motors()
        else:
            self.bus.configure_motors()

    # --------- IO: obs / actions ---------

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        pos_dict = self.bus.sync_read("Present_Position")
        obs_dict: dict[str, Any] = {
            f"{motor}.pos": val for motor, val in pos_dict.items()
        }

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict
    

    def _denormalize_value(
        self, data_name: str, motor_name: str, normalized_value: float
    ) -> float: 
        """Convert a normalized value back to the motor's native unit (degrees)."""
        motor = self.motors[motor_name]
        if motor.norm_mode == MotorNormMode.DEGREES:
            # Already in degrees
            return normalized_value
        elif motor.norm_mode == MotorNormMode.RADIANS:
            # Convert radians to degrees
            return np.degrees(normalized_value)
        else:
            raise ValueError(
                f"Unsupported normalization mode {motor.norm_mode} for motor {motor_name}"
            )






    def send_normalized_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {
            key.removesuffix(".pos"): self._denormalize_value(
                f"{key}.pos", key.removesuffix(".pos"), val
            )
            for key, val in action.items()
            if key.endswith(".pos")
        }

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, self.config.max_relative_target
            )

        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}
    

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, self.config.max_relative_target
            )

        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    # --------- Déconnexion ---------

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    # --------- Utils ---------

    def _deg_to_rad(self, deg: Dict[str, float | int]) -> Dict[str, float]:
        return {m: np.deg2rad(float(v)) for m, v in deg.items()}
