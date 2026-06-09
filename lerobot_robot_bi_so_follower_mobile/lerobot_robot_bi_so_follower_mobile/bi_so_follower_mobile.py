import logging
from functools import cached_property

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots import Robot
from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_bi_so_follower_mobile import BiSOFollowerMobileConfig

logger = logging.getLogger(__name__)

ARM_MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

WHEEL_MOTORS = {
    "base_left_wheel": Motor(9, "sts3250", MotorNormMode.RANGE_M100_100),
    "base_right_wheel": Motor(10, "sts3250", MotorNormMode.RANGE_M100_100),
}


class BiSOFollowerMobile(Robot):
    """
    Bimanual SO-101 follower arms (Henrietta=left, Kermy=right) with a
    dual-wheel mobile base. The wheel motors (STS3250, IDs 9 & 10) share
    Kermy's motor bus with her 6 arm motors (STS3215, IDs 1-6).

    Wheel velocities are included as action and observation dimensions so
    they are recorded as part of the dataset during data collection.
    """
    config_class = BiSOFollowerMobileConfig
    name = "bi_so_follower_mobile"

    def __init__(self, config: BiSOFollowerMobileConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SOFollowerRobotConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            disable_torque_on_disconnect=config.left_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.left_arm_config.max_relative_target,
            use_degrees=config.left_arm_config.use_degrees,
            cameras=config.left_arm_config.cameras,
        )

        right_arm_config = SOFollowerRobotConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            disable_torque_on_disconnect=config.right_arm_config.disable_torque_on_disconnect,
            max_relative_target=config.right_arm_config.max_relative_target,
            use_degrees=config.right_arm_config.use_degrees,
            cameras=config.right_arm_config.cameras,
        )

        self.left_arm = SOFollower(left_arm_config)
        self.right_arm = SOFollower(right_arm_config)
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    def _inject_wheels(self):
        """
        Add wheel motors into the right arm's already-open bus after arm
        connect/calibration is complete. Injecting after connect ensures
        the calibration check only sees arm motors 1-6, not the wheels.
        """
        for name, motor in WHEEL_MOTORS.items():
            self.right_arm.bus.motors[name] = motor
        self.right_arm.bus._id_to_model_dict[9] = "sts3250"
        self.right_arm.bus._id_to_model_dict[10] = "sts3250"
        self.right_arm.bus._id_to_name_dict[9] = "base_left_wheel"
        self.right_arm.bus._id_to_name_dict[10] = "base_right_wheel"

    def _init_wheels(self):
        """Put wheel motors into velocity mode on the shared bus."""
        wheel_names = list(WHEEL_MOTORS.keys())
        self.right_arm.bus.disable_torque(wheel_names)
        for name in wheel_names:
            self.right_arm.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
        self.right_arm.bus.write("Maximum_Velocity_Limit", "base_left_wheel", 255, normalize=False)
        self.right_arm.bus.write("Maximum_Velocity_Limit", "base_right_wheel", 255, normalize=False)
        self.right_arm.bus.enable_torque(wheel_names)
        logger.info("Wheel motors initialized in velocity mode.")

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{m}.pos": float for m in ARM_MOTOR_NAMES},
            **{f"right_{m}.pos": float for m in ARM_MOTOR_NAMES},
            "base_left_wheel.vel": float,
            "base_right_wheel.vel": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm._cameras_ft.items()},
            **{f"right_{k}": v for k, v in self.right_arm._cameras_ft.items()},
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # Connect arms first — calibration only sees motors 1-6 per arm
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        # Inject and initialize wheels on the already-open right arm bus
        self._inject_wheels()
        self._init_wheels()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs_dict = {}

        # Left arm — clean bus, use SOFollower normally
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{k}": v for k, v in left_obs.items()})

        # Right arm — read only the 6 arm motors by name to avoid wheel reads
        right_pos = self.right_arm.bus.sync_read("Present_Position", ARM_MOTOR_NAMES)
        for motor, value in right_pos.items():
            obs_dict[f"right_{motor}.pos"] = float(value)

        # Right arm cameras
        for name, cam in self.right_arm.cameras.items():
            obs_dict[f"right_{name}"] = cam.async_read()

        # Wheel velocities from shared bus
        wheel_vel = self.right_arm.bus.sync_read("Present_Velocity", list(WHEEL_MOTORS.keys()))
        obs_dict["base_left_wheel.vel"] = float(wheel_vel["base_left_wheel"])
        obs_dict["base_right_wheel.vel"] = float(wheel_vel["base_right_wheel"])

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action = {k.removeprefix("left_"): v for k, v in action.items() if k.startswith("left_")}
        right_action = {k.removeprefix("right_"): v for k, v in action.items() if k.startswith("right_")}

        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)

        # Wheel velocity commands — right wheel negated due to mirrored mounting
        if "base_left_wheel.vel" in action:
            self.right_arm.bus.write(
                "Goal_Velocity", "base_left_wheel",
                int(action["base_left_wheel.vel"]), normalize=False
            )
        if "base_right_wheel.vel" in action:
            self.right_arm.bus.write(
                "Goal_Velocity", "base_right_wheel",
                int(action["base_right_wheel.vel"]) * -1, normalize=False
            )

        return {
            **{f"left_{k}": v for k, v in sent_left.items()},
            **{f"right_{k}": v for k, v in sent_right.items()},
            "base_left_wheel.vel": action.get("base_left_wheel.vel", 0.0),
            "base_right_wheel.vel": action.get("base_right_wheel.vel", 0.0),
        }

    @check_if_not_connected
    def disconnect(self) -> None:
        # Stop wheels before closing the bus
        try:
            self.right_arm.bus.write("Goal_Velocity", "base_left_wheel", 0, normalize=False)
            self.right_arm.bus.write("Goal_Velocity", "base_right_wheel", 0, normalize=False)
        except Exception:
            pass
        self.left_arm.disconnect()
        self.right_arm.disconnect()
