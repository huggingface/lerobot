import logging
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.keyboard_input import create_key_listener

from .config_ax_arm import AXArmConfig
from .urdf_mapping import ARM_JOINTS, REFERENCE_URDF_DEG, URDF_LIMITS_DEG

logger = logging.getLogger(__name__)


class AXArm(Robot):
    """A 4-DoF arm driven by Dynamixel AX-series servos over Protocol 1.0.

    Protocol 1.0 has no Sync Read, no Operating_Mode register and no homing offset, so this robot reads
    positions sequentially and calibrates by recording the range of motion only.
    """

    config_class = AXArmConfig
    name = "ax_arm"

    def __init__(self, config: AXArmConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = DynamixelMotorsBus(
            port=self.config.port,
            # NOTE: dict order (pan, lift, elbow, gripper) must stay aligned with the URDF joints
            # (robot_joint_1/2/3) and keep the gripper last for the kinematics pipeline. Only the
            # motor IDs below reflect the physical bus wiring.
            motors={
                "shoulder_pan": Motor(3, "ax-12a", norm_mode_body),
                "shoulder_lift": Motor(4, "ax-12a", norm_mode_body),
                "elbow_flex": Motor(2, "ax-12a", norm_mode_body),
                "gripper": Motor(1, "ax-12a", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
            protocol_version=1,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras}

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
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info("No matching calibration found, running calibration.")
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # Protocol 1.0 cannot read back homing_offset/drive_mode (we repurpose them to store the
        # URDF mapping: tick_ref and sign), so only the range limits are actually stored on the
        # servos. Compare just those to decide whether the on-file calibration matches the hardware.
        if not self.calibration:
            return False
        hw = self.bus.read_calibration()
        return all(
            self.calibration[m].range_min == hw[m].range_min
            and self.calibration[m].range_max == hw[m].range_max
            for m in self.bus.motors
        )

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")

        # This arm cannot be backdriven with torque off, so each joint is jogged with the keyboard
        # (torque on). Arm joints capture a reference pose (known URDF angle) plus the lower/upper
        # travel limits; the reference tick and mounting sign encode the URDF mapping (see urdf_mapping).
        captured = self._record_calibration()

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            c = captured[motor]
            if motor in ARM_JOINTS:
                lower, upper = c["lower limit"], c["upper limit"]
                # sign: +1 if jogging toward the URDF-upper limit increased ticks, else -1.
                drive_mode = 0 if upper >= lower else 1
                self.calibration[motor] = MotorCalibration(
                    id=m.id,
                    drive_mode=drive_mode,
                    homing_offset=c["reference"],  # tick at the URDF reference pose
                    range_min=min(lower, upper),
                    range_max=max(lower, upper),
                )
            else:  # gripper: plain range, no URDF mapping
                lo, hi = c["min"], c["max"]
                self.calibration[motor] = MotorCalibration(
                    id=m.id, drive_mode=0, homing_offset=0, range_min=min(lo, hi), range_max=max(lo, hi)
                )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    # Raw ticks the selected joint moves per keypress (~3° on an AX-12A).
    _JOG_STEP = 10

    def _capture_plan(self) -> list[tuple[str, str, float | None]]:
        """Ordered (motor, label, target_urdf_deg) captures. Arm joints: reference + lower/upper
        URDF limits; gripper: plain min/max. ``target_urdf_deg`` is None when there is no URDF angle."""
        plan: list[tuple[str, str, float | None]] = []
        for motor in self.bus.motors:
            if motor in ARM_JOINTS:
                lower, upper = URDF_LIMITS_DEG[motor]
                plan.append((motor, "reference", REFERENCE_URDF_DEG[motor]))
                plan.append((motor, "lower limit", lower))
                plan.append((motor, "upper limit", upper))
            else:
                plan += [(motor, "min", None), (motor, "max", None)]
        return plan

    def _record_calibration(self) -> dict[str, dict[str, int]]:
        # Protocol 1.0 has no Sync Read, so positions are read sequentially, one motor at a time.
        motor_names = list(self.bus.motors)
        max_pos = min(self.bus.model_resolution_table[m.model] for m in self.bus.motors.values()) - 1

        # Temporarily widen the angle limits to full travel so pre-existing (possibly narrow) limits do
        # not cap the reachable range; write_calibration() sets them to the recorded min/max afterwards.
        for motor in motor_names:
            self.bus.write("CW_Angle_Limit", motor, 0)
            self.bus.write("CCW_Angle_Limit", motor, max_pos)
        self.bus.enable_torque()

        plan = self._capture_plan()
        captured: dict[str, dict[str, int]] = {motor: {} for motor in motor_names}
        state = {"i": 0, "target": 0, "capture": False, "done": False}
        state["target"] = int(self.bus.read("Present_Position", plan[0][0], normalize=False))

        def on_key(name: str) -> None:
            key = name.lower()
            if key == "up":
                state["target"] = min(max_pos, state["target"] + self._JOG_STEP)
            elif key == "down":
                state["target"] = max(0, state["target"] - self._JOG_STEP)
            elif key == "enter":
                state["capture"] = True

        listener = create_key_listener(on_key, controls_help="up/down=jog, enter=capture")
        if listener is None:
            raise RuntimeError(
                "Keyboard calibration requires an interactive terminal with a usable key listener."
            )

        print(
            "Jog each joint to the requested pose and press ENTER to capture it.\n"
            "  up/down = jog | enter = capture"
        )
        try:
            while not state["done"]:
                motor, label, target_deg = plan[state["i"]]
                self.bus.write("Goal_Position", motor, state["target"], normalize=False)
                pos = int(self.bus.read("Present_Position", motor, normalize=False))
                hint = f" (URDF {target_deg:+.0f} deg)" if target_deg is not None else ""
                print(
                    f"  [{state['i'] + 1}/{len(plan)}] jog '{motor}' to {label}{hint}: {pos:4d} tick   ",
                    end="\r",
                    flush=True,
                )

                if state["capture"]:
                    state["capture"] = False
                    captured[motor][label] = pos
                    print(f"  [{state['i'] + 1}/{len(plan)}] {motor} {label}{hint} = {pos} tick")
                    state["i"] += 1
                    state["done"] = state["i"] >= len(plan)
                    if not state["done"]:
                        state["target"] = int(
                            self.bus.read("Present_Position", plan[state["i"]][0], normalize=False)
                        )
                time.sleep(0.02)
        finally:
            listener.stop()
            print()

        for motor, c in captured.items():
            if len(set(c.values())) < len(c):
                raise ValueError(f"Motor '{motor}' has duplicate captured ticks: {c}")
        return captured

    def configure(self) -> None:
        # AX-series has no Operating_Mode/PID registers; configure_motors only lowers the return delay time.
        with self.bus.torque_disabled():
            self.bus.configure_motors()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        # Protocol 1.0 has no Sync Read, so read each motor sequentially.
        obs_dict = {f"{motor}.pos": self.bus.read("Present_Position", motor) for motor in self.bus.motors}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if isinstance(key, str) and key.endswith(".pos")
        }
        if not goal_pos:
            return {}

        if self.config.max_relative_target is not None:
            present_pos = {motor: self.bus.read("Present_Position", motor) for motor in goal_pos}
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Sync Write is available on Protocol 1.0 (only Sync Read and Broadcast Ping are not).
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
