import logging

from ..keyboard import KeyboardTeleop
from lerobot.teleoperators import Teleoperator
from functools import cached_property
from .config_lekiwi_leader import LekiwiLeaderConfig
from lerobot.teleoperators.so101_leader import SO101Leader
import numpy as np

logger = logging.getLogger(__name__)

class LekiwiLeader(Teleoperator):

    config_class = LekiwiLeaderConfig
    name = "lekiwi_leader"

    speed_levels = [
        {"xy": 0.1, "theta": 30},  # slow
        {"xy": 0.2, "theta": 60},  # medium
        {"xy": 0.3, "theta": 90},  # fast
    ]
    speed_index = 0  # Start at slowwwwwwwwwwwwwwwwwwww

    def __init__(self, config: LekiwiLeaderConfig):
        super().__init__(config)
        self.keyboard = KeyboardTeleop(config)
        self.leader_arm = SO101Leader(config)
        self.teleop_keys = config.teleop_keys

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "arm_shoulder_pan.pos",
                "arm_shoulder_lift.pos",
                "arm_elbow_flex.pos",
                "arm_wrist_flex.pos",
                "arm_wrist_roll.pos",
                "arm_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    def action_features(self)-> dict[str, type]:
        return self._state_ft()

    @property
    def is_connected(self) -> bool:
        return self.leader_arm.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.leader_arm.bus.is_calibrated

    def calibrate(self) -> None:
        self.leader_arm.calibrate()

    def connect(self):
        self.keyboard.connect()
        self.leader_arm.connect()

    def disconnect(self) -> None:
        self.keyboard.disconnect()
        self.leader_arm.disconnect()

    def configure(self) -> None:
        self.leader_arm.configure()

    def setup_motors(self) -> None:
        self.leader_arm.setup_motors()

    def get_action(self)->dict[str, float]:
        arm_action = self.leader_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
        keyboard_keys = self.keyboard.get_action()
        base_action = self._from_keyboard_to_base_action(keyboard_keys)
        action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
        return action

    def _from_keyboard_to_base_action(self, pressed_keys: np.ndarray):
        # Speed control
        if self.teleop_keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, 2)
        if self.teleop_keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)
        speed_setting = self.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # e.g. 0.1, 0.25, or 0.4
        theta_speed = speed_setting["theta"]  # e.g. 30, 60, or 90

        x_cmd = 0.0  # m/s forward/backward
        y_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation

        if self.teleop_keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if self.teleop_keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if self.teleop_keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if self.teleop_keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if self.teleop_keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if self.teleop_keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed
        return {
            "x.vel": x_cmd,
            "y.vel": y_cmd,
            "theta.vel": theta_cmd,
        }

    def feedback_features(self)-> dict[str, type]:
        raise NotImplementedError
        return {}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError
        pass

#        log_rerun_data(observation, {**arm_action, **base_action})
