import logging

import numpy as np
import pygame

from lerobot.teleoperators.teleoperator import Teleoperator

from .configuration_rudder_pedal import RudderPedalConfig

logger = logging.getLogger(__name__)


class RudderPedal(Teleoperator):
    """
    Teleoperator for Logitech/Saitek rudder pedals controlling a differential
    drive wheel base. Produces base_left_wheel.vel and base_right_wheel.vel
    actions to be merged with arm teleop actions before sending to the robot.
    """
    config_class = RudderPedalConfig
    name = "rudder_pedal"

    def __init__(self, config: RudderPedalConfig):
        super().__init__(config)
        self.config = config
        self.joystick = None

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "base_left_wheel.vel": float,
            "base_right_wheel.vel": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.joystick is not None

    def connect(self, calibrate: bool = False) -> None:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No joystick detected. Make sure the rudder pedals are plugged in."
            )
        self.joystick = pygame.joystick.Joystick(self.config.joystick_index)
        self.joystick.init()
        logger.info(f"Rudder pedal connected: {self.joystick.get_name()}")

    @property
    def is_calibrated(self) -> bool:
        return True  # No calibration needed for pedals

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.config.deadzone else value

    def get_action(self) -> dict[str, float]:
        pygame.event.pump()

        right_brake = self.joystick.get_axis(0)  # -1=released, +1=pressed → forward
        left_brake  = self.joystick.get_axis(1)  # -1=released, +1=pressed → reverse
        rudder      = self.joystick.get_axis(2)  # 0=center, ±1=full turn

        # Normalize brakes from [-1, +1] to [0, 1]
        reverse  = (left_brake  + 1) / 2
        forward  = (right_brake + 1) / 2
        throttle = self._apply_deadzone(forward - reverse)  # [-1, 1]
        turn     = self._apply_deadzone(rudder)              # [-1, 1]

        left_vel  = float(np.clip(throttle + turn, -1.0, 1.0) * self.config.max_speed)
        right_vel = float(np.clip(throttle - turn, -1.0, 1.0) * self.config.max_speed)

        return {
            "base_left_wheel.vel":  left_vel,
            "base_right_wheel.vel": right_vel,
        }

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        if self.joystick is not None:
            self.joystick.quit()
            self.joystick = None
        pygame.joystick.quit()
        pygame.quit()
        logger.info("Rudder pedal disconnected.")
