#!/usr/bin/env python

import logging
import time

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors.trossen import TrossenArmDriver

from ..teleoperator import Teleoperator
from .config_widow_ai_leader import WidowAILeaderConfig

logger = logging.getLogger(__name__)


class WidowAILeader(Teleoperator):
    config_class = WidowAILeaderConfig
    name = "widow_ai_leader"

    def __init__(self, config: WidowAILeaderConfig):
        super().__init__(config)
        self.config = config
        
        self.bus = TrossenArmDriver(
            port=self.config.port,
            model=self.config.model,
            velocity_limit_scale=self.config.velocity_limit_scale,
        )
        
        # Define motor names for compatibility
        self.motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_1", "wrist_2", "wrist_3", "gripper"]

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motor_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {f"{motor}.effort": float for motor in self.motor_names}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info(f"{self} is pre-calibrated, no calibration needed.")

    def configure(self) -> None:
        self.bus.initialize_for_teleoperation(is_leader=True)

    def setup_motors(self) -> None:
        logger.info(f"{self} motors are pre-configured.")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        positions = self.bus.read("Present_Position")
        
        # Convert numpy array to dict with motor names
        action = {}
        for i, motor in enumerate(self.motor_names):
            if i < len(positions):
                action[f"{motor}.pos"] = float(positions[i])
            else:
                action[f"{motor}.pos"] = 0.0
                
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Send effort feedback to the leader arm."""
        effort_feedback = []
        for motor in self.motor_names:
            effort_key = f"{motor}.effort"
            if effort_key in feedback:
                effort_feedback.append(-1 * self.config.effort_feedback_gain * feedback[effort_key])
            else:
                effort_feedback.append(0.0)

        if effort_feedback:
            self.bus.write("External_Efforts", effort_feedback)
            logger.debug(f"{self} sent effort feedback: {effort_feedback}")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
