#!/usr/bin/env python

import logging
import time

from lerobot.motors.piper.piper import PiperMotorsBus, PiperMotorsBusConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_piper_leader import PIPERLeaderConfig

logger = logging.getLogger(__name__)


class PIPERLeader(Teleoperator):
    config_class = PIPERLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PIPERLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = PiperMotorsBus(
            PiperMotorsBusConfig(
                can_name="can_master",
                motors={
                    "joint_1": (1, "agilex_piper"),
                    "joint_2": (2, "agilex_piper"),
                    "joint_3": (3, "agilex_piper"),
                    "joint_4": (4, "agilex_piper"),
                    "joint_5": (5, "agilex_piper"),
                    "joint_6": (6, "agilex_piper"),
                    "gripper": (7, "agilex_piper"),
                },
            )
        )
        self.logs = {}
        self._is_connected = False
        self._is_calibrated = False

    @property
    def action_features(self) -> dict[str, type]:
        """用于 record/replay 的动作描述"""
        motor_order = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        return {f"{motor}.pos": float for motor in motor_order}

    @property
    def feedback_features(self) -> dict[str, type]:
        """当前主臂没有反馈功能"""
        return {}

    def configure(self, **kwargs):
        # 暂不需要配置逻辑
        pass

    def send_feedback(self, *args, **kwargs):
        # 暂无反馈功能
        pass

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def connect(self, _is_calibrate: bool = True) -> None:
        """连接主臂"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect(enable=True)
        self._is_connected = True

        if _is_calibrate:
            self.calibrate()

        logger.info(f"{self} connected.")

    def calibrate(self):
        """应用主臂标定（回零）"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.apply_calibration_master()
        self._is_calibrated = True

    def get_action(self) -> dict[str, float]:
        """获取主臂当前动作（单位转为 rad）"""
        start = time.perf_counter()
        action_raw = self.bus.read()  # 原始单位 0.001°
        joint_factor = 57324.840764  # 度转弧度比例因子（可调）
        action = {
            f"{motor}.pos": val / joint_factor if motor != "gripper" else val / 1_000_000
            for motor, val in action_raw.items()
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def disconnect(self) -> None:
        """断开主臂连接"""
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.safe_disconnect()
        self._is_connected = False
