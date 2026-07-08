#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import subprocess
import time

from pyAgxArm import ArmModel, NeroFW, AgxArmFactory, create_agx_arm_config
from pyAgxArm.protocols.can_protocol.msgs.nero.default import ArmMsgMotionCtrl

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_nero_leader import NeroLeaderConfig

logger = logging.getLogger(__name__)

_FIRMWARE_MAP = {
    "DEFAULT": NeroFW.DEFAULT,
    "V111": NeroFW.V111,
}


def _reset_can(channel: str):
    subprocess.run(
        ["sudo", "ip", "link", "set", channel, "down"],
        capture_output=True,
        timeout=3,
    )
    subprocess.run(
        ["sudo", "ip", "link", "set", channel, "up", "type", "can", "bitrate", "1000000"],
        capture_output=True,
        timeout=3,
    )


class NeroLeader(Teleoperator):
    config_class = NeroLeaderConfig
    name = "nero_leader"

    def __init__(self, config: NeroLeaderConfig):
        super().__init__(config)
        self.config = config
        self._arm = None
        self._gripper = None
        self._connected = False
        self._drag_teach_active = False

    @property
    def action_features(self) -> dict[str, type]:
        features = {f"{name}.pos": float for name in self.config.joint_names}
        features["gripper.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if self.config.reset_can_on_connect:
            logger.info(f"Resetting CAN channel {self.config.port}")
            _reset_can(self.config.port)

        # 1. Create config
        fw = _FIRMWARE_MAP.get(self.config.firmware_version, NeroFW.V111)
        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=fw,
            interface=self.config.can_interface,
            channel=self.config.port,
        )

        # 2. Connect
        logger.info(f"Connecting Nero leader on {self.config.port}")
        self._arm = AgxArmFactory.create_arm(cfg)
        self._arm.connect()

        # 3. Enable arm (required for drag-teach gravity compensation)
        if self.config.enable_drag_teach:
            self._enable_arm()

        # 4. Lock motion mode
        self._arm.set_auto_set_motion_mode_enabled(False)

        # 5. Activate drag-teach (zero-force / gravity compensation)
        if self.config.enable_drag_teach:
            time.sleep(0.3)
            self._arm._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=0x01))
            self._drag_teach_active = True
            logger.info("Drag-teach mode activated (grag_teach_ctrl=0x01)")

        # 6. Initialize gripper
        self._init_gripper()

        self._connected = True
        logger.info(f"{self} connected.")

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        action = {}

        ja = self._arm.get_joint_angles()
        if ja is not None and ja.msg is not None:
            angles = list(ja.msg)[:7]
            for i, (name, angle) in enumerate(zip(self.config.joint_names, angles)):
                action[f"{name}.pos"] = float(angle * self.config.mirror_sign[i])

        if self._gripper is not None:
            gs = self._gripper.get_gripper_status()
            if gs is not None and gs.msg is not None:
                action["gripper.pos"] = float(gs.msg.value)

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        logger.info(f"Disconnecting {self}")

        # 1. Stop drag-teach
        if self._drag_teach_active and self._arm is not None:
            try:
                self._arm._send_msg(ArmMsgMotionCtrl(grag_teach_ctrl=0x02))
                self._drag_teach_active = False
                logger.info("Drag-teach mode deactivated (grag_teach_ctrl=0x02)")
            except Exception:
                pass

        # 2. Disable arm
        if self._arm is not None:
            try:
                self._arm.disable()
            except Exception:
                pass

        # 3. Disconnect
        if self._arm is not None:
            try:
                self._arm.disconnect()
            except Exception:
                pass

        self._arm = None
        self._gripper = None
        self._connected = False
        logger.info(f"{self} disconnected.")

    def _init_gripper(self):
        try:
            gripper = self._arm.init_effector(self._arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            time.sleep(0.5)
            try:
                gripper.reset_gripper()
                time.sleep(0.5)
            except Exception:
                pass
            self._gripper = gripper
            logger.info("Gripper initialized")
        except Exception as e:
            logger.warning(f"Gripper initialization failed: {e}")
            self._gripper = None

    def _enable_arm(self):
        enabled = False
        for attempt in range(self.config.enable_retries):
            if self._arm.enable():
                enabled = True
                logger.info(f"Arm enable() succeeded (attempt {attempt + 1})")
                break
            time.sleep(self.config.enable_retry_interval)
        if not enabled:
            raise ConnectionError(
                f"Failed to enable Nero leader arm after {self.config.enable_retries} attempts"
            )
