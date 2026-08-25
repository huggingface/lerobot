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
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from ..metal_leader import MetalLeader, MetalLeaderConfig, MetalLeaderConfigBase
from ..teleoperator import Teleoperator
from .config_bi_metal_leader import BiMetalLeaderConfig

logger = logging.getLogger(__name__)


class BiMetalLeader(BimanualMixin, Teleoperator):
    """Bimanual Metal arm leader.

    Composes two single-arm :class:`MetalLeader` instances, one per CAN bus. Action keys are
    namespaced with a ``left_`` / ``right_`` prefix to match :class:`BiMetalFollower`.

    Each arm runs its own gravity-compensation thread against its own bus, so the two are fully
    independent — there is no shared state to serialize between them.
    """

    config_class = BiMetalLeaderConfig
    name = "bi_metal_leader"

    def __init__(self, config: BiMetalLeaderConfig):
        super().__init__(config)
        self.config = config

        def _arm_config(arm_config: MetalLeaderConfigBase, side: str) -> MetalLeaderConfig:
            return MetalLeaderConfig(
                id=f"{config.id}_{side}" if config.id else None,
                calibration_dir=config.calibration_dir,
                port=arm_config.port,
                can_interface=arm_config.can_interface,
                can_bitrate=arm_config.can_bitrate,
                use_can_fd=arm_config.use_can_fd,
                can_data_bitrate=arm_config.can_data_bitrate,
                motor_can_ids=arm_config.motor_can_ids,
                urdf_path=arm_config.urdf_path,
                gravity_hz=arm_config.gravity_hz,
                leader_kd=arm_config.leader_kd,
                use_velocity_feedforward=arm_config.use_velocity_feedforward,
                friction_scale=arm_config.friction_scale,
                velocity_deadzone_rad_s=arm_config.velocity_deadzone_rad_s,
                gripper_friction_scale=arm_config.gripper_friction_scale,
                hold_kp_on_disconnect=arm_config.hold_kp_on_disconnect,
                hold_kd_on_disconnect=arm_config.hold_kd_on_disconnect,
            )

        self.left_arm = MetalLeader(_arm_config(config.left_arm_config, "left"))
        self.right_arm = MetalLeader(_arm_config(config.right_arm_config, "right"))

        # Each arm owns a separate CAN bus and DamiaoMotorsBus instance, so their blocking reads
        # overlap (the GIL is released during the socket wait). This matters more here than for a
        # torque-off leader: each arm's get_action may also wait on its own gravity thread's bus
        # lock, so running the two serially would stack both waits inside one teleop tick.
        # Shut down in disconnect().
        self._io_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bi_metal_leader_io")

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm.action_features.items()},
            **{f"right_{key}": value for key, value in self.right_arm.action_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is done with the manufacturer's tools for Damiao CAN motors."
        )

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        left_future = self._io_pool.submit(self.left_arm.get_action)
        right_future = self._io_pool.submit(self.right_arm.get_action)
        # Always wait on both futures so no arm's read is left dangling if the other raises.
        left_action = left_future.result()
        right_action = right_future.result()

        return {
            **{f"left_{key}": value for key, value in left_action.items()},
            **{f"right_{key}": value for key, value in right_action.items()},
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        raise NotImplementedError("Force feedback is not implemented for the metal leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        # Stop the I/O pool first so no worker touches a bus mid-disconnect, then let
        # BimanualMixin.disconnect() stop each arm's gravity thread and close its bus.
        self._io_pool.shutdown(wait=True)
        super().disconnect()
