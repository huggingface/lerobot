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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import Any

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from ..metal_follower import MetalFollower, MetalFollowerConfig, MetalFollowerConfigBase
from ..robot import Robot
from .config_bi_metal_follower import BiMetalFollowerConfig

logger = logging.getLogger(__name__)


class BiMetalFollower(BimanualMixin, Robot):
    """Bimanual Metal arm follower.

    Composes two single-arm :class:`MetalFollower` instances, one per CAN bus.
    Observation and action keys of each arm are namespaced with a ``left_`` /
    ``right_`` prefix; top-level cameras keep their unprefixed keys.
    """

    config_class = BiMetalFollowerConfig
    name = "bi_metal_follower"

    def __init__(self, config: BiMetalFollowerConfig):
        super().__init__(config)
        self.config = config

        # Top-level cameras are opened by `left_arm` for convenience, but their
        # keys stay unprefixed in observations (tracked via `_top_level_cam_keys`).
        self._top_level_cam_keys = set(config.cameras)
        _collisions = self._top_level_cam_keys & (
            set(config.left_arm_config.cameras) | set(config.right_arm_config.cameras)
        )
        if _collisions:
            raise ValueError(
                f"Top-level camera names collide with per-arm camera names: {sorted(_collisions)}"
            )
        left_arm_cameras = {**config.left_arm_config.cameras, **config.cameras}

        def _arm_config(arm_config: MetalFollowerConfigBase, side: str) -> MetalFollowerConfig:
            return MetalFollowerConfig(
                id=f"{config.id}_{side}" if config.id else None,
                calibration_dir=config.calibration_dir,
                port=arm_config.port,
                can_interface=arm_config.can_interface,
                can_bitrate=arm_config.can_bitrate,
                use_can_fd=arm_config.use_can_fd,
                can_data_bitrate=arm_config.can_data_bitrate,
                motor_can_ids=arm_config.motor_can_ids,
                joint_limits=arm_config.joint_limits,
                gains=arm_config.gains,
                startup_sync_speed_deg=arm_config.startup_sync_speed_deg,
                startup_sync_tolerance_deg=arm_config.startup_sync_tolerance_deg,
                max_relative_target=arm_config.max_relative_target,
                velocity_feedforward=arm_config.velocity_feedforward,
                velocity_ff_alpha=arm_config.velocity_ff_alpha,
                velocity_ff_max_deg_s=arm_config.velocity_ff_max_deg_s,
                disable_torque_on_disconnect=arm_config.disable_torque_on_disconnect,
                cameras=left_arm_cameras if side == "left" else arm_config.cameras,
            )

        self.left_arm = MetalFollower(_arm_config(config.left_arm_config, "left"))
        self.right_arm = MetalFollower(_arm_config(config.right_arm_config, "right"))

        # Only for compatibility with parts of the codebase that expect `robot.cameras`.
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

        # The two arms sit on physically independent CAN buses with independent
        # DamiaoMotorsBus instances, so their blocking I/O runs truly concurrently
        # (the GIL is released during the socket/serial waits). A persistent 2-worker
        # pool avoids per-tick thread creation. Shut down in disconnect().
        self._io_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='bi_metal_io')

    def _run_both(self, left_fn: Callable[[], Any], right_fn: Callable[[], Any]) -> tuple[Any, Any]:
        """Run the left and right arm I/O calls concurrently and return (left, right).

        Each arm owns a separate CAN bus and DamiaoMotorsBus instance, so there is no
        shared mutable state between the two callables. Running them on the pool overlaps
        the two buses' round-trips, roughly halving the per-tick wall time versus serial.
        Exceptions from either arm propagate via future.result().
        """
        left_future = self._io_pool.submit(left_fn)
        right_future = self._io_pool.submit(right_fn)
        # Always wait on both futures so no arm's I/O is left dangling, even if one raises.
        left_result = left_future.result()
        right_result = right_future.result()
        return left_result, right_result

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm._motors_ft.items()},
            **{f"right_{k}": v for k, v in self.right_arm._motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        out: dict[str, tuple] = {}
        for k, v in self.left_arm._cameras_ft.items():
            out[k if k in self._top_level_cam_keys else f"left_{k}"] = v
        for k, v in self.right_arm._cameras_ft.items():
            out[f"right_{k}"] = v
        return out

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @check_if_not_connected
    def disconnect(self) -> None:
        # Stop the I/O pool first so no worker touches a bus mid-disconnect, then let
        # BimanualMixin.disconnect() close both arms.
        self._io_pool.shutdown(wait=True)
        super().disconnect()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        left_obs, right_obs = self._run_both(self.left_arm.get_observation, self.right_arm.get_observation)
        obs_dict: RobotObservation = {}
        for k, v in left_obs.items():
            obs_dict[k if k in self._top_level_cam_keys else f"left_{k}"] = v
        for k, v in right_obs.items():
            obs_dict[f"right_{k}"] = v
        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        sent_action_left, sent_action_right = self._run_both(
            lambda: self.left_arm.send_action(left_action),
            lambda: self.right_arm.send_action(right_action),
        )

        return {
            **{f"left_{k}": v for k, v in sent_action_left.items()},
            **{f"right_{k}": v for k, v in sent_action_right.items()},
        }
