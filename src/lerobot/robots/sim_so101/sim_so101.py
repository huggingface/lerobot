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

"""MuJoCo-backed SO-101 stand-in.

Implements the ``Robot`` contract against a MuJoCo simulation so the async RTC
rollout pipeline can run without physical hardware. Observation/action keys are
kept identical to ``so101_follower`` so the same policy and processors apply.
"""

import logging
from functools import cached_property

import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_sim_so101 import SimSO101Config

logger = logging.getLogger(__name__)

# Motor order is the source of truth shared with so_follower.py.
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BODY_MOTORS = MOTORS[:-1]
GRIPPER = "gripper"


class SimSO101(Robot):
    config_class = SimSO101Config
    name = "sim_so101"

    def __init__(self, config: SimSO101Config):
        super().__init__(config)
        self.config = config
        self._model = None
        self._data = None
        self._renderers: dict[str, object] = {}
        self._actuator_ids: dict[str, int] = {}
        self._qpos_addr: dict[str, int] = {}
        self._gripper_range: tuple[float, float] | None = None
        self._n_substeps = 1

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTORS}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {key: (cam.height, cam.width, 3) for key, cam in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def cameras(self) -> dict[str, object]:
        return self.config.cameras

    @property
    def is_connected(self) -> bool:
        return self._data is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import mujoco

        self._model = mujoco.MjModel.from_xml_path(self.config.mjcf_path)
        self._data = mujoco.MjData(self._model)

        # Resolve so101 motor -> MuJoCo actuator id + joint qpos address. Fail fast
        # if the scene's naming doesn't match joint_map.
        for motor, sim_name in self.config.joint_map.items():
            act_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, sim_name)
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, sim_name)
            if act_id < 0 or joint_id < 0:
                raise ValueError(
                    f"MJCF '{self.config.mjcf_path}' has no actuator/joint named '{sim_name}' "
                    f"(mapped from motor '{motor}'). Check SimSO101Config.joint_map."
                )
            self._actuator_ids[motor] = act_id
            self._qpos_addr[motor] = int(self._model.jnt_qposadr[joint_id])

        gripper_joint = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, self.config.joint_map[GRIPPER])
        self._gripper_range = tuple(float(x) for x in self._model.jnt_range[gripper_joint])

        self._n_substeps = max(1, round((1.0 / self.config.control_fps) / self._model.opt.timestep))

        for key, cam in self.config.cameras.items():
            self._renderers[key] = mujoco.Renderer(self._model, height=cam.height, width=cam.width)

        mujoco.mj_forward(self._model, self._data)
        logger.info(f"{self} connected (sim, {self._n_substeps} substeps per control step).")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict = {}
        for motor in BODY_MOTORS:
            rad = float(self._data.qpos[self._qpos_addr[motor]])
            obs[f"{motor}.pos"] = float(np.rad2deg(rad)) if self.config.use_degrees else rad
        obs[f"{GRIPPER}.pos"] = self._gripper_sim_to_robot(float(self._data.qpos[self._qpos_addr[GRIPPER]]))

        for key, renderer in self._renderers.items():
            renderer.update_scene(self._data, camera=self.config.cameras[key].mujoco_name)
            obs[key] = renderer.render()  # (H, W, 3) uint8

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        import mujoco

        for motor in BODY_MOTORS:
            val = action[f"{motor}.pos"]
            self._data.ctrl[self._actuator_ids[motor]] = float(np.deg2rad(val)) if self.config.use_degrees else val
        self._data.ctrl[self._actuator_ids[GRIPPER]] = self._gripper_robot_to_sim(action[f"{GRIPPER}.pos"])

        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        return {key: val for key, val in action.items() if key.endswith(".pos")}

    def _gripper_sim_to_robot(self, rad: float) -> float:
        lo, hi = self._gripper_range
        return 100.0 * (rad - lo) / (hi - lo)

    def _gripper_robot_to_sim(self, val: float) -> float:
        lo, hi = self._gripper_range
        return lo + (val / 100.0) * (hi - lo)

    @check_if_not_connected
    def disconnect(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers = {}
        self._model = None
        self._data = None
        logger.info(f"{self} disconnected.")
