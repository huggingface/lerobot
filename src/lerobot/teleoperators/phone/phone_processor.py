# !/usr/bin/env python

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

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey
from lerobot.teleoperators.phone.config_phone import PhoneOS


@ProcessorStepRegistry.register("map_phone_action_to_robot_action")
@dataclass
class MapPhoneActionToRobotAction:
    """
    Map calibrated phone pose (actions) to the inputs for robot actions

    Expected input ACTION keys:
    {
        "action.phone.enabled": bool,
        "action.phone.pos": np.ndarray,
        "action.phone.rot": Rotation,
        "action.phone.raw_inputs": dict,
    }

    Output ACTION keys:
    {
        "action.enabled": bool,
        "action.ee.{x,y,z,wx,wy,wz}" : float
        "action.gripper": float,
        "action.x": float,
        "action.y": float,
        "action.theta": float,
    }
    """

    platform: PhoneOS
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_rot: Rotation | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        act = transition.get(TransitionKey.ACTION) or {}

        # Pop them from the action
        enabled = act.pop("action.phone.enabled", 0)
        pos = act.pop("action.phone.pos", None)
        rot = act.pop("action.phone.rot", None)
        inputs = act.pop("action.phone.raw_inputs", {})

        if pos is None or rot is None:
            return transition

        # compute per-frame deltas in the phone frame
        if self._last_pos is None or self._last_rot is None:
            dpos = np.zeros(3)
            drot = Rotation.identity()
        else:
            dpos = pos - self._last_pos
            drot = self._last_rot.inv() * rot

        self._last_pos, self._last_rot = pos, rot

        rotvec = drot.as_rotvec()

        # Map certain inputs to certain actions
        if self.platform == PhoneOS.IOS:
            gripper = float(inputs.get("a3", 0.0))
            x = float(inputs.get("a1", 0.0))
            y = float(inputs.get("a2", 0.0))
            theta = float(inputs.get("a7", 0.0))
        else:
            sc = float(inputs.get("scale", 1.0))
            gripper = max(min(sc - 1.0, 1.0), -1.0)
            x = y = theta = 0.0

        # For some actions we need to invert the axis
        act.update(
            {
                "action.enabled": enabled,
                "action.target_x": -dpos[1] if enabled else 0.0,
                "action.target_y": dpos[0] if enabled else 0.0,
                "action.target_z": dpos[2] if enabled else 0.0,
                "action.target_wx": rotvec[0] if enabled else 0.0,
                "action.target_wy": rotvec[1] if enabled else 0.0,
                "action.target_wz": rotvec[2] if enabled else 0.0,
                "action.gripper": gripper,  # Still send gripper action when disabled
                "action.x": x if enabled else 0.0,
                "action.y": y if enabled else 0.0,
                "action.theta": theta if enabled else 0.0,
            }
        )

        transition[TransitionKey.ACTION] = act
        return transition

    def reset(self):
        self._last_pos = None
        self._last_rot = None
