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

import torch

from lerobot.envs.configs import (
    GripperConfig,
    HILSerlProcessorConfig,
    HILSerlRobotEnvConfig,
    InverseKinematicsConfig,
)
from lerobot.lerobot_types import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.processor.delta_action_processor import (
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
)
from lerobot.processor.hil_processor import InterventionActionProcessorStep
from lerobot.rl.gym_manipulator import _gripper_enabled, make_processors


class _Teleop:
    def get_action(self):
        return {}

    def get_teleop_events(self):
        return {}


class _Env:
    def __init__(self):
        self.robot = type("Robot", (), {})()
        self.robot.bus = type("Bus", (), {})()
        self.robot.bus.motors = {"shoulder_pan": None, "gripper": None}


def _config(use_gripper: bool | None) -> HILSerlRobotEnvConfig:
    return HILSerlRobotEnvConfig(
        processor=HILSerlProcessorConfig(
            gripper=None if use_gripper is None else GripperConfig(use_gripper=use_gripper),
            inverse_kinematics=InverseKinematicsConfig(
                urdf_path="robot.urdf",
                target_frame_name="gripper_frame_link",
                end_effector_bounds={"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
                end_effector_step_sizes={"x": 0.01, "y": 0.01, "z": 0.01},
            ),
        )
    )


def _ik_steps(cfg, monkeypatch):
    monkeypatch.setattr("lerobot.rl.gym_manipulator.RobotKinematics", lambda *args, **kwargs: object())
    _, action_processor = make_processors(_Env(), _Teleop(), cfg)
    return action_processor.steps


def test_unset_gripper_config_enables_the_action_pipeline(monkeypatch):
    """A missing gripper config enables the gripper, matching the environment."""
    assert _gripper_enabled(HILSerlProcessorConfig()) is True

    steps = _ik_steps(_config(None), monkeypatch)
    tensor_step = next(step for step in steps if isinstance(step, MapTensorToDeltaActionDictStep))
    intervention = next(step for step in steps if isinstance(step, InterventionActionProcessorStep))
    assert tensor_step.use_gripper is True
    assert intervention.use_gripper is True

    transition = tensor_step(create_transition(action=torch.tensor([0.0, 0.0, 0.0, 2.0])))
    delta_step = next(step for step in steps if isinstance(step, MapDeltaActionToRobotActionStep))
    transition = delta_step(transition)
    assert transition[TransitionKey.ACTION]["gripper_vel"] == 2.0


def test_explicitly_disabled_gripper_stays_disabled(monkeypatch):
    """use_gripper=False is still honored on the inverse-kinematics steps."""
    steps = _ik_steps(_config(False), monkeypatch)
    tensor_step = next(step for step in steps if isinstance(step, MapTensorToDeltaActionDictStep))
    assert tensor_step.use_gripper is False
