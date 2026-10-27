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

"""SuperMimic whole-body motion tracking: actor, reference motions and robot dynamics spec.

These pieces are shared by the deployment controller
(:class:`lerobot.robots.unitree_g1.controllers.supermimic_tracking.SuperMimicController`)
and the world-model training backend.
"""

from .export import SuperMimicExport, load_export
from .networks import Critic, RunningNorm, SuperMimicActor, WorldModel
from .observation import OBS_DIM, ObservationHistory, proprio_observation
from .reference import GOAL_FRAME_DIM, MotionReference, load_motion_csv
from .robot_spec import G1_JOINT_NAMES, G1_TRACKED_BODIES, RobotDynamicsSpec, g1_29dof_spec

__all__ = [
    "Critic",
    "G1_JOINT_NAMES",
    "G1_TRACKED_BODIES",
    "GOAL_FRAME_DIM",
    "MotionReference",
    "OBS_DIM",
    "ObservationHistory",
    "RobotDynamicsSpec",
    "RunningNorm",
    "SuperMimicActor",
    "SuperMimicExport",
    "WorldModel",
    "g1_29dof_spec",
    "load_export",
    "load_motion_csv",
    "proprio_observation",
]
