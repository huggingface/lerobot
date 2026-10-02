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

"""Shared batch-size-one interaction path for real robots and simulation."""

from .endpoint import Outcome, StepPacer, StepResult, TaskEndpoint
from .gym_endpoint import FeatureMapGymAdapter, GymEndpoint, GymEndpointAdapter, IdentityGymAdapter
from .record import (
    CallbackRecorder,
    DatasetRecorder,
    EpisodeDataset,
    EpisodeRecorder,
    ListRecorder,
    NullRecorder,
    StepRecord,
    step_record_to_dataset_frame,
)
from .robot_endpoint import OutcomeSource, ProcessorRobotAdapter, RobotEndpoint, RobotEndpointAdapter
from .runtime import ActionProvider, EpisodeResult, run_episode

__all__ = [
    "ActionProvider",
    "CallbackRecorder",
    "DatasetRecorder",
    "EpisodeDataset",
    "EpisodeRecorder",
    "EpisodeResult",
    "FeatureMapGymAdapter",
    "GymEndpoint",
    "GymEndpointAdapter",
    "IdentityGymAdapter",
    "ListRecorder",
    "NullRecorder",
    "Outcome",
    "OutcomeSource",
    "ProcessorRobotAdapter",
    "RobotEndpoint",
    "RobotEndpointAdapter",
    "StepRecord",
    "StepPacer",
    "StepResult",
    "TaskEndpoint",
    "run_episode",
    "step_record_to_dataset_frame",
]
