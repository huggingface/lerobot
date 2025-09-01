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

from .batch_processor import AddBatchDimensionProcessorStep
from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .device_processor import DeviceProcessorStep
from .gym_action_processor import Numpy2TorchActionProcessorStep, Torch2NumpyActionProcessorStep
from .hil_processor import (
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    GripperPenaltyProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    RewardClassifierProcessorStep,
    TimeLimitProcessorStep,
)
from .joint_observations_processor import JointVelocityProcessorStep, MotorCurrentProcessorStep
from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep, hotswap_stats
from .observation_processor import VanillaObservationProcessorStep
from .pipeline import (
    ActionProcessorStep,
    DataProcessorPipeline,
    DoneProcessorStep,
    EnvTransition,
    IdentityProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RewardProcessorStep,
    RobotProcessorPipeline,
    TransitionKey,
    TruncatedProcessorStep,
)
from .rename_processor import RenameProcessorStep
from .tokenizer_processor import TokenizerProcessorStep

__all__ = [
    "ActionProcessorStep",
    "AddTeleopActionAsComplimentaryDataStep",
    "AddTeleopEventsAsInfoStep",
    "DeviceProcessorStep",
    "RewardProcessorStep",
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "EnvTransition",
    "GripperPenaltyProcessorStep",
    "IdentityProcessorStep",
    "ImageCropResizeProcessorStep",
    "InfoProcessorStep",
    "InterventionActionProcessorStep",
    "JointVelocityProcessorStep",
    "MapDeltaActionToRobotActionStep",
    "MotorCurrentProcessorStep",
    "NormalizerProcessorStep",
    "UnnormalizerProcessorStep",
    "hotswap_stats",
    "ObservationProcessorStep",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "RenameProcessorStep",
    "RewardClassifierProcessorStep",
    "DoneProcessorStep",
    "DataProcessorPipeline",
    "RobotProcessorPipeline",
    "PolicyProcessorPipeline",
    "AddBatchDimensionProcessorStep",
    "TokenizerProcessorStep",
    "TimeLimitProcessorStep",
    "Numpy2TorchActionProcessorStep",
    "Torch2NumpyActionProcessorStep",
    "TransitionKey",
    "TruncatedProcessorStep",
    "VanillaObservationProcessorStep",
]
