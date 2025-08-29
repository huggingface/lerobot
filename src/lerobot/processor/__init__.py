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
from .delta_action_processor import MapDeltaActionToRobotAction, MapTensorToDeltaActionDict
from .device_processor import DeviceProcessorStep
from .gym_action_processor import Numpy2TorchActionProcessor, Torch2NumpyActionProcessor
from .hil_processor import (
    AddTeleopActionAsComplimentaryData,
    AddTeleopEventsAsInfo,
    GripperPenaltyProcessor,
    ImageCropResizeProcessor,
    InterventionActionProcessor,
    RewardClassifierProcessor,
    TimeLimitProcessor,
)
from .joint_observations_processor import JointVelocityProcessor, MotorCurrentProcessor
from .normalize_processor import NormalizerProcessor, UnnormalizerProcessor, hotswap_stats
from .observation_processor import VanillaObservationProcessor
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
from .rename_processor import RenameProcessor
from .tokenizer_processor import TokenizerProcessorStep

__all__ = [
    "ActionProcessorStep",
    "AddTeleopActionAsComplimentaryData",
    "AddTeleopEventsAsInfo",
    "DeviceProcessorStep",
    "RewardProcessorStep",
    "MapDeltaActionToRobotAction",
    "MapTensorToDeltaActionDict",
    "EnvTransition",
    "GripperPenaltyProcessor",
    "IdentityProcessorStep",
    "ImageCropResizeProcessor",
    "InfoProcessorStep",
    "InterventionActionProcessor",
    "JointVelocityProcessor",
    "MapDeltaActionToRobotAction",
    "MotorCurrentProcessor",
    "NormalizerProcessor",
    "UnnormalizerProcessor",
    "hotswap_stats",
    "ObservationProcessorStep",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "RenameProcessor",
    "RewardClassifierProcessor",
    "DoneProcessorStep",
    "DataProcessorPipeline",
    "RobotProcessorPipeline",
    "PolicyProcessorPipeline",
    "AddBatchDimensionProcessorStep",
    "TokenizerProcessorStep",
    "TimeLimitProcessor",
    "Numpy2TorchActionProcessor",
    "Torch2NumpyActionProcessor",
    "TransitionKey",
    "TruncatedProcessorStep",
    "VanillaObservationProcessor",
]
