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

from .batch_processor import ToBatchProcessor
from .delta_action_processor import MapDeltaActionToRobotAction, MapTensorToDeltaActionDict
from .device_processor import DeviceProcessor
from .hil_processor import (
    AddTeleopActionAsComplimentaryData,
    AddTeleopEventsAsInfo,
    GripperPenaltyProcessor,
    ImageCropResizeProcessor,
    InterventionActionProcessor,
    Numpy2TorchActionProcessor,
    RewardClassifierProcessor,
    TimeLimitProcessor,
    Torch2NumpyActionProcessor,
)
from .joint_observations_processor import JointVelocityProcessor, MotorCurrentProcessor
from .normalize_processor import NormalizerProcessor, UnnormalizerProcessor, hotswap_stats
from .observation_processor import VanillaObservationProcessor
from .pipeline import (
    ActionProcessor,
    DoneProcessor,
    EnvTransition,
    IdentityProcessor,
    InfoProcessor,
    ObservationProcessor,
    ProcessorStep,
    ProcessorStepRegistry,
    RewardProcessor,
    RobotProcessor,
    TransitionKey,
    TruncatedProcessor,
)
from .rename_processor import RenameProcessor
from .tokenizer_processor import TokenizerProcessor

__all__ = [
    "ActionProcessor",
    "AddTeleopActionAsComplimentaryData",
    "AddTeleopEventsAsInfo",
    "DeviceProcessor",
    "DoneProcessor",
    "MapDeltaActionToRobotAction",
    "MapTensorToDeltaActionDict",
    "EnvTransition",
    "GripperPenaltyProcessor",
    "IdentityProcessor",
    "ImageCropResizeProcessor",
    "InfoProcessor",
    "InterventionActionProcessor",
    "JointVelocityProcessor",
    "MapDeltaActionToRobotAction",
    "MotorCurrentProcessor",
    "NormalizerProcessor",
    "UnnormalizerProcessor",
    "hotswap_stats",
    "ObservationProcessor",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "RenameProcessor",
    "RewardClassifierProcessor",
    "RewardProcessor",
    "RobotProcessor",
    "ToBatchProcessor",
    "TokenizerProcessor",
    "TimeLimitProcessor",
    "Numpy2TorchActionProcessor",
    "Torch2NumpyActionProcessor",
    "TransitionKey",
    "TruncatedProcessor",
    "VanillaObservationProcessor",
]
