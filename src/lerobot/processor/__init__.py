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
from .converters import (
    batch_to_transition,
    create_transition,
    merge_transitions,
    transition_to_batch,
    transition_to_dataset_frame,
)
from .core import EnvTransition, TransitionKey
from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .device_processor import DeviceProcessor
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
    ComplementaryDataProcessorStep,
    DataProcessorPipeline,
    DoneProcessorStep,
    IdentityProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    ProcessorKwargs,
    ProcessorStep,
    ProcessorStepRegistry,
    RewardProcessorStep,
    TruncatedProcessorStep,
)
from .rename_processor import RenameProcessor
from .tokenizer_processor import TokenizerProcessor

__all__ = [
    "ActionProcessorStep",
    "AddTeleopActionAsComplimentaryData",
    "AddTeleopEventsAsInfo",
    "ComplementaryDataProcessorStep",
    "batch_to_transition",
    "create_transition",
    "DeviceProcessor",
    "DoneProcessorStep",
    "EnvTransition",
    "GripperPenaltyProcessor",
    "hotswap_stats",
    "IdentityProcessorStep",
    "ImageCropResizeProcessor",
    "InfoProcessorStep",
    "InterventionActionProcessor",
    "JointVelocityProcessor",
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "merge_transitions",
    "MotorCurrentProcessor",
    "NormalizerProcessor",
    "Numpy2TorchActionProcessor",
    "ObservationProcessorStep",
    "ProcessorKwargs",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "RenameProcessor",
    "RewardClassifierProcessor",
    "RewardProcessorStep",
    "DataProcessorPipeline",
    "TimeLimitProcessor",
    "ToBatchProcessor",
    "TokenizerProcessor",
    "Torch2NumpyActionProcessor",
    "transition_to_batch",
    "transition_to_dataset_frame",
    "TransitionKey",
    "TruncatedProcessorStep",
    "UnnormalizerProcessor",
    "VanillaObservationProcessor",
]
