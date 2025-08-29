#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
from typing_extensions import Unpack

from lerobot.constants import POSTPROCESSOR_DEFAULT_NAME, PREPROCESSOR_DEFAULT_NAME
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.processor_types import ProcessorFactoryKwargs
from lerobot.processor import (
    DeviceProcessor,
    NormalizerProcessor,
    RenameProcessor,
    RobotProcessor,
    ToBatchProcessor,
    UnnormalizerProcessor,
)


def make_pi0fast_pre_post_processors(
    config: PI0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    **kwargs: Unpack[ProcessorFactoryKwargs],
) -> tuple[RobotProcessor, RobotProcessor]:
    input_steps = [
        RenameProcessor(rename_map={}),  # To mimic the same processor as pretrained one
        NormalizerProcessor(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        ToBatchProcessor(),
        DeviceProcessor(device=config.device),
    ]
    output_steps = [
        DeviceProcessor(device="cpu"),
        UnnormalizerProcessor(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    # Extract processor kwargs
    preprocessor_kwargs = kwargs.get("preprocessor_kwargs") or {}
    postprocessor_kwargs = kwargs.get("postprocessor_kwargs") or {}

    return (
        RobotProcessor(
            steps=input_steps,
            name=PREPROCESSOR_DEFAULT_NAME,
            **preprocessor_kwargs,
        ),
        RobotProcessor(
            steps=output_steps,
            name=POSTPROCESSOR_DEFAULT_NAME,
            **postprocessor_kwargs,
        ),
    )
