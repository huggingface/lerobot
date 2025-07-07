#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.processor import (
    NormalizerProcessor,
    RobotProcessor,
    UnnormalizerProcessor,
)


def make_diffusion_processor(
    config: DiffusionConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[RobotProcessor, RobotProcessor]:
    input_steps = [
        NormalizerProcessor(
            features=config.input_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        NormalizerProcessor(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    output_steps = [
        UnnormalizerProcessor(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    return RobotProcessor(steps=input_steps, name="diffusion_preprocessor"), RobotProcessor(
        steps=output_steps, name="diffusion_postprocessor"
    )
