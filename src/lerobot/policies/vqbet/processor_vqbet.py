#!/usr/bin/env python

# Copyright 2024 Seungjae Lee and Yibin Wang and Haritheja Etukuru
# and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto
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

from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.processor import (
    NormalizerProcessor,
    RobotProcessor,
    UnnormalizerProcessor,
)


def make_vqbet_processor(
    config: VQBeTConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
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
    return RobotProcessor(steps=input_steps, name="vqbet_preprocessor"), RobotProcessor(
        steps=output_steps, name="vqbet_postprocessor"
    )
