#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su,
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

from lerobot.constants import POSTPROCESSOR_DEFAULT_NAME, PREPROCESSOR_DEFAULT_NAME
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameProcessorStep,
    UnnormalizerProcessorStep,
)


def make_tdmpc_pre_post_processors(
    config: TDMPCConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    input_steps = [
        RenameProcessorStep(rename_map={}),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]
    output_steps = [
        DeviceProcessorStep(device="cpu"),
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    return PolicyProcessorPipeline(
        steps=input_steps, name=PREPROCESSOR_DEFAULT_NAME
    ), PolicyProcessorPipeline(steps=output_steps, name=POSTPROCESSOR_DEFAULT_NAME)
