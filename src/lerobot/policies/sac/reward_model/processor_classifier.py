# !/usr/bin/env python

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
import torch

from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.processor import (
    IdentityProcessor,
    NormalizerProcessor,
    RobotProcessor,
)


def make_classifier_processor(
    config: RewardClassifierConfig, dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
) -> tuple[RobotProcessor, RobotProcessor]:
    input_steps = [
        NormalizerProcessor(
            features=config.input_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        NormalizerProcessor(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    output_steps = [IdentityProcessor()]
    return RobotProcessor(steps=input_steps, name="classifier_preprocessor"), RobotProcessor(
        steps=output_steps, name="classifier_postprocessor"
    )
