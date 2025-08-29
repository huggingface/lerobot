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
    RobotProcessor,
    ToBatchProcessor,
    TokenizerProcessor,
    UnnormalizerProcessor,
)
from lerobot.processor.pipeline import (
    ComplementaryDataProcessor,
    ProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.processor.rename_processor import RenameProcessor


@ProcessorStepRegistry.register(name="pi0_new_line_processor")
class Pi0NewLineProcessor(ComplementaryDataProcessor):
    """Add a new line to the end of the task if it doesn't have one.
    This is required for the PaliGemma tokenizer.
    """

    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: add newline if not present
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: add newline to each if not present
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        return new_complementary_data


def make_pi0_pre_post_processors(
    config: PI0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    **kwargs: Unpack[ProcessorFactoryKwargs],
) -> tuple[RobotProcessor, RobotProcessor]:
    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameProcessor(rename_map={}),  # To mimic the same processor as pretrained one
        NormalizerProcessor(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        ToBatchProcessor(),
        Pi0NewLineProcessor(),  # Add newlines before tokenization for PaliGemma
        TokenizerProcessor(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessor(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
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
