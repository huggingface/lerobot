#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_dm05 import DM05Config


@dataclass
@ProcessorStepRegistry.register(name="dm05_task_processor")
class DM05TaskProcessor(ComplementaryDataProcessorStep):
    """Normalize the task prompt field expected by DM05 tokenization."""

    default_task: str = "Execute the robot action."

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        if (task := complementary_data.get("task")) is None:
            return {**complementary_data, "task": self.default_task}

        if isinstance(task, str):
            return {**complementary_data, "task": task.strip() or self.default_task}
        if isinstance(task, list):
            return {
                **complementary_data,
                "task": [str(item).strip() or self.default_task for item in task],
            }
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {"default_task": self.default_task}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_dm05_pre_post_processors(
    config: DM05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build LeRobot processors for DM05.

    DM05 numerical state/action normalization is handled by the DM05 batch
    converter and ``norm_stats.json``. The LeRobot processor pipeline only
    adapts task, batch, and device placement.
    """

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[
                RenameObservationsProcessorStep(rename_map={}),
                AddBatchDimensionProcessorStep(),
                DM05TaskProcessor(),
                DeviceProcessorStep(device=config.device),
            ],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[DeviceProcessorStep(device="cpu")],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
