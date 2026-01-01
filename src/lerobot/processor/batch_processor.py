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

"""
This script defines processor steps for adding a batch dimension to various components of an environment transition.

These steps are designed to process actions, observations, and complementary data, making them suitable for batch processing by adding a leading dimension. This is a common requirement before feeding data into a neural network model.
"""

from dataclasses import dataclass, field

from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

from .core import EnvTransition, PolicyAction
from .pipeline import (
    ComplementaryDataProcessorStep,
    ObservationProcessorStep,
    PolicyActionProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_action")
class AddBatchDimensionActionStep(PolicyActionProcessorStep):
    """
    Processor step to add a batch dimension to a 1D tensor action.

    This is useful for creating a batch of size 1 from a single action sample.
    """

    def action(self, action: PolicyAction) -> PolicyAction:
        """
        Adds a batch dimension to the action if it's a 1D tensor.

        Args:
            action: The action tensor.

        Returns:
            The action tensor with an added batch dimension.
        """
        if action.dim() != 1:
            return action
        return action.unsqueeze(0)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Adding a batch dimension does not alter the feature definition.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_observation")
class AddBatchDimensionObservationStep(ObservationProcessorStep):
    """
    Processor step to add a batch dimension to observations.

    It handles different types of observations:
    - State vectors (1D tensors).
    - Single images (3D tensors).
    - Dictionaries of multiple images (3D tensors).
    """

    def observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Adds a batch dimension to tensor-based observations in the observation dictionary.

        Args:
            observation: The observation dictionary.

        Returns:
            The observation dictionary with batch dimensions added to tensors.
        """
        # Process state observations - add batch dim if 1D
        for state_key in [OBS_STATE, OBS_ENV_STATE]:
            if state_key in observation:
                state_value = observation[state_key]
                if isinstance(state_value, Tensor) and state_value.dim() == 1:
                    observation[state_key] = state_value.unsqueeze(0)

        # Process single image observation - add batch dim if 3D
        if OBS_IMAGE in observation:
            image_value = observation[OBS_IMAGE]
            if isinstance(image_value, Tensor) and image_value.dim() == 3:
                observation[OBS_IMAGE] = image_value.unsqueeze(0)

        # Process multiple image observations - add batch dim if 3D
        for key, value in observation.items():
            if key.startswith(f"{OBS_IMAGES}.") and isinstance(value, Tensor) and value.dim() == 3:
                observation[key] = value.unsqueeze(0)
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Adding a batch dimension does not alter the feature definition.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_complementary_data")
class AddBatchDimensionComplementaryDataStep(ComplementaryDataProcessorStep):
    """
    Processor step to add a batch dimension to complementary data fields.

    Handles specific keys like 'task', 'index', and 'task_index' to make them batched.
    - 'task' (str) is wrapped in a list.
    - 'index' and 'task_index' (0D tensors) get a batch dimension.
    """

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Adds a batch dimension to specific fields in the complementary data dictionary.

        Args:
            complementary_data: The complementary data dictionary.

        Returns:
            The complementary data dictionary with batch dimensions added.
        """
        # Process task field - wrap string in list to add batch dimension
        if "task" in complementary_data:
            task_value = complementary_data["task"]
            if isinstance(task_value, str):
                complementary_data["task"] = [task_value]

        # Process index field - add batch dim if 0D
        if "index" in complementary_data:
            index_value = complementary_data["index"]
            if isinstance(index_value, Tensor) and index_value.dim() == 0:
                complementary_data["index"] = index_value.unsqueeze(0)

        # Process task_index field - add batch dim if 0D
        if "task_index" in complementary_data:
            task_index_value = complementary_data["task_index"]
            if isinstance(task_index_value, Tensor) and task_index_value.dim() == 0:
                complementary_data["task_index"] = task_index_value.unsqueeze(0)
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Adding a batch dimension does not alter the feature definition.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor")
class AddBatchDimensionProcessorStep(ProcessorStep):
    """
    A composite processor step that adds a batch dimension to the entire environment transition.

    This step combines individual processors for actions, observations, and complementary data
    to create a batched transition (batch size 1) from a single-instance transition.

    Attributes:
        to_batch_action_processor: Processor for the action component.
        to_batch_observation_processor: Processor for the observation component.
        to_batch_complementary_data_processor: Processor for the complementary data component.
    """

    to_batch_action_processor: AddBatchDimensionActionStep = field(
        default_factory=AddBatchDimensionActionStep
    )
    to_batch_observation_processor: AddBatchDimensionObservationStep = field(
        default_factory=AddBatchDimensionObservationStep
    )
    to_batch_complementary_data_processor: AddBatchDimensionComplementaryDataStep = field(
        default_factory=AddBatchDimensionComplementaryDataStep
    )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Applies the batching process to all relevant parts of an environment transition.

        Args:
            transition: The environment transition to process.

        Returns:
            The environment transition with a batch dimension added.
        """
        if transition[TransitionKey.ACTION] is not None:
            transition = self.to_batch_action_processor(transition)
        if transition[TransitionKey.OBSERVATION] is not None:
            transition = self.to_batch_observation_processor(transition)
        if transition[TransitionKey.COMPLEMENTARY_DATA] is not None:
            transition = self.to_batch_complementary_data_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Adding a batch dimension does not alter the feature definition.

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        # NOTE: We ignore the batch dimension when transforming features
        return features
