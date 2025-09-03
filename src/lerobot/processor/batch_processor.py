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
from dataclasses import dataclass, field

from torch import Tensor

from lerobot.configs.types import PolicyFeature
from lerobot.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

from .core import EnvTransition
from .pipeline import (
    ActionProcessor,
    ComplementaryDataProcessor,
    ObservationProcessor,
    ProcessorStep,
    ProcessorStepRegistry,
)


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_action")
class ToBatchProcessorAction(ActionProcessor):
    """Process action component in-place, adding batch dimension if needed."""

    def action(self, action):
        if not isinstance(action, Tensor) or action.dim() != 1:
            return action

        return action.unsqueeze(0)

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_observation")
class ToBatchProcessorObservation(ObservationProcessor):
    """Process observation component in-place, adding batch dimensions where needed."""

    def observation(self, observation):
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

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_complementary_data")
class ToBatchProcessorComplementaryData(ComplementaryDataProcessor):
    """Process complementary data in-place, handling task field batching."""

    def complementary_data(self, complementary_data):
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

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor")
class ToBatchProcessor(ProcessorStep):
    """Processor that adds batch dimensions to observations and actions when needed.

    This processor ensures that observations and actions have proper batch dimensions for model processing:

    - For state observations (observation.state, observation.environment_state):
      Adds batch dimension (unsqueeze at dim=0) if tensor is 1-dimensional

    - For image observations (observation.image, observation.images.*):
      Adds batch dimension (unsqueeze at dim=0) if tensor is 3-dimensional (H, W, C)

    - For actions:
      Adds batch dimension (unsqueeze at dim=0) if tensor is 1-dimensional

    - For task field in complementary data:
      Wraps string task in a list to add batch dimension
      (task must be a string or list of strings)

    This is useful when processing single transitions that need to be batched for
    model inference or when converting from unbatched environment outputs to
    batched model inputs.

    The processor only modifies tensors that need batching and leaves already
    batched tensors unchanged.

    Example:
        ```python
        # State: (7,) -> (1, 7)
        # Image: (224, 224, 3) -> (1, 224, 224, 3)
        # Action: (4,) -> (1, 4)
        # Task: "pick_cube" -> ["pick_cube"]
        # Already batched: (1, 7) -> (1, 7) [unchanged]
        ```
    """

    to_batch_action_processor: ToBatchProcessorAction = field(default_factory=ToBatchProcessorAction)
    to_batch_observation_processor: ToBatchProcessorObservation = field(
        default_factory=ToBatchProcessorObservation
    )
    to_batch_complementary_data_processor: ToBatchProcessorComplementaryData = field(
        default_factory=ToBatchProcessorComplementaryData
    )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = self.to_batch_action_processor(transition)
        transition = self.to_batch_observation_processor(transition)
        transition = self.to_batch_complementary_data_processor(transition)
        return transition

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        # NOTE: We ignore the batch dimension when transforming features
        return features
