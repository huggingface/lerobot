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
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor")
class ToBatchProcessor:
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

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        self._process_observation(transition)
        self._process_action(transition)
        self._process_complementary_data(transition)
        return transition

    def _process_observation(self, transition: EnvTransition) -> None:
        """Process observation component in-place, adding batch dimensions where needed."""
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            return

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

    def _process_action(self, transition: EnvTransition) -> None:
        """Process action component in-place, adding batch dimension if needed."""
        action = transition.get(TransitionKey.ACTION)
        if action is not None and isinstance(action, Tensor) and action.dim() == 1:
            transition[TransitionKey.ACTION] = action.unsqueeze(0)

    def _process_complementary_data(self, transition: EnvTransition) -> None:
        """Process complementary data in-place, handling task field batching."""
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            return

        # Process task field - wrap string in list to add batch dimension
        if "task" in complementary_data:
            task_value = complementary_data["task"]
            if isinstance(task_value, str):
                complementary_data["task"] = [task_value]

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary (empty for this processor)."""
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state dictionary (no-op for this processor)."""
        pass

    def reset(self) -> None:
        """Reset processor state (no-op for this processor)."""
        pass
