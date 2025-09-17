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

from dataclasses import dataclass

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

from .converters import to_tensor
from .core import EnvAction, EnvTransition, PolicyAction
from .pipeline import ActionProcessorStep, ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("torch2numpy_action_processor")
@dataclass
class Torch2NumpyActionProcessorStep(ActionProcessorStep):
    """
    Converts a PyTorch tensor action to a NumPy array.

    This step is useful when the output of a policy (typically a torch.Tensor)
    needs to be passed to an environment or component that expects a NumPy array.

    Attributes:
        squeeze_batch_dim: If True, removes the first dimension of the array
                           if it is of size 1. This is useful for converting a
                           batched action of size (1, D) to a single action of size (D,).
    """

    squeeze_batch_dim: bool = True

    def action(self, action: PolicyAction) -> EnvAction:
        if not isinstance(action, PolicyAction):
            raise TypeError(
                f"Expected PolicyAction or None, got {type(action).__name__}. "
                "Use appropriate processor for non-tensor actions."
            )

        numpy_action = action.detach().cpu().numpy()

        # Remove batch dimensions but preserve action dimensions.
        # Only squeeze if there's a batch dimension (first dim == 1).
        if (
            self.squeeze_batch_dim
            and numpy_action.shape
            and len(numpy_action.shape) > 1
            and numpy_action.shape[0] == 1
        ):
            numpy_action = numpy_action.squeeze(0)

        return numpy_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("numpy2torch_action_processor")
@dataclass
class Numpy2TorchActionProcessorStep(ProcessorStep):
    """Converts a NumPy array action to a PyTorch tensor when action is present."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Converts numpy action to torch tensor if action exists, otherwise passes through."""
        from .core import TransitionKey

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            if not isinstance(action, EnvAction):
                raise TypeError(
                    f"Expected np.ndarray or None, got {type(action).__name__}. "
                    "Use appropriate processor for non-tensor actions."
                )
            torch_action = to_tensor(action, dtype=None)  # Preserve original dtype
            new_transition[TransitionKey.ACTION] = torch_action

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
