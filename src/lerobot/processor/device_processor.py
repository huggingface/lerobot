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
from typing import Any

import torch

from lerobot.configs.types import PolicyFeature
from lerobot.processor.pipeline import EnvTransition, ProcessorStepRegistry, TransitionKey
from lerobot.utils.utils import get_safe_torch_device


@ProcessorStepRegistry.register("device_processor")
@dataclass
class DeviceProcessor:
    """Processes transitions by moving tensors to the specified device and optionally converting float dtypes.

    This processor ensures that all tensors in the transition are moved to the
    specified device (CPU or GPU) before they are returned. It can also convert
    floating-point tensors to a specified dtype while preserving non-float types
    (int, long, bool, etc.).
    """

    device: torch.device = "cpu"
    float_dtype: str | None = None

    def __post_init__(self):
        self.device = get_safe_torch_device(self.device)
        self.non_blocking = "cuda" in str(self.device)

        # Validate and convert float_dtype string to torch dtype
        if self.float_dtype is not None:
            dtype_mapping = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
                "bfloat16": torch.bfloat16,
                "half": torch.float16,
                "float": torch.float32,
                "double": torch.float64,
            }

            if self.float_dtype not in dtype_mapping:
                available_dtypes = list(dtype_mapping.keys())
                raise ValueError(
                    f"Invalid float_dtype '{self.float_dtype}'. Available options: {available_dtypes}"
                )

            self._target_float_dtype = dtype_mapping[self.float_dtype]
        else:
            self._target_float_dtype = None

    def _process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process a tensor by moving to device and optionally converting float dtype."""
        # Move to device first
        tensor = tensor.to(self.device, non_blocking=self.non_blocking)

        # Convert float dtype if specified and tensor is floating point
        if self._target_float_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=self._target_float_dtype)

        return tensor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Create a copy of the transition
        new_transition = transition.copy()

        # Process observation tensors
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_observation = {
                k: self._process_tensor(v) if isinstance(v, torch.Tensor) else v
                for k, v in observation.items()
            }
            new_transition[TransitionKey.OBSERVATION] = new_observation

        # Process action tensor
        action = transition.get(TransitionKey.ACTION)
        if action is not None and isinstance(action, torch.Tensor):
            new_transition[TransitionKey.ACTION] = self._process_tensor(action)

        # Process reward tensor
        reward = transition.get(TransitionKey.REWARD)
        if reward is not None and isinstance(reward, torch.Tensor):
            new_transition[TransitionKey.REWARD] = self._process_tensor(reward)

        # Process done tensor
        done = transition.get(TransitionKey.DONE)
        if done is not None and isinstance(done, torch.Tensor):
            new_transition[TransitionKey.DONE] = self._process_tensor(done)

        # Process truncated tensor
        truncated = transition.get(TransitionKey.TRUNCATED)
        if truncated is not None and isinstance(truncated, torch.Tensor):
            new_transition[TransitionKey.TRUNCATED] = self._process_tensor(truncated)

        # Process complementary data tensors
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is not None:
            new_complementary_data = {}

            # Process all items in complementary_data
            for key, value in complementary_data.items():
                if isinstance(value, torch.Tensor):
                    new_complementary_data[key] = self._process_tensor(value)
                else:
                    new_complementary_data[key] = value

            new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {"device": self.device, "float_dtype": self.float_dtype}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return state dictionary (empty for this processor)."""
        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load state dictionary (no-op for this processor)."""
        pass

    def reset(self) -> None:
        """Reset processor state (no-op for this processor)."""
        pass

    def feature_contract(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features
