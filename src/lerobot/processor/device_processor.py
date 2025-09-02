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
from lerobot.utils.utils import get_safe_torch_device

from .core import EnvTransition, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("device_processor")
@dataclass
class DeviceProcessor(ProcessorStep):
    """Processes transitions by moving tensors to the specified device and optionally converting float dtypes.

    This processor ensures that all tensors in the transition are moved to the
    specified device (CPU or GPU) before they are returned. It can also convert
    floating-point tensors to a specified dtype while preserving non-float types
    (int, long, bool, etc.).
    """

    device: str = "cpu"
    float_dtype: str | None = None

    DTYPE_MAPPING = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "half": torch.float16,
        "float": torch.float32,
        "double": torch.float64,
    }

    def __post_init__(self):
        self._device: torch.device = get_safe_torch_device(self.device)
        self.device = self._device.type  # cuda might have changed to cuda:1
        self.non_blocking = "cuda" in str(self.device)

        # Validate and convert float_dtype string to torch dtype
        if self.float_dtype is not None:
            if self.float_dtype not in self.DTYPE_MAPPING:
                raise ValueError(
                    f"Invalid float_dtype '{self.float_dtype}'. Available options: {list(self.DTYPE_MAPPING.keys())}"
                )

            self._target_float_dtype = self.DTYPE_MAPPING[self.float_dtype]
        else:
            self._target_float_dtype = None

    def _process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Process a tensor by moving to device and optionally converting float dtype.

        If the tensor is already on a GPU and we're configured for a GPU, it preserves
        that GPU placement (useful for multi-GPU training with Accelerate).
        Otherwise, it moves to the configured device.
        """
        # Determine target device
        if tensor.is_cuda and self._device.type == "cuda":
            # Both tensor and target are on GPU - preserve tensor's GPU placement
            # This handles multi-GPU scenarios where Accelerate has already placed
            # tensors on the correct GPU for each process
            target_device = tensor.device
        else:
            # Either tensor is on CPU, or we're configured for CPU
            # In both cases, use the configured device
            target_device = self._device

        # Only move if necessary
        if tensor.device != target_device:
            tensor = tensor.to(target_device, non_blocking=self.non_blocking)

        # Convert float dtype if specified and tensor is floating point
        if self._target_float_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=self._target_float_dtype)

        return tensor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        simple_tensor_keys = [
            TransitionKey.ACTION,
            TransitionKey.REWARD,
            TransitionKey.DONE,
            TransitionKey.TRUNCATED,
        ]

        dict_tensor_keys = [
            TransitionKey.OBSERVATION,
            TransitionKey.COMPLEMENTARY_DATA,
        ]

        # Process simple tensors
        for key in simple_tensor_keys:
            value = transition.get(key)
            if isinstance(value, torch.Tensor):
                new_transition[key] = self._process_tensor(value)

        # Process dictionary-like tensors
        for key in dict_tensor_keys:
            data_dict = transition.get(key)
            if data_dict is not None:
                new_data_dict = {
                    k: self._process_tensor(v) if isinstance(v, torch.Tensor) else v
                    for k, v in data_dict.items()
                }
                new_transition[key] = new_data_dict

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Return configuration for serialization."""
        return {"device": self.device, "float_dtype": self.float_dtype}

    def transform_features(self, features: dict[str, PolicyFeature]) -> dict[str, PolicyFeature]:
        return features
