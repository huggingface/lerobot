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
This script defines a processor step for moving environment transition data to a specific torch device and casting
its floating-point precision.
"""

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.utils.utils import get_safe_torch_device

from .core import EnvTransition, PolicyAction, TransitionKey
from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("device_processor")
@dataclass
class DeviceProcessorStep(ProcessorStep):
    """
    Processor step to move all tensors within an `EnvTransition` to a specified device and optionally cast their
    floating-point data type.

    This is crucial for preparing data for model training or inference on hardware like GPUs.

    Attributes:
        device: The target device for tensors (e.g., "cpu", "cuda", "cuda:0").
        float_dtype: The target floating-point dtype as a string (e.g., "float32", "float16", "bfloat16").
                     If None, the dtype is not changed.
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
        """
        Initializes the processor by converting string configurations to torch objects.

        This method sets up the `torch.device`, determines if transfers can be non-blocking, and validates the
        `float_dtype` string, converting it to a `torch.dtype` object.
        """
        self.tensor_device: torch.device = get_safe_torch_device(self.device)
        # Update device string in case a specific GPU was selected (e.g., "cuda" -> "cuda:0")
        self.device = self.tensor_device.type
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
        """
        Moves a single tensor to the target device and casts its dtype.

        Handles multi-GPU scenarios by not moving a tensor if it's already on a different CUDA device than
        the target, which is useful when using frameworks like Accelerate.

        Args:
            tensor: The input torch.Tensor.

        Returns:
            The processed tensor on the correct device and with the correct dtype.
        """
        # Determine target device
        if tensor.is_cuda and self.tensor_device.type == "cuda":
            # Both tensor and target are on GPU - preserve tensor's GPU placement.
            # This handles multi-GPU scenarios where Accelerate has already placed
            # tensors on the correct GPU for each process.
            target_device = tensor.device
        else:
            # Either tensor is on CPU, or we're configured for CPU.
            # In both cases, use the configured device.
            target_device = self.tensor_device

        # MPS workaround: Convert float64 to float32 since MPS doesn't support float64
        if target_device.type == "mps" and tensor.dtype == torch.float64:
            tensor = tensor.to(dtype=torch.float32)

        # Only move if necessary
        if tensor.device != target_device:
            tensor = tensor.to(target_device, non_blocking=self.non_blocking)

        # Convert float dtype if specified and tensor is floating point
        if self._target_float_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=self._target_float_dtype)

        return tensor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Applies device and dtype conversion to all tensors in an environment transition.

        It iterates through the transition, finds all `torch.Tensor` objects (including those nested in
        dictionaries like `observation`), and processes them.

        Args:
            transition: The input `EnvTransition` object.

        Returns:
            A new `EnvTransition` object with all tensors moved to the target device and dtype.
        """
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is not None and not isinstance(action, PolicyAction):
            raise ValueError(f"If action is not None should be a PolicyAction type got {type(action)}")

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

        # Process simple, top-level tensors
        for key in simple_tensor_keys:
            value = transition.get(key)
            if isinstance(value, torch.Tensor):
                new_transition[key] = self._process_tensor(value)

        # Process tensors nested within dictionaries
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
        """
        Returns the serializable configuration of the processor.

        Returns:
            A dictionary containing the device and float_dtype settings.
        """
        return {"device": self.device, "float_dtype": self.float_dtype}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Returns the input features unchanged.

        Device and dtype transformations do not alter the fundamental definition of the features (e.g., shape).

        Args:
            features: A dictionary of policy features.

        Returns:
            The original dictionary of policy features.
        """
        return features
