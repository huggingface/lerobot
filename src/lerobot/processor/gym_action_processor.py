#! /usr/bin/env python

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

from dataclasses import dataclass

import numpy as np
import torch

from lerobot.processor.converters import to_tensor
from lerobot.processor.pipeline import ActionProcessor, ProcessorStepRegistry


@ProcessorStepRegistry.register("torch2numpy_action_processor")
@dataclass
class Torch2NumpyActionProcessor(ActionProcessor):
    """Convert PyTorch tensor actions to NumPy arrays."""

    squeeze_batch_dim: bool = True

    def action(self, action: torch.Tensor) -> np.ndarray:
        if not isinstance(action, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor or None, got {type(action).__name__}. "
                "Use appropriate processor for non-tensor actions."
            )

        numpy_action = action.detach().cpu().numpy()

        # Remove batch dimensions but preserve action dimensions
        # Only squeeze if there's a batch dimension (first dim == 1)
        if (
            self.squeeze_batch_dim
            and numpy_action.shape
            and len(numpy_action.shape) > 1
            and numpy_action.shape[0] == 1
        ):
            numpy_action = numpy_action.squeeze(0)

        return numpy_action


@ProcessorStepRegistry.register("numpy2torch_action_processor")
@dataclass
class Numpy2TorchActionProcessor(ActionProcessor):
    """Convert NumPy array action to PyTorch tensor."""

    def action(self, action: np.ndarray) -> torch.Tensor:
        if not isinstance(action, np.ndarray):
            raise TypeError(
                f"Expected np.ndarray or None, got {type(action).__name__}. "
                "Use appropriate processor for non-tensor actions."
            )
        torch_action = to_tensor(action, dtype=None)  # Preserve original dtype
        return torch_action
