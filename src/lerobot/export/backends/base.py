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
"""Backend protocol and factory for model execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Backend(Protocol):
    """Minimal interface for model execution.

    Backends are intentionally minimal—they execute a single forward pass.
    The runtime handles the higher-level logic like normalization and iterative loops.
    """

    @property
    def input_names(self) -> list[str]:
        """Return the list of input tensor names."""
        ...

    @property
    def output_names(self) -> list[str]:
        """Return the list of output tensor names."""
        ...

    def run(self, inputs: dict[str, NDArray[np.floating]]) -> dict[str, NDArray[np.floating]]:
        """Execute one forward pass.

        Args:
            inputs: Dictionary mapping input names to numpy arrays.

        Returns:
            Dictionary mapping output names to numpy arrays.
        """
        ...


def get_backend(backend_name: str, model_path: Path, device: str = "cpu") -> Backend:
    """Factory function to get the appropriate backend.

    Args:
        backend_name: Name of the backend ("onnx" or "openvino").
        model_path: Path to the model file.
        device: Device for inference ("cpu", "cuda", "cuda:0").

    Returns:
        Backend instance ready for inference.

    Raises:
        ValueError: If the backend is not supported.
    """
    if backend_name == "onnx":
        from .onnx import ONNXBackend

        return ONNXBackend(model_path, device)
    elif backend_name == "openvino":
        from .openvino import OpenVINOBackend

        return OpenVINOBackend(model_path, device)
    else:
        raise ValueError(f"Unsupported backend: {backend_name}. Supported: onnx, openvino")
