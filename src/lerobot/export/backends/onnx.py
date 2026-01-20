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
"""ONNX Runtime backend for model execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ONNXBackend:
    """ONNX Runtime backend for model inference.

    This backend wraps onnxruntime.InferenceSession to provide a simple
    interface for running ONNX models.
    """

    def __init__(self, model_path: Path | str, device: str = "cpu"):
        """Initialize the ONNX Runtime backend.

        Args:
            model_path: Path to the ONNX model file.
            device: Device for inference ("cpu", "cuda", "cuda:0", etc.).

        Raises:
            ImportError: If onnxruntime is not installed.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for ONNX backend. "
                "Install with: pip install onnxruntime or pip install onnxruntime-gpu"
            ) from e

        self._model_path = Path(model_path)
        self._device = device

        # Configure execution providers based on device
        providers = self._get_providers(device)

        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self._model_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Cache input/output metadata
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]

        # Build input metadata for validation
        self._input_metadata = {
            i.name: {"shape": i.shape, "dtype": i.type} for i in self._session.get_inputs()
        }

    @property
    def input_names(self) -> list[str]:
        """Return the list of input tensor names."""
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Return the list of output tensor names."""
        return self._output_names

    def run(self, inputs: dict[str, NDArray[np.floating]]) -> dict[str, NDArray[np.floating]]:
        """Execute one forward pass.

        Args:
            inputs: Dictionary mapping input names to numpy arrays.

        Returns:
            Dictionary mapping output names to numpy arrays.
        """
        # Filter inputs to only include those expected by the model
        ort_inputs = {k: v for k, v in inputs.items() if k in self._input_names}

        # Ensure all required inputs are present
        missing = set(self._input_names) - set(ort_inputs.keys())
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")

        # Run inference
        outputs = self._session.run(self._output_names, ort_inputs)

        return dict(zip(self._output_names, outputs, strict=True))

    def _get_providers(self, device: str) -> list[str | tuple[str, dict]]:
        """Get execution providers based on device.

        Args:
            device: Device specification ("cpu", "cuda", "cuda:0", etc.).

        Returns:
            List of execution providers in priority order.
        """
        if device.startswith("cuda"):
            # Parse device index if specified
            device_id = 0
            if ":" in device:
                try:
                    device_id = int(device.split(":")[1])
                except (ValueError, IndexError):
                    device_id = 0

            return [
                ("CUDAExecutionProvider", {"device_id": device_id}),
                "CPUExecutionProvider",
            ]
        return ["CPUExecutionProvider"]

    def get_input_shape(self, name: str) -> list | None:
        """Get the shape of an input tensor.

        Args:
            name: Name of the input tensor.

        Returns:
            Shape as a list, or None if not found.
        """
        if name in self._input_metadata:
            return self._input_metadata[name]["shape"]
        return None

    def __repr__(self) -> str:
        return f"ONNXBackend(model={self._model_path.name}, device={self._device})"
