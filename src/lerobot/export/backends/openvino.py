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
"""OpenVINO backend for model execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class OpenVINOBackend:
    """OpenVINO backend for model inference.

    This backend wraps OpenVINO's Core and CompiledModel to provide a simple
    interface for running models optimized for Intel hardware.
    """

    def __init__(self, model_path: Path | str, device: str = "cpu"):
        """Initialize the OpenVINO backend.

        Args:
            model_path: Path to the ONNX or OpenVINO IR model file.
            device: Device for inference ("cpu", "gpu", "npu", etc.).

        Raises:
            ImportError: If openvino is not installed.
        """
        try:
            import openvino as ov
        except ImportError as e:
            raise ImportError(
                "openvino is required for OpenVINO backend. Install with: pip install openvino"
            ) from e

        self._model_path = Path(model_path)
        self._device = self._normalize_device(device)

        # Initialize OpenVINO Core
        self._core = ov.Core()

        # Read the model (before compilation to get original names)
        model = self._core.read_model(str(self._model_path))

        # Cache original input/output names before compilation
        # (OpenVINO can rename inputs during optimization, e.g., image_0 -> /Cast_output_0)
        original_input_names = [inp.get_any_name() for inp in model.inputs]
        original_output_names = [out.get_any_name() for out in model.outputs]

        # Compile the model
        self._compiled_model = self._core.compile_model(model, self._device)

        # Create inference request for reuse
        self._infer_request = self._compiled_model.create_infer_request()

        # Get compiled model names (may differ from original due to optimization)
        compiled_input_names = [inp.get_any_name() for inp in self._compiled_model.inputs]
        compiled_output_names = [out.get_any_name() for out in self._compiled_model.outputs]

        # Use original names externally, but map to compiled names internally
        self._input_names = original_input_names
        self._output_names = original_output_names

        # Build mapping from original to compiled names (for inputs that got renamed)
        self._input_name_mapping: dict[str, str] = {}
        for orig, comp in zip(original_input_names, compiled_input_names, strict=True):
            if orig != comp:
                self._input_name_mapping[orig] = comp

        self._output_name_mapping: dict[str, str] = {}
        for orig, comp in zip(original_output_names, compiled_output_names, strict=True):
            if orig != comp:
                self._output_name_mapping[orig] = comp

        # Build input metadata using original names
        self._input_metadata = {}
        for inp, orig_name in zip(self._compiled_model.inputs, original_input_names, strict=True):
            shape = inp.partial_shape
            if shape.is_static:
                shape_list = [d.get_length() for d in shape]
            else:
                shape_list = [d.get_length() if d.is_static else -1 for d in shape]
            self._input_metadata[orig_name] = {
                "shape": shape_list,
                "dtype": str(inp.element_type),
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
        ov_inputs = {k: v for k, v in inputs.items() if k in self._input_names}

        missing = set(self._input_names) - set(ov_inputs.keys())
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")

        # Map original input names to compiled names if they were renamed
        mapped_inputs = {}
        for name, value in ov_inputs.items():
            compiled_name = self._input_name_mapping.get(name, name)
            mapped_inputs[compiled_name] = value

        self._infer_request.infer(mapped_inputs)

        # Map compiled output names back to original names
        outputs = {}
        for orig_name in self._output_names:
            compiled_name = self._output_name_mapping.get(orig_name, orig_name)
            tensor = self._infer_request.get_tensor(compiled_name)
            outputs[orig_name] = tensor.data.copy()

        return outputs

    def _normalize_device(self, device: str) -> str:
        """Normalize device string to OpenVINO format.

        Args:
            device: Device specification ("cpu", "cuda", "cuda:0", "gpu", etc.).

        Returns:
            OpenVINO-compatible device string.
        """
        device = device.lower()

        # Map common device names to OpenVINO equivalents
        if device == "cpu":
            return "CPU"
        elif device.startswith("cuda") or device == "gpu":
            return "GPU"
        elif device == "npu":
            return "NPU"
        elif device == "auto":
            return "AUTO"

        # Return uppercase version for other devices
        return device.upper()

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
        return f"OpenVINOBackend(model={self._model_path.name}, device={self._device})"
