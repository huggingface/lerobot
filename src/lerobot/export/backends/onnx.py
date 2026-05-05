#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import register_backend, resolve_artifact_paths

if TYPE_CHECKING:
    from ..interfaces import _RuntimeSession
    from ..runners.base import ExportModule

__all__ = ["ONNXBackend"]


class _ONNXRuntimeSession:
    """Live ONNX inference session wrapping one or more ``onnxruntime`` sessions.

    Holds a dict of named :class:`onnxruntime.InferenceSession` objects (one
    per exported stage) and exposes a unified :meth:`run` interface that
    handles input filtering, dtype coercion, and output name mapping.
    """

    _ONNX_TYPE_TO_NUMPY = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }

    def __init__(self, sessions: dict[str, Any]):
        """Initialise from a dict of named ``onnxruntime.InferenceSession`` objects.

        Args:
            sessions: Dict mapping stage name (e.g. ``"model"``) to an
                ``onnxruntime.InferenceSession`` instance.
        """
        self._sessions = sessions
        self._input_names = {
            name: [item.name for item in session.get_inputs()] for name, session in sessions.items()
        }
        self._output_names = {
            name: [item.name for item in session.get_outputs()] for name, session in sessions.items()
        }
        self._input_metadata = {
            name: {item.name: item.type for item in session.get_inputs()}
            for name, session in sessions.items()
        }

    def run(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute a named ONNX stage.

        Filters inputs to only those expected by the session, coerces dtypes
        to match the model's declared input types, and returns a dict of
        output arrays keyed by output name.

        Args:
            name: Stage identifier (e.g. ``"model"``, ``"encoder"``).
            inputs: Dict of input arrays; extra keys are silently ignored.

        Returns:
            Dict mapping output tensor names to numpy arrays.

        Raises:
            ValueError: If required inputs are missing.
        """
        session = self._sessions[name]
        input_names = self._input_names[name]
        ort_inputs = {k: v for k, v in inputs.items() if k in input_names}
        missing = set(input_names) - set(ort_inputs)
        if missing:
            raise ValueError(f"Missing required inputs for {name!r}: {sorted(missing)}")

        metadata = self._input_metadata[name]
        for input_name, value in list(ort_inputs.items()):
            np_dtype = self._ONNX_TYPE_TO_NUMPY.get(metadata[input_name])
            if np_dtype is not None and value.dtype != np_dtype:
                ort_inputs[input_name] = value.astype(np_dtype)

        outputs = session.run(self._output_names[name], ort_inputs)
        return dict(zip(self._output_names[name], outputs, strict=True))


@register_backend
class ONNXBackend:
    """ONNX serialisation and runtime backend.

    Uses ``torch.onnx.export`` to trace and serialise model stages to
    ``.onnx`` files, and ``onnxruntime`` to load and run them at inference
    time.
    """

    name = "onnx"
    extension = ".onnx"
    runtime_only = False

    def serialize(
        self,
        modules: list[ExportModule],
        artifacts_dir: Path,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Trace and serialise export modules to ``.onnx`` files.

        Applies any ``onnx_fixups`` declared in each module's ``hints`` dict
        after export.

        Args:
            modules: Ordered list of :class:`ExportModule` specs to serialise.
            artifacts_dir: Directory where ``.onnx`` files are written.
            **kwargs: Must include ``opset_version`` (int).

        Returns:
            Dict mapping module name to the ``.onnx`` filename.

        Raises:
            ValueError: If an unknown fixup name is encountered.
        """
        import torch

        opset_version = cast(int, kwargs["opset_version"])
        artifacts: dict[str, str] = {}
        for module in modules:
            output_path = artifacts_dir / f"{module.name}{self.extension}"
            torch.onnx.export(
                module.wrapper,
                module.example_inputs,
                str(output_path),
                input_names=module.input_names,
                output_names=module.output_names,
                dynamic_axes=module.dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                dynamo=False,
            )
            for fixup_name in module.hints.get("onnx_fixups", []):
                fixup = _ONNX_FIXUPS.get(fixup_name)
                if fixup is None:
                    raise ValueError(f"Unknown onnx fixup {fixup_name!r}. Available: {sorted(_ONNX_FIXUPS)}")
                fixup(output_path)
            artifacts[module.name] = output_path.name
        return artifacts

    def open(
        self,
        artifacts_dir: Path,
        manifest: dict[str, Any],
        *,
        device: str = "cpu",
    ) -> _RuntimeSession:
        """Load ``.onnx`` artifacts and return a ready-to-use session.

        Args:
            artifacts_dir: Directory containing the ``.onnx`` files.
            manifest: Parsed manifest dict used to resolve artifact paths.
            device: Target device string (``"cpu"``, ``"cuda"``,
                ``"cuda:N"``).

        Returns:
            An internal :class:`_ONNXRuntimeSession` wrapping the loaded sessions.

        Raises:
            ImportError: If ``onnxruntime`` is not installed.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for ONNX backend. "
                "Install with: pip install onnxruntime or pip install onnxruntime-gpu"
            ) from e

        providers = _get_providers(device)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sessions = {
            name: ort.InferenceSession(str(path), sess_options=sess_options, providers=providers)
            for name, path in resolve_artifact_paths(artifacts_dir, manifest).items()
            if path.suffix == self.extension
        }
        return _ONNXRuntimeSession(sessions)


def _get_providers(device: str) -> list[str | tuple[str, dict[str, int]]]:
    if device.startswith("cuda"):
        device_id = 0
        if ":" in device:
            try:
                device_id = int(device.split(":", maxsplit=1)[1])
            except (ValueError, IndexError):
                device_id = 0
        return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _fix_onnx_scatter_gather_dtypes(onnx_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, shape_inference

    model = onnx.load(str(onnx_path))
    inferred = shape_inference.infer_shapes(model)
    type_map: dict[str, int] = {}
    for value_info in [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]:
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type:
            type_map[value_info.name] = tensor_type.elem_type

    nodes_to_insert: list[tuple[int, onnx.NodeProto]] = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "ScatterND" and len(node.input) >= 3:
            data_input, _, updates_input = node.input[:3]
            data_type = type_map.get(data_input)
            updates_type = type_map.get(updates_input)
            if data_type is not None and updates_type is not None and data_type != updates_type:
                cast_output = updates_input + f"_cast_to_{TensorProto.DataType.Name(data_type).lower()}"
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[updates_input],
                    outputs=[cast_output],
                    name=updates_input + f"/Cast_to_{TensorProto.DataType.Name(data_type).lower()}",
                    to=data_type,
                )
                nodes_to_insert.append((idx, cast_node))
                node.input[2] = cast_output
        if node.op_type == "Gather" and len(node.input) >= 2 and "position_embedding" in node.name:
            indices_input = node.input[1]
            indices_type = type_map.get(indices_input)
            if indices_type is not None and indices_type != TensorProto.INT64:
                cast_output = indices_input + "_cast_to_int64"
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[indices_input],
                    outputs=[cast_output],
                    name=indices_input + "/Cast_to_int64",
                    to=TensorProto.INT64,
                )
                nodes_to_insert.append((idx, cast_node))
                node.input[1] = cast_output
    for idx, cast_node in reversed(nodes_to_insert):
        model.graph.node.insert(idx, cast_node)
    if nodes_to_insert:
        onnx.save(model, str(onnx_path))


_ONNX_FIXUPS: dict[str, Callable[[Path], None]] = {
    "scatter_gather_dtypes": _fix_onnx_scatter_gather_dtypes,
}
