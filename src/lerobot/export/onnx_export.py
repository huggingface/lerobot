# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Export an nn.Module wrapper to ONNX format.

Two exporter backends are supported:

- ``"legacy"`` — ``torch.onnx.export`` with the TorchScript tracer. Stable but
  bakes Python control flow and ``torch.zeros([batch_size, ...])`` calls as
  constants (e.g. ACT becomes batch_size=1 only).
- ``"dynamo"`` — ``torch.onnx.export(..., dynamo=True)``. Uses ``torch.export``
  and supports symbolic shapes via ``torch.export.Dim``. Required for ACT
  with batch_size > 1.

The default ``"auto"`` mode tries dynamo first and falls back to legacy with
a warning if dynamo fails.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from lerobot.utils.import_utils import require_package

from .core import ExportSpec

logger = logging.getLogger(__name__)


def export_to_onnx(
    wrapper: nn.Module,
    spec: ExportSpec,
    output_path: Path | str,
    opset_version: int = 18,
    precision: str = "fp32",
    exporter: str = "auto",
) -> Path:
    """Export an ONNX-compatible wrapper module to a ``.onnx`` file.

    Args:
        wrapper:       The ``nn.Module`` wrapper to export.
        spec:          ``ExportSpec`` with input/output names, axes/shapes, and sample inputs.
        output_path:   Destination path. The ``.onnx`` suffix is added automatically.
        opset_version: ONNX opset (default 18 for native LayerNorm + improved attention).
        precision:     ``"fp32"`` or ``"fp16"``. When ``"fp16"`` the wrapper and sample
                       inputs are cast to half precision before tracing.
        exporter:      ``"auto"`` (default) | ``"dynamo"`` | ``"legacy"``.

    Returns:
        Path to the written ``.onnx`` file.
    """
    require_package("onnx", extra="export")
    import onnx

    if exporter not in ("auto", "dynamo", "legacy"):
        raise ValueError(f"Invalid exporter='{exporter}'. Use 'auto', 'dynamo', or 'legacy'.")

    output_path = Path(output_path).with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper.eval()
    sample_inputs = spec.sample_inputs

    if precision == "fp16":
        wrapper = wrapper.half()
        sample_inputs = tuple(x.half() if x.is_floating_point() else x for x in sample_inputs)

    logger.info(
        f"Exporting ONNX to {output_path} (opset={opset_version}, precision={precision}, exporter={exporter})"
    )
    if spec.policy_note:
        logger.info(f"  Note: {spec.policy_note}")

    def _try_dynamo() -> None:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                sample_inputs,
                str(output_path),
                opset_version=opset_version,
                input_names=spec.input_names,
                output_names=spec.output_names,
                dynamic_shapes=spec.dynamic_shapes,
                dynamo=True,
            )

    def _legacy() -> None:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                sample_inputs,
                str(output_path),
                opset_version=opset_version,
                input_names=spec.input_names,
                output_names=spec.output_names,
                dynamic_axes=spec.dynamic_axes,
                do_constant_folding=True,
                dynamo=False,  # explicit: PyTorch 2.7+ may default this to True
            )

    if exporter == "dynamo":
        _try_dynamo()
    elif exporter == "legacy":
        _legacy()
    else:  # auto
        try:
            _try_dynamo()
        except Exception as exc:
            logger.warning(
                f"dynamo export failed ({exc.__class__.__name__}: {exc}); falling back to legacy tracing."
            )
            _legacy()

    # Structural validity check. Pass the path so the checker can lazily
    # load external data for >2 GiB models (e.g. VLA-class checkpoints).
    onnx.checker.check_model(str(output_path))
    logger.info(f"ONNX export successful: {output_path}")
    return output_path
