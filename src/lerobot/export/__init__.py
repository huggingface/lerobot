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
"""LeRobot Policy Export module.

Export LeRobot policies to portable formats (ONNX, OpenVINO) for inference
without the full training stack.

Example::

    from lerobot.export import export_policy, load_exported_policy

    package_path = export_policy(policy, "./exported", backend="onnx")
    exported_policy = load_exported_policy(package_path, backend="onnx", device="cpu")
    action = exported_policy.select_action(observation)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import (
    backends as _backends,  # noqa: F401
    runners as _runners,  # noqa: F401
)
from .manifest import Manifest
from .policy import ExportedPolicy

if TYPE_CHECKING:
    from torch import Tensor


def export_policy(
    policy: object,
    output_dir: str | Path,
    *,
    backend: str = "onnx",
    example_batch: dict[str, Tensor] | None = None,
    opset_version: int = 17,
    include_normalization: bool = True,
) -> Path:
    """Export a policy package, importing the PyTorch exporter only when needed."""
    from .exporter import export_policy as _export_policy

    return _export_policy(
        policy,
        output_dir,
        backend=backend,
        example_batch=example_batch,
        opset_version=opset_version,
        include_normalization=include_normalization,
    )


def load_exported_policy(
    package_path: str | Path,
    backend: str | None = None,
    device: str = "cpu",
) -> ExportedPolicy:
    """Load a policy package and return an exported policy.

    Args:
        package_path: Path to the policy package directory.
        backend: Runtime backend (auto-detected if ``None``).
        device: Device for inference.

    Returns:
        An :class:`ExportedPolicy` ready for inference.
    """
    return ExportedPolicy.load(package_path, backend=backend, device=device)


__all__ = [
    # Main API
    "export_policy",
    "load_exported_policy",
    "ExportedPolicy",
    "Manifest",
]
