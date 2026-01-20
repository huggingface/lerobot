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
"""LeRobot Policy Export module.

Export LeRobot policies to portable formats (ONNX, OpenVINO) for inference
without the full training stack.

Supported Policies:
    - ACT: Single-shot transformer policy
    - Diffusion: Iterative diffusion policy (DDIM/DDPM schedulers)
    - PI0: Two-phase VLA with encoder + decoder (requires patched transformers)
    - SmolVLA: Two-phase VLA with encoder + decoder

Supported Backends:
    - ONNX Runtime: Primary backend, works on all platforms
    - OpenVINO: Optimized for Intel hardware

Example::

    from lerobot.export import export_policy, load

    package_path = export_policy(policy, "./exported", backend="onnx")
    runtime = load(package_path, backend="onnx", device="cpu")
    action_chunk = runtime.predict_action_chunk(observation)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .exporter import export_policy
from .manifest import InferenceConfig, IterativeConfig, Manifest, TwoPhaseConfig
from .runtime import (
    ActionChunkingWrapper,
    IterativeRuntime,
    PolicyRuntime,
    SingleShotRuntime,
    create_runtime,
)

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from lerobot.policies.pretrained import PreTrainedPolicy


def load(
    package_path: str | Path,
    backend: str | None = None,
    device: str = "cpu",
) -> PolicyRuntime:
    """Load a PolicyPackage and return a runtime.

    Args:
        package_path: Path to PolicyPackage directory.
        backend: Backend to use (auto-detected from artifacts if None).
        device: Device for inference ("cpu", "cuda", "cuda:0").

    Returns:
        PolicyRuntime instance ready for inference.

    Example:
        >>> runtime = load("./my_policy.pkg", backend="onnx", device="cuda")
        >>> action_chunk = runtime.predict_action_chunk(observation)
    """
    return create_runtime(package_path, backend=backend, device=device)


__all__ = [
    # Main API
    "export_policy",
    "load",
    # Runtime classes
    "PolicyRuntime",
    "SingleShotRuntime",
    "IterativeRuntime",
    "ActionChunkingWrapper",
    # Manifest and inference configs
    "Manifest",
    "InferenceConfig",
    "IterativeConfig",
    "TwoPhaseConfig",
]
