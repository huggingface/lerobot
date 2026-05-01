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
"""Registry and dispatch for policy-specific ONNX export wrappers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


@dataclass
class ExportSpec:
    """Specification for a single ONNX export: names, axes, and sample inputs."""

    input_names: list[str]
    output_names: list[str]
    sample_inputs: tuple[Tensor, ...]
    # Used by the legacy ONNX exporter (torch.onnx.export without dynamo).
    dynamic_axes: dict[str, dict[int, str]] | None = None
    # Used by the dynamo ONNX exporter (torch.onnx.export(..., dynamo=True)).
    # Maps input name -> {dim_idx: torch.export.Dim} (or None for static).
    dynamic_shapes: dict | None = None
    policy_note: str = ""  # human-readable description of what is exported


# Type alias for wrapper factory functions.
WrapperFactory = Callable[["PreTrainedPolicy", object], tuple[nn.Module, ExportSpec]]

WRAPPER_REGISTRY: dict[str, WrapperFactory] = {}


def register_export_wrapper(policy_type: str, factory: WrapperFactory | None = None):
    """Register a custom export wrapper factory for a policy type.

    Can be used as a decorator or called directly:

        @register_export_wrapper("my_policy")
        def make_my_wrapper(policy, cfg):
            return MyWrapper(policy), ExportSpec(...)

        register_export_wrapper("my_policy", make_my_wrapper)
    """
    if factory is None:
        # Used as a decorator: @register_export_wrapper("type")
        def decorator(fn: WrapperFactory) -> WrapperFactory:
            WRAPPER_REGISTRY[policy_type] = fn
            return fn

        return decorator
    WRAPPER_REGISTRY[policy_type] = factory
    return factory


def make_export_wrapper(policy: PreTrainedPolicy, cfg) -> tuple[nn.Module, ExportSpec]:
    """Return an ONNX-compatible (wrapper, spec) pair for the given policy.

    Dispatches to a registered factory when available, otherwise falls back to a
    generic wrapper that calls ``policy.model`` directly.
    """
    from .wrappers import _make_act_wrapper, _make_diffusion_wrapper, _make_generic_wrapper

    # Register built-in factories the first time we look up a policy type.
    # User-registered factories (via register_export_wrapper) take precedence.
    WRAPPER_REGISTRY.setdefault("act", _make_act_wrapper)
    WRAPPER_REGISTRY.setdefault("diffusion", _make_diffusion_wrapper)

    policy_type = policy.config.type
    factory = WRAPPER_REGISTRY.get(policy_type)
    if factory is not None:
        return factory(policy, cfg)

    logger.warning(
        f"No dedicated export wrapper found for policy type '{policy_type}'. "
        "Falling back to a generic wrapper — this may not work for all architectures. "
        f"Register a custom wrapper with: register_export_wrapper('{policy_type}', my_factory)"
    )
    return _make_generic_wrapper(policy, cfg)
