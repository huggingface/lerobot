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
"""Dispatch and helpers for policy-specific ONNX export wrappers.

Per-policy export adapters live next to each policy module under
``lerobot/policies/<type>/export_<type>.py`` and expose a function named
``make_<type>_export_wrapper(policy, cfg) -> (nn.Module, ExportSpec)``. They
are auto-discovered by :func:`make_export_wrapper`.

External plugins or runtime overrides can also register a factory explicitly
via :func:`register_export_wrapper`.
"""

from __future__ import annotations

import importlib
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
    # Tuple form (one entry per positional sample input) is recommended because
    # it works with both regular and *args-style wrapper signatures.
    dynamic_shapes: dict | tuple | list | None = None
    policy_note: str = ""  # human-readable description of what is exported


# Type alias for wrapper factory functions.
WrapperFactory = Callable[["PreTrainedPolicy", object], tuple[nn.Module, ExportSpec]]

# Explicit registry for runtime overrides and third-party plugins. In-tree
# policies should not populate this dict directly; they are auto-discovered
# via the lerobot.policies.<type>.export_<type> convention.
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
        def decorator(fn: WrapperFactory) -> WrapperFactory:
            WRAPPER_REGISTRY[policy_type] = fn
            return fn

        return decorator
    WRAPPER_REGISTRY[policy_type] = factory
    return factory


def _try_load_builtin_factory(policy_type: str) -> WrapperFactory | None:
    """Auto-discover a per-policy export factory via the lerobot convention.

    Looks for ``lerobot.policies.<policy_type>.export_<policy_type>:make_<policy_type>_export_wrapper``.
    Returns ``None`` if the module or function is missing.
    """
    module_path = f"lerobot.policies.{policy_type}.export_{policy_type}"
    factory_name = f"make_{policy_type}_export_wrapper"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return None
    return getattr(module, factory_name, None)


def make_export_wrapper(policy: PreTrainedPolicy, cfg) -> tuple[nn.Module, ExportSpec]:
    """Return an ONNX-compatible (wrapper, spec) pair for the given policy.

    Dispatch order:

    1. Explicitly registered factories (``register_export_wrapper`` / WRAPPER_REGISTRY).
    2. Auto-discovered factories at ``lerobot.policies.<type>.export_<type>``.
    3. Otherwise, raise ``NotImplementedError`` with a helpful message.
    """
    policy_type = policy.config.type

    # 1. Explicit registration takes precedence (third-party / runtime overrides).
    factory = WRAPPER_REGISTRY.get(policy_type)
    if factory is not None:
        return factory(policy, cfg)

    # 2. Auto-discovery via lerobot convention.
    factory = _try_load_builtin_factory(policy_type)
    if factory is not None:
        return factory(policy, cfg)

    raise NotImplementedError(
        f"Export support for policy type '{policy_type}' is not implemented.\n"
        f"To add it, create src/lerobot/policies/{policy_type}/export_{policy_type}.py "
        f"with a `make_{policy_type}_export_wrapper(policy, cfg)` factory.\n"
        f"See src/lerobot/policies/act/export_act.py for a reference implementation.\n"
        f"For third-party policies, register a factory at runtime instead:\n"
        f"  from lerobot.export import register_export_wrapper\n"
        f"  @register_export_wrapper('{policy_type}')\n"
        f"  def my_factory(policy, cfg): ..."
    )


def make_batch_dynamic_axes_and_shapes(
    input_names: list[str],
    sample_inputs: tuple[Tensor, ...],
    output_names: list[str] | None = None,
    batch_dim_name: str = "batch_size",
) -> tuple[dict[str, dict[int, str]], tuple]:
    """Generate (dynamic_axes, dynamic_shapes) for the common batch-dynamic case.

    Every input gets dim 0 declared as a symbolic batch dimension. Output names
    are also added to ``dynamic_axes`` (the legacy exporter requires this).

    Args:
        input_names:    Names of inputs in the ONNX graph.
        sample_inputs:  Positional sample tensors used for tracing. Length must
                        match the number of positional args of the wrapper's
                        ``forward`` (after `*args` flattening).
        output_names:   Optional output names to include in ``dynamic_axes``.
        batch_dim_name: Name of the symbolic batch dimension.

    Returns:
        ``(dynamic_axes, dynamic_shapes)`` — the former for the legacy exporter,
        the latter for the dynamo exporter.
    """
    from torch.export import Dim

    batch_dim = Dim(batch_dim_name, min=1, max=64)
    dynamic_axes: dict[str, dict[int, str]] = {n: {0: batch_dim_name} for n in input_names}
    if output_names:
        for n in output_names:
            dynamic_axes[n] = {0: batch_dim_name}
    dynamic_shapes = tuple({0: batch_dim} for _ in sample_inputs)
    return dynamic_axes, dynamic_shapes
