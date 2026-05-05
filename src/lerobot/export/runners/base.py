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
"""Runner protocol and shared building blocks.

A :class:`Runner` is the export-time and runtime translator for one inference
pattern (single-pass or KV-cache in this carve-out). At export it consumes a
policy and produces ``ExportModule`` specs the backend will trace; at runtime
it loads artifacts via a :class:`~lerobot.export.interfaces._RuntimeSession` and
turns observations into action chunks.

Concrete runners live alongside this module and self-register via
:func:`register_runner`. To resolve which runner handles a manifest, callers
match ``manifest.model.runner["type"]`` against ``Runner.type``.

Example::

    from lerobot.export.runners.base import RUNNERS, register_runner


    @register_runner
    class MyRunner:
        type = "my_pattern"
        ...


    runner_cls = next(r for r in RUNNERS if r.type == "my_pattern")
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from ..interfaces import _RuntimeSession

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor, nn

logger = logging.getLogger(__name__)

__all__ = [
    "ExportModule",
    "RUNNERS",
    "Runner",
    "build_dynamic_axes",
    "get_output_by_names",
    "register_runner",
]


@dataclass
class ExportModule:
    """Specification for a single traceable model stage.

    Bundles everything the backend needs to trace and serialise one stage of
    the policy (e.g. ``"model"``, ``"encoder"``, ``"denoise"``).

    Attributes:
        name: Stage identifier used as the artifact filename stem and as the
            key passed to :meth:`~lerobot.export.interfaces._RuntimeSession.run`.
        wrapper: The ``nn.Module`` to trace.
        example_inputs: Tuple of example tensors used for tracing.
        input_names: Ordered list of input tensor names.
        output_names: Ordered list of output tensor names.
        dynamic_axes: Optional dict mapping tensor names to dynamic axis
            indices (e.g. ``{"obs": {0: "batch_size"}}``).
        hints: Backend-specific hints (e.g. ``{"onnx_fixups": [...]}``) that
            are applied after serialisation.
    """

    name: str
    wrapper: nn.Module
    example_inputs: tuple[Tensor, ...]
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]] | None = None
    hints: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Runner(Protocol):
    """Protocol that every inference runner must satisfy.

    A runner translates between the export framework and a specific inference
    pattern.  At export time it produces :class:`ExportModule` specs; at
    runtime it wraps a :class:`~lerobot.export.interfaces._RuntimeSession` and
    turns observation dicts into action arrays.

    Concrete runners self-register via :func:`register_runner` and are
    discovered by matching ``Runner.type`` against the manifest's
    ``model.runner["type"]`` field.
    """

    type: ClassVar[str]

    @classmethod
    def matches(cls, policy: object) -> bool:
        """Return ``True`` if this runner handles the given policy.

        Args:
            policy: Policy instance to test.

        Returns:
            ``True`` when the runner can export and run this policy.
        """
        ...

    @classmethod
    def export(
        cls,
        policy: object,
        example_batch: dict[str, Tensor],
    ) -> tuple[list[ExportModule], dict[str, Any]]:
        """Produce export modules and runner config from a policy.

        Args:
            policy: Policy instance implementing the Exportable protocol.
            example_batch: Representative input batch for tracing.

        Returns:
            A 2-tuple of ``(modules, runner_cfg)`` where ``modules`` is the
            ordered list of :class:`ExportModule` specs and ``runner_cfg`` is
            a flat dict of runner parameters written into the manifest.
        """
        ...

    @classmethod
    def load(
        cls,
        manifest: dict[str, Any],
        artifacts_dir: Path,
        runtime_session: _RuntimeSession,
    ) -> Runner:
        """Instantiate a runner from a loaded manifest and backend session.

        Args:
            manifest: Parsed manifest dict.
            artifacts_dir: Directory containing the artifact files.
            runtime_session: Open runtime session for inference.

        Returns:
            A ready-to-use :class:`Runner` instance.
        """
        ...

    def run(self, batch: dict[str, np.ndarray], **kwargs: Any) -> np.ndarray:
        """Run inference and return the full action chunk.

        Args:
            batch: Dict mapping observation key names to numpy arrays.

        Returns:
            Action array of shape ``(chunk_size, action_dim)`` or
            ``(batch, chunk_size, action_dim)``.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state between episodes."""
        ...


RUNNERS: list[type[Runner]] = []


def register_runner(cls: type[Runner]) -> type[Runner]:
    """Register a runner class in the global :data:`RUNNERS` list.

    Intended as a class decorator.  The runner is appended in declaration
    order; the first matching runner wins during export and load.

    Args:
        cls: Runner class to register.

    Returns:
        The same class (decorator pass-through).
    """
    RUNNERS.append(cls)
    return cls


def build_dynamic_axes(input_names: list[str], output_names: list[str]) -> dict[str, dict[int, str]]:
    """Build an ONNX ``dynamic_axes`` dict with batch dimension marked for all tensors.

    Args:
        input_names: List of input tensor names.
        output_names: List of output tensor names.

    Returns:
        Dict mapping every tensor name to ``{0: "batch_size"}``.
    """
    dynamic_axes: dict[str, dict[int, str]] = {}
    for name in input_names:
        dynamic_axes[name] = {0: "batch_size"}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}
    return dynamic_axes


def get_output_by_names(
    outputs: dict[str, NDArray[np.floating]],
    primary_name: str,
    fallback_names: list[str],
    context: str,
) -> NDArray[np.floating]:
    """Retrieve a named output tensor with graceful fallback handling.

    Tries ``primary_name`` first, then each name in ``fallback_names`` in
    order.  If none match but exactly one output exists, emits a warning and
    returns it.  Raises :exc:`KeyError` when multiple outputs exist and none
    match.

    Args:
        outputs: Dict of output tensors returned by the backend session.
        primary_name: Preferred output tensor name.
        fallback_names: Alternative names to try if ``primary_name`` is absent.
        context: Human-readable caller label used in log/warning messages.

    Returns:
        The matched output numpy array.

    Raises:
        KeyError: If no matching output is found and multiple outputs exist.
    """
    if primary_name in outputs:
        return outputs[primary_name]

    for name in fallback_names:
        if name in outputs:
            logger.debug("%s: using fallback output '%s' instead of '%s'", context, name, primary_name)
            return outputs[name]

    if len(outputs) == 1:
        actual_name = next(iter(outputs.keys()))
        warnings.warn(
            f"{context}: Expected output '{primary_name}' not found. "
            f"Using only available output '{actual_name}'.",
            stacklevel=3,
        )
        return outputs[actual_name]

    raise KeyError(
        f"{context}: Expected output '{primary_name}' not found. Available: {list(outputs.keys())}."
    )
