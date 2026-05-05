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

"""Public and internal protocols for the export subsystem.

This module is the canonical home for the public ``Backend`` protocol and the
internal ``_RuntimeSession`` protocol. Both ``backends`` and ``runners``
packages import from here so neither needs to depend on the other; this is the
boundary that keeps the two halves of the export subsystem decoupled regardless
of artifact format or runtime engine.

External consumers should import only ``Backend`` from this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from .runners.base import ExportModule

__all__ = ["Backend"]


@runtime_checkable
class _RuntimeSession(Protocol):
    """Internal protocol for a live inference session opened by a :class:`Backend`.

    A runtime adapter wraps one or more loaded model stages and exposes a single
    :meth:`run` method that accepts named numpy inputs and returns named
    numpy outputs.
    """

    def run(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Execute one named model stage.

        Args:
            name: Stage identifier (e.g. ``"model"``, ``"encoder"``,
                ``"denoise"``).
            inputs: Dict mapping input tensor names to numpy arrays.

        Returns:
            Dict mapping output tensor names to numpy arrays.
        """
        ...


@runtime_checkable
class Backend(Protocol):
    """Protocol for a serialisation / runtime backend.

    Backends are responsible for two concerns:

    1. **Serialisation** — tracing a list of :class:`ExportModule` specs and
       writing artifact files to disk (e.g. ``.onnx`` files).
    2. **Opening** — loading those artifacts at runtime and returning a
       :class:`_RuntimeSession` ready for inference.

    Concrete implementations register themselves via
    :func:`~lerobot.export.backends.base.register_backend`.
    """

    name: ClassVar[str]
    extension: ClassVar[str]
    runtime_only: ClassVar[bool] = False

    def serialize(
        self,
        modules: list[ExportModule],
        artifacts_dir: Path,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Trace and serialise export modules to artifact files.

        Args:
            modules: Ordered list of :class:`ExportModule` specs to serialise.
            artifacts_dir: Directory where artifact files should be written.
            **kwargs: Backend-specific options (e.g. ``opset_version`` for
                ONNX).

        Returns:
            Dict mapping module name to the artifact filename (relative to
            ``artifacts_dir``).
        """
        ...

    def open(
        self,
        artifacts_dir: Path,
        manifest: dict[str, Any],
        *,
        device: str = "cpu",
    ) -> _RuntimeSession:
        """Load artifact files and return a ready-to-use session.

        Args:
            artifacts_dir: Directory containing the artifact files.
            manifest: Parsed manifest dict (used to resolve artifact paths).
            device: Target device string (e.g. ``"cpu"``, ``"cuda:0"``).

        Returns:
            A :class:`_RuntimeSession` ready for inference.
        """
        ...
