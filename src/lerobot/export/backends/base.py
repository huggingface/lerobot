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

from pathlib import Path
from typing import Any

from ..interfaces import Backend

__all__ = ["BACKENDS", "register_backend", "resolve_artifact_paths"]


BACKENDS: dict[str, Backend] = {}


def register_backend(cls: type[Backend]) -> type[Backend]:
    """Register a backend class in the global :data:`BACKENDS` dict.

    Instantiates the class, ensures ``runtime_only`` is set, and stores the
    instance keyed by ``cls.name``.  Intended as a class decorator.

    Args:
        cls: Backend class to register.

    Returns:
        The same class (decorator pass-through).
    """
    backend = cls()
    if not hasattr(backend, "runtime_only"):
        backend.runtime_only = False
    BACKENDS[cls.name] = backend
    return cls


def resolve_artifact_paths(artifacts_dir: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    """Resolve artifact names from the manifest to absolute paths.

    Args:
        artifacts_dir: Directory containing the artifact files.
        manifest: Parsed manifest dict with a ``model.artifacts`` section.

    Returns:
        Dict mapping artifact role name (e.g. ``"model"``) to the absolute
        :class:`~pathlib.Path` of the corresponding file.
    """
    return {
        name: artifacts_dir / Path(relative_path).name
        for name, relative_path in manifest["model"]["artifacts"].items()
    }
