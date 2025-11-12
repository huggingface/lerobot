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

"""Utilities for handling optional dependencies with consistent error messages."""

from __future__ import annotations

import importlib
from typing import Any


def optional_import(module: str, extra: str, attr: str | None = None) -> Any:
    """
    Import a module (or attribute) required for an optional feature.

    If the import fails, raise a clear ImportError instructing the user to install the
    corresponding extra, e.g. `pip install lerobot[<extra>]`.

    Args:
        module: Dotted module path to import (e.g., "reachy2_sdk").
        extra: Name of the extras group that provides this dependency (e.g., "reachy2").
        attr: Optional attribute/class/function name to fetch from the imported module.

    Returns:
        The imported module or attribute.
    """
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:  # pragma: no cover - message formatting only
        raise ImportError(
            f"Missing optional dependency '{module}'. Install with `pip install lerobot[{extra}]`."
        ) from e

    return getattr(mod, attr) if attr else mod
