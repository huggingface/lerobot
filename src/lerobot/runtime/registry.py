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

"""Lazy mapping from policy types to language-runtime adapters."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

_ADAPTERS: dict[str, str] = {
    "pi052": "lerobot.policies.pi052.inference.pi052_adapter:PI052PolicyAdapter",
    "pi05": "lerobot.runtime.adapter:DirectTaskPolicyAdapter",
    "molmoact2": "lerobot.runtime.adapter:DirectTaskPolicyAdapter",
}


def get_language_adapter_factory(policy_type: str) -> Callable[..., Any]:
    """Return the adapter class registered for ``policy_type``."""
    spec = _ADAPTERS.get(policy_type)
    if spec is None:
        raise ValueError(
            f"No language-runtime adapter registered for policy type {policy_type!r}. "
            f"Registered: {sorted(_ADAPTERS)}. Add an entry to lerobot.runtime.registry."
        )
    module_path, class_name = spec.split(":")
    return getattr(importlib.import_module(module_path), class_name)
