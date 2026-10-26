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

"""LIBERO-plus integration using LIBERO's environment and observation conventions."""

from typing import Any

from .libero import conventions, create_envs as _create_envs
from .libero_observation import canonical_observation

__all__ = ["canonical_observation", "conventions", "create_envs"]


def create_envs(**kwargs: Any) -> dict[str, dict[int, Any]]:
    """Create LIBERO-plus environments through the shared LIBERO wrapper."""
    return _create_envs(**{**kwargs, "is_libero_plus": True})
