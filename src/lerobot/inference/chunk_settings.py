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

"""Transport-independent validation of chunk merge options."""

import math
from typing import Any


def chunk_settings(
    chunk_merge: str, blend_steps: int, blend_weight: float, blend_components: list[str]
) -> dict[str, Any]:
    """Validate explicit merge options without requiring a connected deployment."""
    if chunk_merge not in {"append", "aligned"}:
        raise ValueError("chunk_merge must be append or aligned")
    if type(blend_steps) is not int or not 0 <= blend_steps < 2**31:
        raise ValueError("blend_steps must be a bounded nonnegative integer")
    if type(blend_weight) not in (float, int) or not math.isfinite(blend_weight) or not 0 < blend_weight <= 1:
        raise ValueError("blend_weight must be finite and in (0, 1]")
    if (
        not isinstance(blend_components, list)
        or any(not isinstance(name, str) or not name for name in blend_components)
        or len(set(blend_components)) != len(blend_components)
    ):
        raise ValueError("blend_components must contain unique, nonempty component names")
    if blend_steps and (chunk_merge != "aligned" or not blend_components):
        raise ValueError("Blending requires chunk_merge=aligned and explicit blend_components")
    if not blend_steps and blend_components:
        raise ValueError("blend_components requires positive blend_steps")
    return {
        "chunk_merge": chunk_merge,
        "blend_steps": blend_steps,
        "blend_weight": float(blend_weight),
        "blend_components": list(blend_components),
    }
