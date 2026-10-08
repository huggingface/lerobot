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

"""Alignment admission and client-owned blending validation."""

from typing import Any

from lerobot.inference import ExecutionMode, PolicyCapabilities, chunk_settings as chunk_settings

CHUNK_ALIGNMENT = "chunk_alignment_v2"
RTC_MODEL_SPACE = "rtc_model_space_v1"


def default_chunk_settings() -> dict[str, Any]:
    """Return fresh settings for peers that omit the optional chunk contract."""
    return {"chunk_merge": "append"}


def required_chunk_capabilities(settings: dict[str, Any]) -> list[str]:
    """Return the exact additional protocol semantics requested by these options."""
    return [CHUNK_ALIGNMENT] if settings["chunk_merge"] == "aligned" else []


def validate_chunk_contract(settings: dict[str, Any], caps: PolicyCapabilities) -> None:
    """Negotiate cursor alignment independently of local playback options."""
    if not isinstance(settings, dict) or set(settings) != {"chunk_merge"}:
        raise ValueError("Invalid chunk_settings fields")
    if not isinstance(settings["chunk_merge"], str) or settings["chunk_merge"] not in {"append", "aligned"}:
        raise ValueError("chunk_merge must be append or aligned")
    if settings["chunk_merge"] == "aligned" and (
        ExecutionMode.CHUNK not in caps.modes
        or caps.action_representation != "canonical"
        or caps.action_feature.kind != "tensor"
        or caps.action_feature.dtype != "float32"
        or len(caps.action_feature.shape) != 1
    ):
        raise ValueError("Aligned chunks require canonical float32 vector actions")


def validate_blend_settings(settings: dict[str, Any], caps: PolicyCapabilities) -> tuple[int, ...]:
    """Resolve user-selected canonical coordinates for client-side blending."""
    chunk_settings(**settings)
    validate_chunk_contract({"chunk_merge": settings["chunk_merge"]}, caps)
    if settings["blend_steps"] > caps.execution_steps:
        raise ValueError("blend_steps must not exceed the configured execution slice")
    if not set(settings["blend_components"]).issubset(caps.action_feature.names):
        raise ValueError("blend_components must name canonical action coordinates")
    return tuple(caps.action_feature.names.index(name) for name in settings["blend_components"])
