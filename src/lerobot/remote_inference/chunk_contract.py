# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Validation shared by serving configuration and plain-chunk admission."""

import math
from typing import Any

from lerobot.inference.contracts import ExecutionMode, FeatureSpec, PolicyCapabilities

CHUNK_ALIGNMENT = "chunk_alignment_v1"
CHUNK_BLENDING = "chunk_blending_v1"
RTC_MODEL_SPACE = "rtc_model_space_v1"


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


def validate_blendable_components(action: FeatureSpec, components: list[str] | tuple[str, ...]) -> None:
    """Require an operator-selected subset of named canonical float coordinates."""
    if not isinstance(components, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in components
    ):
        raise ValueError("blendable_components must contain nonempty component names")
    if len(set(components)) != len(components):
        raise ValueError("blendable_components must be unique")
    if components and (
        action.kind != "tensor"
        or action.dtype != "float32"
        or len(action.shape) != 1
        or not action.names
        or not set(components).issubset(action.names)
    ):
        raise ValueError("blendable_components requires named float32 canonical action coordinates")


def required_chunk_capabilities(settings: dict[str, Any]) -> list[str]:
    """Return the exact additional protocol semantics requested by these options."""
    required = [CHUNK_ALIGNMENT] if settings["chunk_merge"] == "aligned" else []
    if settings["blend_steps"]:
        required.append(CHUNK_BLENDING)
    return required


def validate_chunk_contract(
    settings: dict[str, Any], caps: PolicyCapabilities, blendable_components: list[str] | tuple[str, ...]
) -> tuple[int, ...]:
    """Validate negotiated settings and resolve selected names to canonical indices."""
    if not isinstance(settings, dict) or set(settings) != {
        "chunk_merge",
        "blend_steps",
        "blend_weight",
        "blend_components",
    }:
        raise ValueError("Invalid chunk_settings fields")
    chunk_settings(**settings)
    if settings["chunk_merge"] == "aligned" and (
        ExecutionMode.CHUNK not in caps.modes
        or caps.action_representation != "canonical"
        or caps.action_feature.kind != "tensor"
        or caps.action_feature.dtype != "float32"
        or len(caps.action_feature.shape) != 1
    ):
        raise ValueError("Aligned chunks require canonical float32 vector actions")
    validate_blendable_components(caps.action_feature, blendable_components)
    if settings["blend_steps"] > caps.execution_steps:
        raise ValueError("blend_steps must not exceed the configured execution slice")
    if not set(settings["blend_components"]).issubset(blendable_components):
        raise ValueError("Requested blend_components are not enabled by the deployment")
    return tuple(caps.action_feature.names.index(name) for name in settings["blend_components"])
