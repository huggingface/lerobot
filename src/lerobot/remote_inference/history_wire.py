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

"""Wire form of declared chunk history and of the history sampled for one request.

The robot client samples history from its own execution record and sends it with
each request, so a server session never reconstructs past ticks. Descriptors only
advertise history fields for deployments that declare them, which keeps the wire
schema of current-observation deployments unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from lerobot.inference import (
    ChunkHistory,
    FeatureHistory,
    FeatureSpec,
    HistoryInputs,
    PolicyCapabilities,
)

from .codec import RGBImage
from .protocol import ErrorCode, ProtocolError


def capabilities_to_wire(capabilities: dict[str, Any]) -> dict[str, Any]:
    """Drop history fields a current-observation deployment does not use (``asdict`` input)."""
    if not capabilities["history"]["observations"] and not capabilities["history"]["actions"]:
        capabilities.pop("history")
    return capabilities


def history_from_wire(value: dict[str, Any] | None) -> ChunkHistory:
    """Rebuild a declared history from a descriptor."""
    if value is None:
        return ChunkHistory()
    return ChunkHistory(
        observations=tuple(FeatureHistory(w["feature"], tuple(w["offsets"])) for w in value["observations"]),
        actions=tuple(value["actions"]),
    )


def encode_history(
    history: HistoryInputs,
    capabilities: PolicyCapabilities,
    encoding: Literal["raw", "jpeg"],
    jpeg_quality: int,
) -> dict[str, Any]:
    """Encode sampled history; RGB values use the observation image encoding."""
    rgb = {feature.name for feature in capabilities.features if feature.kind == "rgb"}

    def values(name: str, items: tuple[Any, ...]) -> list[Any]:
        return [RGBImage(item, encoding, jpeg_quality) if name in rgb else item for item in items]

    return {
        "observations": {name: values(name, items) for name, items in history.observations.items()},
        "observation_padding": {name: list(pads) for name, pads in history.observation_padding.items()},
        "actions": list(history.actions),
        "action_padding": list(history.action_padding),
        "max_skew_s": history.max_skew_s,
    }


def decode_history(value: Any, capabilities: PolicyCapabilities) -> HistoryInputs | None:
    """Validate a request's history against the deployment's declaration.

    Raises:
        ProtocolError: For malformed values or a history that differs from the declaration.
    """
    declared = capabilities.history
    if value is None:
        if not declared.empty:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "This deployment requires sampled history")
        return None
    if declared.empty:
        raise ProtocolError(ErrorCode.INCOMPATIBLE, "This deployment does not accept history")
    if not isinstance(value, dict):
        raise ProtocolError(ErrorCode.MALFORMED, "Invalid history")
    features = {feature.name: feature for feature in capabilities.features}

    def arrays(feature: FeatureSpec, items: Any, count: int, what: str) -> tuple[np.ndarray, ...]:
        if not isinstance(items, list) or len(items) != count:
            raise ProtocolError(
                ErrorCode.INCOMPATIBLE, f"{what} for {feature.name} differs from the deployment"
            )
        for item in items:
            if (
                not isinstance(item, np.ndarray)
                or item.shape != feature.shape
                or item.dtype.name != feature.dtype
            ):
                raise ProtocolError(ErrorCode.INCOMPATIBLE, f"Invalid {what.lower()} for {feature.name}")
            if not np.isfinite(item).all():
                raise ProtocolError(ErrorCode.MALFORMED, f"Non-finite {what.lower()} for {feature.name}")
        return tuple(items)

    def flags(items: Any, count: int, what: str) -> tuple[bool, ...]:
        if (
            not isinstance(items, list)
            or len(items) != count
            or any(type(item) is not bool for item in items)
        ):
            raise ProtocolError(ErrorCode.MALFORMED, f"Invalid {what} padding")
        return tuple(items)

    observations, padding = value.get("observations"), value.get("observation_padding")
    if not isinstance(observations, dict) or not isinstance(padding, dict):
        raise ProtocolError(ErrorCode.MALFORMED, "Invalid history windows")
    if set(observations) != {window.feature for window in declared.observations} or set(padding) != set(
        observations
    ):
        raise ProtocolError(ErrorCode.INCOMPATIBLE, "History windows differ from the deployment")
    windows = {
        w.feature: arrays(features[w.feature], observations[w.feature], len(w.offsets) - 1, "History")
        for w in declared.observations
    }
    window_pads = {
        w.feature: flags(padding[w.feature], len(w.offsets) - 1, "history") for w in declared.observations
    }
    actions = arrays(
        capabilities.action_feature, value.get("actions"), len(declared.actions), "Command history"
    )
    action_pads = flags(value.get("action_padding"), len(declared.actions), "command history")
    skew = value.get("max_skew_s")
    if not isinstance(skew, (int, float)) or not np.isfinite(skew) or skew < 0:
        raise ProtocolError(ErrorCode.MALFORMED, "Invalid history skew")
    return HistoryInputs(
        observations=windows,
        observation_padding=window_pads,
        actions=actions,
        action_padding=action_pads,
        max_skew_s=float(skew),
    )
