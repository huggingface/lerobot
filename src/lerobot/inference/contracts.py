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

"""Canonical, transport-independent contracts for asynchronous inference.

Feature order is explicit. RGB observations use HWC uint8; arbitrary tensors are
never implicitly treated as images. Action values are canonical processor outputs,
before robot-side processing. Client monotonic capture times are opaque on a server.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import numpy as np
import torch

from lerobot.policies import (
    ChunkPolicySpec as ChunkPolicySpec,
    ExecutionMode as ExecutionMode,
    FeatureSpec as FeatureSpec,
)


class QueryKind(Enum):
    """What the policy's text head is being asked for."""

    VQA = "vqa"
    """A free-form question about the current scene; the reply goes to the operator."""

    NEXT_SUBTASK = "next_subtask"
    """A high-level goal; the reply is the next subtask and is fed to ``set_task``."""


@dataclass(frozen=True)
class PolicyCapabilities:
    """Immutable serving contract advertised before session admission."""

    modes: tuple[ExecutionMode, ...]
    prediction_steps: int
    execution_steps: int
    action_interval: float
    features: tuple[FeatureSpec, ...]
    action_feature: FeatureSpec
    language: bool = False
    training_max_delay: int = 0
    rtc_horizon: int = 0
    current_observation_only: bool = True
    retains_session_state: bool = True
    action_representation: str = "canonical"
    model_action_dim: int | None = None

    def __post_init__(self) -> None:
        """Validate lengths, interval, and feature identity."""
        if not math.isfinite(self.action_interval) or self.action_interval <= 0:
            raise ValueError("Policy action interval must be finite and positive.")
        if not self.modes or not 0 < self.execution_steps <= self.prediction_steps:
            raise ValueError("A valid mode and execution length are required.")
        if len({feature.name for feature in self.features}) != len(self.features):
            raise ValueError("Duplicate observation feature names.")
        if self.model_action_dim is not None and (
            type(self.model_action_dim) is not int or self.model_action_dim <= 0
        ):
            raise ValueError("Model action dimension must be a positive integer.")


@dataclass(frozen=True)
class ObservationSnapshot:
    """Owned immutable current observations with original capture provenance."""

    features: Mapping[str, np.ndarray]
    capture_time: float
    task: str
    task_version: int = 0
    observation_id: str = ""
    action_cursor: int | None = None
    execution_generation: int | None = None

    def __post_init__(self) -> None:
        """Copy camera buffers so asynchronous encoding sees a stable frame."""
        if not math.isfinite(self.capture_time):
            raise ValueError("Observation capture time must be finite.")
        for anchor in (self.action_cursor, self.execution_generation):
            if anchor is not None and (type(anchor) is not int or anchor < 0):
                raise ValueError("Observation action anchors must be nonnegative integers.")
        # Cameras commonly recycle their arrays. Own exactly one bounded request
        # snapshot; copies remain read-only until the runner creates its batch.
        owned: dict[str, np.ndarray] = {}
        for name, value in self.features.items():
            array = np.array(value, copy=True, order="C")
            array.setflags(write=False)
            owned[name] = array
        object.__setattr__(self, "features", MappingProxyType(owned))


@dataclass(frozen=True)
class ActionSource:
    """One original contributor identity, without recursive blend history."""

    capture_time: float
    observation_id: str
    request_id: str


@dataclass(frozen=True)
class ActionProvenance:
    """Source identity carried by every queued action, including across merges."""

    capture_time: float
    task: str
    task_version: int = 0
    observation_id: str = ""
    request_id: str = ""
    generation: int = 0
    session_id: str = ""
    server_instance_id: str = ""
    artifact_identity: str = ""
    # The primary IDs identify the incoming prediction. capture_time is the
    # conservative age bound; this summary retains the oldest original source.
    oldest_contributor: ActionSource | None = None
    contributor_count: int = 1
    contributor_digest: str = ""


@dataclass(frozen=True)
class ActionChunk:
    """A runner result in canonical coordinates plus optional RTC model values."""

    model_actions: torch.Tensor | None
    canonical_actions: torch.Tensor
    provenance: ActionProvenance
    execution_steps: int
    server_durations: dict[str, float] = field(default_factory=dict)
