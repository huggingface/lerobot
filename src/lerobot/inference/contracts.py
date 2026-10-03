# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Canonical, transport-independent contracts for asynchronous inference.

Feature order is explicit. RGB observations use HWC uint8; arbitrary tensors are
never implicitly treated as images. Action values are canonical processor outputs,
before robot-side processing. Client monotonic capture times are opaque on a server.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import numpy as np
import torch


class ExecutionMode(StrEnum):
    """An explicitly negotiated action execution mode."""

    CHUNK = "chunk"
    RTC_GUIDED = "rtc_guided"
    RTC_TRAINED = "rtc_trained"


@dataclass(frozen=True)
class FeatureSpec:
    """Named canonical values, ordered components, and semantic conventions."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str = "tensor"
    names: tuple[str, ...] = ()
    semantics: str = ""

    def __post_init__(self) -> None:
        """Reject unsupported modalities and ambiguous feature metadata."""
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "names", tuple(self.names))
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or not isinstance(self.semantics, str)
            or not self.semantics.strip()
        ):
            raise ValueError("Feature names and explicit semantic conventions are required.")
        if self.kind not in {"tensor", "rgb"}:
            raise ValueError(f"Unsupported feature modality: {self.kind!r}")
        if len(self.shape) > 4 or any(type(dim) is not int or dim <= 0 for dim in self.shape):
            raise ValueError(f"Invalid feature shape: {self.shape!r}")
        if self.dtype not in {"bool", "uint8", "int32", "int64", "float32", "float64"}:
            raise ValueError(f"Unsupported feature dtype: {self.dtype!r}")
        if self.kind == "rgb" and (len(self.shape) != 3 or self.shape[-1] != 3 or self.dtype != "uint8"):
            raise ValueError("RGB features require HWC uint8 with three channels.")
        if self.names and (len(self.shape) != 1 or len(self.names) != self.shape[0]):
            raise ValueError("Ordered component names must match a one-dimensional feature.")
        if any(not isinstance(name, str) or not name.strip() for name in self.names):
            raise ValueError("Feature component names must be nonempty strings.")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Feature component names must be unique.")


@dataclass(frozen=True)
class ChunkPolicySpec:
    """Policy author declaration; inheriting the default is an explicit contract.

    The current observation suffices, direct chunk calls perform all preparation,
    outputs are [B, prediction_steps, A], and canonical processors accept full chunks.
    Mutable policy/processor state belongs to the exclusive session, never the network.
    """

    prediction_steps: int
    execution_steps: int
    modes: tuple[ExecutionMode, ...] = (ExecutionMode.CHUNK,)
    current_observation_only: bool = True
    retains_session_state: bool = True
    training_max_delay: int = 0

    def __post_init__(self) -> None:
        """Validate the policy's declared prediction and execution lengths."""
        if not 0 < self.execution_steps <= self.prediction_steps:
            raise ValueError("Execution length must be positive and no longer than prediction length.")


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
