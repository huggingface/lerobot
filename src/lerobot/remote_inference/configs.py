# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Operator-owned serving configuration; clients cannot request model downloads."""

import math
from dataclasses import dataclass, field
from typing import Literal

from lerobot.inference.contracts import FeatureSpec
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.transport.zenoh import ZenohConfig

from .chunk_contract import validate_blendable_components


@dataclass
class ModelConfig:
    """Operator-selected checkpoint and compute placement."""

    repo_or_path: str = ""
    revision: str | None = None
    device: str = "cpu"


@dataclass
class ExecutionConfig:
    """Deployment-owned modes, timing bounds and warmup settings."""

    supported_modes: list[str] = field(default_factory=lambda: ["chunk"])
    action_fps: float = 30.0
    rtc: RTCConfig = field(default_factory=RTCConfig)
    action_deadline_s: float = 5.0
    idle_timeout_s: float = 10.0
    warmup_calls: int = 2
    # Explicit continuous canonical coordinates; never infer gripper suitability.
    blendable_components: list[str] = field(default_factory=list)


@dataclass
class LanguageConfig:
    """Bounded, serialized text generation under a planned local hold."""

    enabled: bool = False
    motion_during_query: str = "hold"
    deadline_s: float = 60.0
    max_input_chars: int = 4096
    max_output_chars: int = 8192


@dataclass
class ServerConfig:
    """One named, explicitly described model deployment."""

    deployment: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    zenoh: ZenohConfig = field(default_factory=ZenohConfig)
    # Explicit feature conventions are required when checkpoint metadata is incomplete.
    semantics: str = ""
    robot_type: str = ""
    features: list[FeatureSpec] = field(default_factory=list)
    action_feature: FeatureSpec | None = None
    # INFO summarizes operation; DEBUG includes request-level diagnostics.
    log_level: Literal["INFO", "DEBUG"] = "INFO"

    def __post_init__(self) -> None:
        """Reject incomplete semantics and unsupported serving behavior before loading."""
        if self.log_level not in {"INFO", "DEBUG"}:
            raise ValueError("Server log_level must be INFO or DEBUG")
        if not self.deployment or not self.model.repo_or_path or not self.semantics:
            raise ValueError("Deployment, model.repo_or_path and explicit semantics are required")
        if not self.features or self.action_feature is None:
            raise ValueError("Explicit canonical features and action_feature are required")
        validate_blendable_components(self.action_feature, self.execution.blendable_components)
        if self.execution.blendable_components and "chunk" not in self.execution.supported_modes:
            raise ValueError("blendable_components requires the chunk execution mode")
        if self.language.motion_during_query != "hold":
            raise ValueError("Only planned holds during language generation are supported")
        for value in (
            self.execution.action_fps,
            self.execution.action_deadline_s,
            self.execution.idle_timeout_s,
            self.language.deadline_s,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Serving rates and deadlines must be finite and positive")
        if self.execution.warmup_calls < 1:
            raise ValueError("At least one warmup call is required before advertising readiness")
