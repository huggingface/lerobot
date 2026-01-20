#!/usr/bin/env python

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
"""Manifest schema for PolicyPackage v1.0.

The manifest is the contract between export and runtime. It is pure JSON data—no code references.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

MANIFEST_FORMAT = "policy_package"
MANIFEST_VERSION = "1.0"


class NormalizationType(str, Enum):
    """Normalization type for stats."""

    STANDARD = "standard"
    MIN_MAX = "min_max"
    QUANTILES = "quantiles"
    QUANTILE10 = "quantile10"
    IDENTITY = "identity"


@dataclass
class TensorSpec:
    """Specification for an input or output tensor."""

    name: str
    dtype: str
    shape: list[str | int]
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
        }
        if self.description:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorSpec:
        return cls(
            name=data["name"],
            dtype=data["dtype"],
            shape=data["shape"],
            description=data.get("description"),
        )


@dataclass
class PolicySource:
    """Source information for policy provenance."""

    repo_id: str | None = None
    revision: str | None = None
    commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {}
        if self.repo_id:
            result["repo_id"] = self.repo_id
        if self.revision:
            result["revision"] = self.revision
        if self.commit:
            result["commit"] = self.commit
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySource:
        return cls(
            repo_id=data.get("repo_id"),
            revision=data.get("revision"),
            commit=data.get("commit"),
        )


@dataclass
class PolicyInfo:
    """Policy metadata."""

    name: str
    source: PolicySource | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
        }
        if self.source:
            result["source"] = self.source.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyInfo:
        return cls(
            name=data["name"],
            source=PolicySource.from_dict(data["source"]) if "source" in data else None,
        )


@dataclass
class IOSpec:
    """Input/output specifications."""

    inputs: list[TensorSpec]
    outputs: list[TensorSpec]

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IOSpec:
        return cls(
            inputs=[TensorSpec.from_dict(t) for t in data["inputs"]],
            outputs=[TensorSpec.from_dict(t) for t in data["outputs"]],
        )


@dataclass
class ActionSpec:
    """Action semantics specification."""

    dim: int
    chunk_size: int
    n_action_steps: int
    representation: str = "absolute"
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "dim": self.dim,
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "representation": self.representation,
        }
        if self.description:
            result["description"] = self.description
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionSpec:
        return cls(
            dim=data["dim"],
            chunk_size=data["chunk_size"],
            n_action_steps=data["n_action_steps"],
            representation=data.get("representation", "absolute"),
            description=data.get("description"),
        )


@dataclass
class IterativeConfig:
    """Configuration for iterative policies (flow matching, diffusion)."""

    num_steps: int = 10
    scheduler: str = "euler"
    timestep_spacing: str = "linear"
    timestep_range: list[float] = field(default_factory=lambda: [1.0, 0.0])
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        result = {
            "num_steps": self.num_steps,
            "scheduler": self.scheduler,
            "timestep_spacing": self.timestep_spacing,
            "timestep_range": self.timestep_range,
        }
        if self.scheduler in ("ddpm", "ddim"):
            result["num_train_timesteps"] = self.num_train_timesteps
            result["beta_start"] = self.beta_start
            result["beta_end"] = self.beta_end
            result["beta_schedule"] = self.beta_schedule
            result["prediction_type"] = self.prediction_type
            result["clip_sample"] = self.clip_sample
            result["clip_sample_range"] = self.clip_sample_range
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IterativeConfig:
        return cls(
            num_steps=data.get("num_steps", 10),
            scheduler=data.get("scheduler", "euler"),
            timestep_spacing=data.get("timestep_spacing", "linear"),
            timestep_range=data.get("timestep_range", [1.0, 0.0]),
            num_train_timesteps=data.get("num_train_timesteps", 1000),
            beta_start=data.get("beta_start", 0.0001),
            beta_end=data.get("beta_end", 0.02),
            beta_schedule=data.get("beta_schedule", "squaredcos_cap_v2"),
            prediction_type=data.get("prediction_type", "epsilon"),
            clip_sample=data.get("clip_sample", True),
            clip_sample_range=data.get("clip_sample_range", 1.0),
        )


@dataclass
class TwoPhaseConfig:
    """Configuration for two-phase VLA policies (PI0, SmolVLA)."""

    num_steps: int = 10
    encoder_artifact: str = "onnx_encoder"
    denoise_artifact: str = "onnx_denoise"
    num_layers: int = 18
    num_kv_heads: int = 8
    head_dim: int = 256
    input_mapping: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "num_steps": self.num_steps,
            "encoder_artifact": self.encoder_artifact,
            "denoise_artifact": self.denoise_artifact,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }
        if self.input_mapping:
            result["input_mapping"] = self.input_mapping
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TwoPhaseConfig:
        return cls(
            num_steps=data.get("num_steps", 10),
            encoder_artifact=data.get("encoder_artifact", "onnx_encoder"),
            denoise_artifact=data.get("denoise_artifact", "onnx_denoise"),
            num_layers=data.get("num_layers", 18),
            num_kv_heads=data.get("num_kv_heads", 8),
            head_dim=data.get("head_dim", 256),
            input_mapping=data.get("input_mapping", {}),
        )


# Discriminated union type for inference configs
InferenceConfig = IterativeConfig | TwoPhaseConfig


def inference_config_from_dict(data: dict[str, Any]) -> InferenceConfig:
    """Parse inference config from dict, inferring type from structure.

    - Has `encoder_artifact` -> TwoPhaseConfig
    - Has `scheduler` -> IterativeConfig
    """
    if "encoder_artifact" in data:
        return TwoPhaseConfig.from_dict(data)
    elif "scheduler" in data:
        return IterativeConfig.from_dict(data)
    else:
        raise ValueError(
            "Cannot determine inference config type. "
            "Expected 'encoder_artifact' (TwoPhaseConfig) or 'scheduler' (IterativeConfig)."
        )


def inference_config_to_dict(config: InferenceConfig) -> dict[str, Any]:
    """Convert inference config to dict."""
    return config.to_dict()


def is_two_phase(config: InferenceConfig | None) -> bool:
    """Check if config is TwoPhaseConfig."""
    return isinstance(config, TwoPhaseConfig)


def is_iterative(config: InferenceConfig | None) -> bool:
    """Check if config is IterativeConfig."""
    return isinstance(config, IterativeConfig)


@dataclass
class NormalizationConfig:
    """Normalization configuration and stats location."""

    type: NormalizationType
    artifact: str
    input_features: list[str]
    output_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "artifact": self.artifact,
            "input_features": self.input_features,
            "output_features": self.output_features,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizationConfig:
        return cls(
            type=NormalizationType(data["type"]),
            artifact=data["artifact"],
            input_features=data["input_features"],
            output_features=data["output_features"],
        )


@dataclass
class ExportMetadata:
    """Export metadata (timestamps, versions)."""

    created_at: str | None = None
    created_by: str = "lerobot.export"
    lerobot_version: str | None = None
    export_device: str = "cpu"
    export_dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
            "created_by": self.created_by,
            "lerobot_version": self.lerobot_version,
            "export_device": self.export_device,
            "export_dtype": self.export_dtype,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportMetadata:
        return cls(
            created_at=data.get("created_at"),
            created_by=data.get("created_by", "lerobot.export"),
            lerobot_version=data.get("lerobot_version"),
            export_device=data.get("export_device", "cpu"),
            export_dtype=data.get("export_dtype", "float32"),
        )


@dataclass
class Manifest:
    """PolicyPackage manifest schema v1.0.

    The manifest is the contract between export and runtime. It is pure JSON data—no code references.

    The inference pattern is determined by the structure of the `inference` field:
    - None -> single-pass (ACT, Groot)
    - IterativeConfig -> iterative (Diffusion)
    - TwoPhaseConfig -> two-phase (PI0, SmolVLA)
    """

    policy: PolicyInfo
    artifacts: dict[str, str]
    io: IOSpec
    action: ActionSpec
    format: str = MANIFEST_FORMAT
    version: str = MANIFEST_VERSION
    inference: InferenceConfig | None = None
    normalization: NormalizationConfig | None = None
    metadata: ExportMetadata | None = None

    def __post_init__(self):
        """Validate manifest after construction."""
        self.validate()

    @property
    def is_single_pass(self) -> bool:
        """Check if this is a single-pass policy (no inference loop)."""
        return self.inference is None

    @property
    def is_iterative(self) -> bool:
        """Check if this is an iterative policy (Diffusion)."""
        return is_iterative(self.inference)

    @property
    def is_two_phase(self) -> bool:
        """Check if this is a two-phase policy (PI0, SmolVLA)."""
        return is_two_phase(self.inference)

    def validate(self) -> None:
        """Validate the manifest schema.

        Raises:
            ValueError: If validation fails.
        """
        if self.format != MANIFEST_FORMAT:
            raise ValueError(f"Invalid format: {self.format}, expected {MANIFEST_FORMAT}")

        if not self.version.startswith("1."):
            raise ValueError(f"Unsupported version: {self.version}, expected 1.x")

        if not self.artifacts:
            raise ValueError("At least one artifact is required")

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to a dictionary for JSON serialization."""
        result = {
            "format": self.format,
            "version": self.version,
            "policy": self.policy.to_dict(),
            "artifacts": self.artifacts,
            "io": self.io.to_dict(),
            "action": self.action.to_dict(),
        }
        if self.inference:
            result["inference"] = inference_config_to_dict(self.inference)
        if self.normalization:
            result["normalization"] = self.normalization.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Create a manifest from a dictionary (e.g., parsed JSON)."""
        inference: InferenceConfig | None = None
        if "inference" in data:
            inference = inference_config_from_dict(data["inference"])

        return cls(
            format=data.get("format", MANIFEST_FORMAT),
            version=data.get("version", MANIFEST_VERSION),
            policy=PolicyInfo.from_dict(data["policy"]),
            artifacts=data["artifacts"],
            io=IOSpec.from_dict(data["io"]),
            action=ActionSpec.from_dict(data["action"]),
            inference=inference,
            normalization=NormalizationConfig.from_dict(data["normalization"])
            if "normalization" in data
            else None,
            metadata=ExportMetadata.from_dict(data["metadata"]) if "metadata" in data else None,
        )

    def save(self, path: Path | str) -> None:
        """Save manifest to a JSON file.

        Args:
            path: Path to save the manifest to.
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: Path | str) -> Manifest:
        """Load manifest from a JSON file.

        Args:
            path: Path to load the manifest from.

        Returns:
            Loaded manifest instance.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
