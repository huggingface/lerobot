#!/usr/bin/env python

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
"""Manifest schema for the converged ``policy_package`` v1.0 format.

The manifest is the contract between export and runtime.  It is pure JSON
data.

LeRobot uses the ``type`` + flat-params style for components (runners,
preprocessors, postprocessors).

Schema overview::

    manifest.json
    ├── format + version          (envelope)
    ├── policy                    (identity — what policy is this?)
    │   ├── name
    │   └── source                (provenance: repo_id, class_path)
    ├── model                     (exported model — how to run it?)
    │   ├── n_obs_steps
    │   ├── runner                (execution pattern + parameters)
    │   ├── artifacts             (model files by named role)
    │   ├── preprocessors         (input transforms)
    │   └── postprocessors        (output transforms)
    ├── hardware                  (deployment — what hardware?)
    │   ├── robots
    │   └── cameras
    └── metadata                  (provenance — when/who created this?)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

MANIFEST_FORMAT = "policy_package"
MANIFEST_VERSION = "1.0"

__all__ = [
    "MANIFEST_FORMAT",
    "MANIFEST_VERSION",
    "CameraConfig",
    "HardwareConfig",
    "Manifest",
    "Metadata",
    "ModelConfig",
    "PolicyInfo",
    "PolicySource",
    "ProcessorSpec",
    "RobotConfig",
    "TensorSpec",
]


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value for JSON output."""
    if is_dataclass(value):
        return _to_dict(value)
    if isinstance(value, list):
        if not value:
            return None
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        if not value:
            return None
        return {key: _serialize_value(val) for key, val in value.items() if val is not None}
    return value


def _to_dict(instance: Any) -> dict[str, Any]:
    """Convert a dataclass instance to a dict, omitting ``None`` values."""
    result: dict[str, Any] = {}
    for field_info in fields(instance):
        value = getattr(instance, field_info.name)
        if value is None:
            continue
        serialized = _serialize_value(value)
        if serialized is None:
            continue
        result[field_info.name] = serialized
    return result


def _from_dict(
    cls: type[Any],
    data: dict[str, Any],
    converters: dict[str, Callable[[Any], Any]] | None = None,
) -> Any:
    """Instantiate a dataclass from a dict, applying optional *converters*."""
    values: dict[str, Any] = {}
    field_converters = converters or {}
    for field_info in fields(cls):
        if field_info.name in data:
            raw_value = data[field_info.name]
            if field_info.name in field_converters:
                values[field_info.name] = field_converters[field_info.name](raw_value)
            else:
                values[field_info.name] = raw_value
            continue
        if field_info.default is not MISSING or field_info.default_factory is not MISSING:
            continue
    return cls(**values)


@dataclass
class PolicySource:
    """Provenance information for the exported policy."""

    repo_id: str | None = None
    revision: str | None = None
    class_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict, omitting ``None`` fields."""
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySource:
        """Deserialise from a plain dict (e.g. parsed JSON).

        Args:
            data: Raw dict with optional keys ``repo_id``, ``revision``,
                ``class_path``.

        Returns:
            A new :class:`PolicySource` instance.
        """
        return _from_dict(cls, data)


@dataclass
class PolicyInfo:
    """Policy identity section."""

    name: str
    source: PolicySource | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict, omitting ``None`` fields."""
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyInfo:
        """Deserialise from a plain dict.

        Args:
            data: Raw dict with required key ``name`` and optional ``source``.

        Returns:
            A new :class:`PolicyInfo` instance.
        """
        return _from_dict(cls, data, converters={"source": PolicySource.from_dict})


@dataclass
class ProcessorSpec:
    """Specification for a preprocessor or postprocessor entry.

    Uses the ``type`` + flat-params format for interoperability::

        {"type": "normalize", "mode": "mean_std", "artifact": "stats.safetensors", "features": [...]}
    """

    type: str
    class_path: str | None = None
    mode: str | None = None
    artifact: str | None = None
    features: list[str] | None = None
    extra_params: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-compatible dict.

        Known fields are emitted first; any ``extra_params`` are merged in
        at the top level, matching the flat-params convention.

        Returns:
            Dict with at minimum a ``"type"`` key.
        """
        result: dict[str, object] = {"type": self.type}
        if self.class_path is not None:
            result["class_path"] = self.class_path
        if self.mode is not None:
            result["mode"] = self.mode
        if self.artifact is not None:
            result["artifact"] = self.artifact
        if self.features is not None:
            result["features"] = self.features
        for key, value in self.extra_params.items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessorSpec:
        """Deserialise from a flat dict, separating known fields from extras.

        Any keys not in ``{type, class_path, mode, artifact, features}`` are
        collected into ``extra_params`` so round-trip fidelity is preserved.

        Args:
            data: Raw dict with at minimum a ``"type"`` key.

        Returns:
            A new :class:`ProcessorSpec` instance.
        """
        known_fields = {"type", "class_path", "mode", "artifact", "features"}
        extra_params = {key: value for key, value in data.items() if key not in known_fields}
        return cls(
            type=data["type"],
            class_path=data.get("class_path"),
            mode=data.get("mode"),
            artifact=data.get("artifact"),
            features=data.get("features"),
            extra_params=extra_params,
        )


@dataclass
class ModelConfig:
    """Model configuration — how to run the exported policy.

    The ``runner`` field is an open-ended dict with a ``type`` key that
    determines the inference pattern.  Policy-specific parameters sit
    alongside ``type`` as flat keys.
    """

    n_obs_steps: int
    runner: dict[str, Any]
    artifacts: dict[str, str]
    preprocessors: list[ProcessorSpec] | None = None
    postprocessors: list[ProcessorSpec] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Preprocessors and postprocessors are omitted when empty.

        Returns:
            Dict with keys ``n_obs_steps``, ``runner``, ``artifacts``, and
            optionally ``preprocessors`` / ``postprocessors``.
        """
        result: dict[str, Any] = {
            "n_obs_steps": self.n_obs_steps,
            "runner": self.runner,
            "artifacts": self.artifacts,
        }
        if self.preprocessors:
            result["preprocessors"] = [p.to_dict() for p in self.preprocessors]
        if self.postprocessors:
            result["postprocessors"] = [p.to_dict() for p in self.postprocessors]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Deserialise from a plain dict, converting processor lists.

        Args:
            data: Raw dict with keys ``n_obs_steps``, ``runner``,
                ``artifacts``, and optionally ``preprocessors`` /
                ``postprocessors``.

        Returns:
            A new :class:`ModelConfig` instance.
        """
        return _from_dict(
            cls,
            data,
            converters={
                "preprocessors": lambda items: [ProcessorSpec.from_dict(item) for item in items],
                "postprocessors": lambda items: [ProcessorSpec.from_dict(item) for item in items],
            },
        )


@dataclass
class TensorSpec:
    """Shape and dtype specification for a hardware tensor (state/action)."""

    shape: list[int]
    dtype: str = "float32"
    order: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict, omitting ``None`` fields."""
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorSpec:
        """Deserialise from a plain dict.

        Args:
            data: Raw dict with required key ``shape`` and optional ``dtype``,
                ``order``.

        Returns:
            A new :class:`TensorSpec` instance.
        """
        return _from_dict(cls, data)


@dataclass
class RobotConfig:
    """Robot hardware declaration."""

    name: str
    type: str | None = None
    state: TensorSpec | None = None
    action: TensorSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict, omitting ``None`` fields."""
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotConfig:
        """Deserialise from a plain dict, converting nested tensor specs.

        Args:
            data: Raw dict with required key ``name`` and optional ``type``,
                ``state``, ``action``.

        Returns:
            A new :class:`RobotConfig` instance.
        """
        return _from_dict(
            cls,
            data,
            converters={
                "state": TensorSpec.from_dict,
                "action": TensorSpec.from_dict,
            },
        )


@dataclass
class CameraConfig:
    """Camera hardware declaration."""

    name: str
    shape: list[int] | None = None
    dtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict, omitting ``None`` fields."""
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraConfig:
        """Deserialise from a plain dict.

        Args:
            data: Raw dict with required key ``name`` and optional ``shape``,
                ``dtype``.

        Returns:
            A new :class:`CameraConfig` instance.
        """
        return _from_dict(cls, data)


@dataclass
class HardwareConfig:
    """Hardware section — what the policy expects at inference time."""

    robots: list[RobotConfig] | None = None
    cameras: list[CameraConfig] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict, omitting ``None`` fields."""
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareConfig:
        """Deserialise from a plain dict, converting nested robot/camera lists.

        Args:
            data: Raw dict with optional keys ``robots`` and ``cameras``.

        Returns:
            A new :class:`HardwareConfig` instance.
        """
        return _from_dict(
            cls,
            data,
            converters={
                "robots": lambda items: [RobotConfig.from_dict(item) for item in items],
                "cameras": lambda items: [CameraConfig.from_dict(item) for item in items],
            },
        )


@dataclass
class Metadata:
    """Export provenance metadata."""

    created_at: str | None = None
    created_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict, omitting ``None`` fields."""
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metadata:
        """Deserialise from a plain dict.

        Args:
            data: Raw dict with optional keys ``created_at``, ``created_by``.

        Returns:
            A new :class:`Metadata` instance.
        """
        return _from_dict(cls, data)


@dataclass
class Manifest:
    """Policy-package manifest v1.0.

    This is the converged schema shared by LeRobot and PhysicalAI.  The
    runner ``type`` determines the inference pattern:

    - ``single_pass`` — one forward pass that emits the full action chunk
      (e.g. ACT and other feedforward chunk-emitting policies)
    - ``iterative`` — legacy multi-step denoising / flow-matching manifests
    - ``kv_cache_flow`` — encode prefix once with KV cache, then iterate
      flow-matching Euler denoise steps (e.g. PI05)
    """

    policy: PolicyInfo
    model: ModelConfig
    format: str = MANIFEST_FORMAT
    version: str = MANIFEST_VERSION
    hardware: HardwareConfig | None = None
    metadata: Metadata | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def runner_type(self) -> str:
        """Return the runner type string (e.g. ``"single_pass"``)."""
        return self.model.runner.get("type", "single_pass")

    def validate(self) -> None:
        """Validate required fields.

        Raises:
            ValueError: If validation fails.
        """
        if self.format != MANIFEST_FORMAT:
            raise ValueError(f"Invalid format: {self.format!r}, expected {MANIFEST_FORMAT!r}")
        if not self.version.startswith("1."):
            raise ValueError(f"Unsupported version: {self.version!r}, expected 1.x")
        if not self.model.artifacts:
            raise ValueError("At least one artifact is required in model.artifacts")
        if "type" not in self.model.runner:
            raise ValueError("model.runner must contain a 'type' key")

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        result: dict[str, Any] = {
            "format": self.format,
            "version": self.version,
            "policy": self.policy.to_dict(),
            "model": self.model.to_dict(),
        }
        if self.hardware:
            result["hardware"] = self.hardware.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Create a manifest from a dict (e.g. parsed JSON)."""
        return _from_dict(
            cls,
            data,
            converters={
                "policy": PolicyInfo.from_dict,
                "model": ModelConfig.from_dict,
                "hardware": HardwareConfig.from_dict,
                "metadata": Metadata.from_dict,
            },
        )

    def save(self, path: Path | str) -> None:
        """Save manifest to a JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> Manifest:
        """Load manifest from a JSON file."""
        path = Path(path)
        try:
            with open(path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            raise ValueError(f"failed to parse manifest at {path}: {e}") from e
