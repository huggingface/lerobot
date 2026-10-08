# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Small, serializable contracts for offline processing (not online robot processors)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import pyarrow as pa

    from .worker import WorkerContext


def canonical_json(value: Any) -> bytes:
    """Serialize semantic configuration reproducibly; reject NaN and non-JSON values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def checked_name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value):
        raise ValueError(f"Expected a path-safe identifier, got {value!r}")
    return value


class Outcome(StrEnum):
    COMPLETED = "completed"
    MASKED = "masked"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class DatasetRef:
    """An immutable Hub commit or a content-hashed local/raw input manifest."""

    uri: str
    revision: str

    def __post_init__(self):
        if not self.uri or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.revision):
            raise ValueError("DatasetRef requires a URI and a commit SHA or SHA256 input-manifest digest")


@dataclass(frozen=True)
class Resources:
    cpus: int = 1
    gpus: int = 0
    memory_gb: float = 4

    def __post_init__(self):
        if self.cpus < 1 or self.gpus < 0 or not math.isfinite(self.memory_gb) or self.memory_gb <= 0:
            raise ValueError("Invalid CPU/GPU/memory request")


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    version: str
    scope: str
    outputs: dict[str, pa.Schema | None]
    resources: Resources = field(default_factory=Resources)
    hf_jobs_compatible: bool = False

    def __post_init__(self):
        checked_name(self.name)
        if not self.version or self.scope not in {"asset", "episode", "camera_stream", "window", "reduction"}:
            raise ValueError("A module needs a version and a supported scope")
        for name in self.outputs:
            checked_name(name)


@dataclass(frozen=True)
class InputItem:
    """Source-defined semantic key and JSON metadata; never decoded media or tensors."""

    key: str
    payload: dict[str, Any]
    cost: float = 1

    def __post_init__(self):
        if not self.key or not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("Input items need a nonempty key and finite nonnegative cost")
        canonical_json(self.payload)


@dataclass(frozen=True)
class WorkItem:
    item_id: str
    key: str
    payload: dict[str, Any]
    cost: float
    seed: int


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str
    size: int
    name: str = ""
    rows: int | None = None


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    outcome: Outcome
    artifacts: tuple[Artifact, ...] = ()
    reason: str | None = None

    def __post_init__(self):
        if self.outcome != Outcome.COMPLETED and not self.reason:
            raise ValueError("Non-completed outcomes require a reason")
        if self.outcome == Outcome.FAILED and self.artifacts:
            raise ValueError("Failed work cannot publish artifacts")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ItemResult:
        return cls(
            item_id=value["item_id"],
            outcome=Outcome(value["outcome"]),
            artifacts=tuple(Artifact(**a) for a in value["artifacts"]),
            reason=value.get("reason"),
        )


class ProcessingModule(Protocol):
    spec: ModuleSpec

    def setup(self, context: WorkerContext) -> None: ...

    def process_batch(self, items: list[WorkItem], context: WorkerContext) -> list[ItemResult]: ...

    def teardown(self) -> None: ...
