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
"""Parsing and representation of W&B artifact references.

See ``CONTEXT.md`` at the repo root for the "requested ref" / "resolved ref"
distinction this integration relies on throughout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Exactly entity/project/name:version_or_alias — no more, no fewer components,
# no surrounding whitespace, no empty component. Deliberately rejects local
# paths (no ":") and "wandb://"-style values (the "wandb:" prefix can never
# satisfy the leading "no '/' or ':'" entity component).
_REF_PATTERN = re.compile(
    r"^(?P<entity>[^/:\s]+)/(?P<project>[^/:\s]+)/(?P<name>[^/:\s]+):(?P<version_or_alias>[^/:\s]+)$"
)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """An immutable pointer to a single version (or alias) of a W&B artifact."""

    entity: str
    project: str
    name: str
    version_or_alias: str

    def __str__(self) -> str:
        return f"{self.entity}/{self.project}/{self.name}:{self.version_or_alias}"


def parse_artifact_ref(raw: str) -> ArtifactRef:
    """Parse a ``entity/project/name:version`` or ``entity/project/name:alias`` string.

    Raises:
        ValueError: ``raw`` isn't a string, or doesn't match the required
            three-component-plus-version-or-alias shape (this rejects local
            paths, ``wandb://``-style values, and any missing/empty/
            whitespace-padded component).
    """
    if not isinstance(raw, str):
        raise ValueError(f"Artifact reference must be a string, got {type(raw).__name__}.")

    match = _REF_PATTERN.fullmatch(raw)
    if match is None:
        raise ValueError(
            f"Invalid W&B artifact reference {raw!r}. Expected the form "
            "'entity/project/name:version' or 'entity/project/name:alias'."
        )

    return ArtifactRef(**match.groupdict())
