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
"""Thin wrapper around the W&B operations this integration needs: uploading a local directory as
a versioned Artifact, and downloading a named Artifact into a local directory.

Every call returns a :class:`MaterializedArtifact` carrying both the requested and the resolved
(immutable) reference — see the "requested ref" / "resolved ref" entries in ``CONTEXT.md`` — never
only one or the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb

from .refs import ArtifactRef, parse_artifact_ref


class ArtifactTypeMismatchError(ValueError):
    """A fetched artifact's declared type doesn't match what the caller expected."""


@dataclass(frozen=True, slots=True)
class MaterializedArtifact:
    """The result of uploading or downloading a W&B Artifact.

    ``requested_ref`` is what the caller asked for (a collection path for an upload, or the
    possibly-mutable-alias reference string for a download). ``resolved_ref`` is always the
    immutable ``entity/project/name:vN`` W&B actually resolved to. ``local_path`` is the directory
    that now holds (or, for an upload, already held) the artifact's contents on disk — the source
    directory for an upload, the download destination for a download.
    """

    requested_ref: str
    resolved_ref: str
    local_path: Path
    version: str
    digest: str
    metadata: dict[str, Any]


def upload_directory(
    run: wandb.sdk.wandb_run.Run,
    directory: Path | str,
    *,
    name: str,
    artifact_type: str,
    aliases: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> MaterializedArtifact:
    """Upload ``directory`` as a new version of the ``name`` Artifact collection.

    Waits for W&B to confirm the upload is fully committed before returning, so a caller that
    gets a result back knows the artifact genuinely exists in W&B, not just that local files were
    queued for upload.
    """
    directory = Path(directory)
    requested_ref = f"{run.entity}/{run.project}/{name}"

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=dict(metadata) if metadata else None)
    artifact.add_dir(str(directory))

    logged = run.log_artifact(artifact, aliases=list(aliases) if aliases else None)
    logged.wait()

    return MaterializedArtifact(
        requested_ref=requested_ref,
        resolved_ref=logged.qualified_name,
        local_path=directory,
        version=logged.version,
        digest=logged.digest,
        metadata=dict(logged.metadata or {}),
    )


def download_artifact(
    run: wandb.sdk.wandb_run.Run,
    ref: str | ArtifactRef,
    *,
    expected_type: str,
    download_root: Path | str,
) -> MaterializedArtifact:
    """Declare ``ref`` as an input to ``run`` and download it into ``download_root``.

    Declaring it as a run input (rather than fetching it out-of-band via the public API) is what
    gives the run lineage back to the exact artifact version it consumed. Never deletes or
    overwrites anything already at ``download_root`` outside of what the artifact's own manifest
    writes.

    Raises:
        ArtifactTypeMismatchError: the fetched artifact's declared type isn't ``expected_type``.
            Raised before any download happens.
    """
    parsed = ref if isinstance(ref, ArtifactRef) else parse_artifact_ref(ref)
    download_root = Path(download_root)

    artifact = run.use_artifact(str(parsed))
    if artifact.type != expected_type:
        raise ArtifactTypeMismatchError(
            f"Expected an artifact of type {expected_type!r} but {parsed} is of type {artifact.type!r}."
        )

    local_path = artifact.download(root=str(download_root))

    return MaterializedArtifact(
        requested_ref=str(parsed),
        resolved_ref=artifact.qualified_name,
        local_path=Path(local_path),
        version=artifact.version,
        digest=artifact.digest,
        metadata=dict(artifact.metadata or {}),
    )
