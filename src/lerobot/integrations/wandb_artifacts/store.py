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

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb

from .refs import ArtifactRef, parse_artifact_ref


class ArtifactTypeMismatchError(ValueError):
    """A fetched artifact's declared type doesn't match what the caller expected."""


class DownloadDestinationNotEmptyError(ValueError):
    """A download destination contains content that must not be overwritten."""


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
    validator: Callable[[Path], object] | None = None,
) -> MaterializedArtifact:
    """Declare ``ref`` as a run input and transactionally materialize it at ``download_root``.

    The artifact is downloaded into a temporary sibling directory. If supplied, ``validator`` runs
    against that staged directory. Only a complete, valid result is atomically promoted to the final
    destination, so interrupted downloads and validation failures never poison the caller-visible
    path. A pre-existing empty directory is accepted; any file, symlink, or nonempty directory is
    rejected and never modified.

    Raises:
        DownloadDestinationNotEmptyError: ``download_root`` is not absent or an empty directory.
        ArtifactTypeMismatchError: the fetched artifact's declared type isn't ``expected_type``.
            Raised before any download happens.
        Exception: Any download or validation failure, after cleaning up the private staging path.
    """
    parsed = ref if isinstance(ref, ArtifactRef) else parse_artifact_ref(ref)
    download_root = Path(download_root)

    destination_was_empty = False
    if download_root.exists() or download_root.is_symlink():
        if download_root.is_symlink() or not download_root.is_dir() or any(download_root.iterdir()):
            raise DownloadDestinationNotEmptyError(
                f"{download_root} already contains or points to local content. Point at an empty "
                "directory or a path that does not exist."
            )
        destination_was_empty = True

    artifact = run.use_artifact(str(parsed))
    if artifact.type != expected_type:
        raise ArtifactTypeMismatchError(
            f"Expected an artifact of type {expected_type!r} but {parsed} is of type {artifact.type!r}."
        )

    download_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{download_root.name}.download-", dir=download_root.parent)
    )
    removed_empty_destination = False

    try:
        staged_path = Path(artifact.download(root=str(staging_root)))
        if validator is not None:
            validator(staged_path)

        if destination_was_empty:
            # rmdir is deliberately race-safe: it fails rather than removing newly-created content.
            download_root.rmdir()
            removed_empty_destination = True

        os.replace(staged_path, download_root)
    except Exception:
        if removed_empty_destination and not download_root.exists():
            download_root.mkdir()
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    return MaterializedArtifact(
        requested_ref=str(parsed),
        resolved_ref=artifact.qualified_name,
        local_path=download_root,
        version=artifact.version,
        digest=artifact.digest,
        metadata=dict(artifact.metadata or {}),
    )
