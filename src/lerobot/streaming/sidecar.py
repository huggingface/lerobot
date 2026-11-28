# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Revision-safe lifecycle for locally cached MP4 byte-index sidecars."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout

from .manifest import EpisodeVideoManifest
from .sidecar_utils import (  # re-export; defined there to avoid a cycle
    SidecarSpec as SidecarSpec,
    install_sidecar,
)


class SidecarLockTimeoutError(TimeoutError):
    """Raised when another process does not finish a sidecar build in time."""


SidecarBuilder = Callable[[Path, SidecarSpec], None]
SidecarDownloader = Callable[[Path, SidecarSpec], bool]


def sidecar_file_prefix(spec: SidecarSpec) -> str:
    """File name prefix shared by every sidecar of one dataset revision."""
    revision_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", spec.revision).strip("-")[:32] or "revision"
    return f"mp4-v{spec.schema_version}-{revision_slug}-"


def sidecar_cache_path(cache_root: str | Path, spec: SidecarSpec, *, partial: bool = False) -> Path:
    """Derive a revision-keyed local cache path for a sidecar.

    A ``partial`` sidecar indexes a camera selection rather than every camera of the dataset;
    it gets its own name, keyed by its source files, so selections never overwrite each other
    or the full-dataset sidecar that serves them all.
    """
    identity = json.dumps(
        {
            "schema_version": spec.schema_version,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "data_root": spec.data_root,
            **({"source_fingerprints": dict(spec.source_fingerprints)} if spec.source_fingerprints else {}),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    if partial:
        sources = "\n".join(path for path, _size in spec.source_files)
        digest += f"-src{hashlib.sha256(sources.encode()).hexdigest()[:8]}"
    repo_slug = re.sub(r"[^A-Za-z0-9_.-]+", "--", spec.repo_id).strip("-") or "dataset"
    return Path(cache_root).expanduser() / repo_slug / f"{sidecar_file_prefix(spec)}{digest}.npz"


def _find_valid_sidecar(destination: Path, spec: SidecarSpec) -> Path | None:
    """Return ``destination`` or a cached sidecar of the same revision that satisfies ``spec``.

    Where cache names carry source fingerprints, a camera selection is named differently from
    the full-dataset sidecar that serves it, so siblings are checked as well.
    """
    if EpisodeVideoManifest.validate_file_sidecar(destination, spec):
        return destination
    for candidate in sorted(destination.parent.glob(f"{sidecar_file_prefix(spec)}*.npz")):
        if candidate != destination and EpisodeVideoManifest.validate_file_sidecar(candidate, spec):
            return candidate
    return None


def ensure_mp4_sidecar(
    spec: SidecarSpec,
    cache_root: str | Path,
    *,
    build: SidecarBuilder,
    download: SidecarDownloader | None = None,
    lock_timeout_s: float = 30 * 60,
    partial: bool = False,
) -> Path:
    """Return a valid local sidecar, downloading or building it exactly once.

    This function never uploads. ``download`` and ``build`` must write only to the temporary path
    provided to them; a validated file becomes visible at the cache path through ``os.replace``.
    ``partial`` marks a spec that covers a camera selection (see ``sidecar_cache_path``).
    """
    destination = sidecar_cache_path(cache_root, spec, partial=partial)
    if (cached := _find_valid_sidecar(destination, spec)) is not None:
        return cached

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(f"{destination.suffix}.lock")
    try:
        with FileLock(lock_path, timeout=lock_timeout_s):
            if (cached := _find_valid_sidecar(destination, spec)) is not None:
                return cached

            temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp.npz"
            try:
                if download is not None:
                    logging.info("Looking for published MP4 sidecar for %s@%s", spec.repo_id, spec.revision)
                    if download(temporary, spec) and EpisodeVideoManifest.validate_file_sidecar(
                        temporary, spec
                    ):
                        install_sidecar(temporary, destination)
                        return destination
                    temporary.unlink(missing_ok=True)

                logging.info("Building MP4 sidecar for %s@%s", spec.repo_id, spec.revision)
                build(temporary, spec)
                if not EpisodeVideoManifest.validate_file_sidecar(temporary, spec):
                    raise ValueError("Built MP4 sidecar failed revision and source validation")
                install_sidecar(temporary, destination)
                return destination
            finally:
                temporary.unlink(missing_ok=True)
    except Timeout as exc:
        raise SidecarLockTimeoutError(
            f"Timed out waiting {lock_timeout_s:g}s for MP4 sidecar lock {lock_path}"
        ) from exc
