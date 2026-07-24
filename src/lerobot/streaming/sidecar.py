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
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout

from lerobot.streaming.manifest import EpisodeVideoManifest

SIDECAR_SCHEMA_VERSION = 2


class SidecarLockTimeoutError(TimeoutError):
    """Raised when another process does not finish a sidecar build in time."""


@dataclass(frozen=True)
class SidecarSpec:
    repo_id: str
    revision: str
    data_root: str
    source_files: tuple[tuple[str, int | None], ...]
    schema_version: int = SIDECAR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.repo_id:
            raise ValueError("repo_id must not be empty")
        if not self.revision:
            raise ValueError("revision must not be empty")
        normalized = tuple(
            sorted((str(path), None if size is None else int(size)) for path, size in self.source_files)
        )
        if any(not path or size is not None and size < 0 for path, size in normalized):
            raise ValueError("source file paths must be non-empty and sizes must be non-negative")
        object.__setattr__(self, "source_files", normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "data_root": self.data_root,
            "source_files": [{"path": path, "size": size} for path, size in self.source_files],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SidecarSpec:
        source_files = data.get("source_files")
        if not isinstance(source_files, list):
            raise ValueError("MP4 sidecar source_files must be a list")
        parsed_files: list[tuple[str, int | None]] = []
        for item in source_files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("Invalid MP4 sidecar source file entry")
            size = item.get("size")
            parsed_files.append((item["path"], None if size is None else int(size)))
        return cls(
            schema_version=int(data["schema_version"]),
            repo_id=str(data["repo_id"]),
            revision=str(data["revision"]),
            data_root=str(data["data_root"]),
            source_files=tuple(parsed_files),
        )

    def with_source_files(self, source_files: tuple[tuple[str, int], ...]) -> SidecarSpec:
        return SidecarSpec(
            repo_id=self.repo_id,
            revision=self.revision,
            data_root=self.data_root,
            source_files=source_files,
            schema_version=self.schema_version,
        )

    def matches(self, candidate: SidecarSpec) -> bool:
        if (
            self.schema_version != candidate.schema_version
            or self.repo_id != candidate.repo_id
            or self.revision != candidate.revision
            or self.data_root != candidate.data_root
        ):
            return False
        expected = dict(self.source_files)
        actual = dict(candidate.source_files)
        if expected.keys() != actual.keys():
            return False
        return all(size is None or actual[path] == size for path, size in expected.items())


SidecarBuilder = Callable[[Path, SidecarSpec], None]
SidecarDownloader = Callable[[Path, SidecarSpec], bool]


def sidecar_cache_path(cache_root: str | Path, spec: SidecarSpec) -> Path:
    identity = json.dumps(
        {
            "schema_version": spec.schema_version,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "data_root": spec.data_root,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    repo_slug = re.sub(r"[^A-Za-z0-9_.-]+", "--", spec.repo_id).strip("-") or "dataset"
    revision_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", spec.revision).strip("-")[:32] or "revision"
    return (
        Path(cache_root).expanduser() / repo_slug / f"mp4-v{spec.schema_version}-{revision_slug}-{digest}.npz"
    )


def ensure_mp4_sidecar(
    spec: SidecarSpec,
    cache_root: str | Path,
    *,
    build: SidecarBuilder,
    download: SidecarDownloader | None = None,
    lock_timeout_s: float = 30 * 60,
) -> Path:
    """Return a valid local sidecar, downloading or building it exactly once.

    This function never uploads. ``download`` and ``build`` must write only to the temporary path
    provided to them; a validated file becomes visible at the cache path through ``os.replace``.
    """

    destination = sidecar_cache_path(cache_root, spec)
    if EpisodeVideoManifest.validate_file_sidecar(destination, spec):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(f"{destination.suffix}.lock")
    try:
        with FileLock(lock_path, timeout=lock_timeout_s):
            if EpisodeVideoManifest.validate_file_sidecar(destination, spec):
                return destination

            temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp.npz"
            try:
                if download is not None:
                    logging.info("Looking for published MP4 sidecar for %s@%s", spec.repo_id, spec.revision)
                    if download(temporary, spec) and EpisodeVideoManifest.validate_file_sidecar(
                        temporary, spec
                    ):
                        os.replace(temporary, destination)
                        return destination
                    temporary.unlink(missing_ok=True)

                logging.info("Building MP4 sidecar for %s@%s", spec.repo_id, spec.revision)
                build(temporary, spec)
                if not EpisodeVideoManifest.validate_file_sidecar(temporary, spec):
                    raise ValueError("Built MP4 sidecar failed revision and source validation")
                os.replace(temporary, destination)
                return destination
            finally:
                temporary.unlink(missing_ok=True)
    except Timeout as exc:
        raise SidecarLockTimeoutError(
            f"Timed out waiting {lock_timeout_s:g}s for MP4 sidecar lock {lock_path}"
        ) from exc
