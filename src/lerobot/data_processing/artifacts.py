# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Immutable artifacts; a checked manifest, not directory rename, commits remote outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, BinaryIO

from lerobot.streaming.location import StorageLocation
from lerobot.utils.import_utils import _fsspec_available, require_package

from .types import Artifact, canonical_json

if TYPE_CHECKING or _fsspec_available:
    import fsspec


class ArtifactStore:
    """One writer per immutable key; remote stores need no rename or transaction API.

    Concurrent controllers for the same plan are unsupported. Worker attempts have
    unique keys; only the controller publishes the stage's accepted index.
    """

    def __init__(self, uri: str | Path, *, storage_options: dict | None = None):
        require_package("fsspec", "dataset")
        self.uri = str(uri).rstrip("/")
        self.location = StorageLocation.parse(uri)
        self.storage_options = storage_options or {}
        self.fs, self.root = fsspec.core.url_to_fs(self.uri, **self.storage_options)
        protocols = self.fs.protocol if isinstance(self.fs.protocol, tuple) else (self.fs.protocol,)
        self.is_local = bool({"file", "local"}.intersection(protocols))

    def path(self, relative: str) -> str:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in relative:
            raise ValueError(f"Unsafe artifact path: {relative!r}")
        return f"{self.root.rstrip('/')}/{path}"

    def exists(self, relative: str) -> bool:
        return self.fs.exists(self.path(relative))

    def open(self, relative: str) -> BinaryIO:
        return self.fs.open(self.path(relative), "rb")

    def read_json(self, relative: str):
        with self.open(relative) as stream:
            return json.load(stream)

    def put_file(self, relative: str, source: Path, *, name: str = "", rows: int | None = None) -> Artifact:
        """Close and hash a file before publishing it; local outputs use an atomic link."""
        digest, size = file_checksum(source)
        path = self.path(relative)
        if self.fs.exists(path):
            raise FileExistsError(path)
        self.fs.makedirs(str(PurePosixPath(path).parent), exist_ok=True)
        if self.is_local:
            descriptor, temporary = tempfile.mkstemp(prefix=".processing-", dir=str(Path(path).parent))
            try:
                with os.fdopen(descriptor, "wb") as destination, source.open("rb") as stream:
                    shutil.copyfileobj(stream, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                # Unlike replace(), link() cannot clobber an already published key.
                os.link(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        else:
            # No remote atomic rename assumed. Unique attempts plus manifest-last
            # publication prevent consumers from accepting a partial upload.
            with source.open("rb") as stream, self.fs.open(path, "wb") as destination:
                shutil.copyfileobj(stream, destination)
        if self.checksum(relative) != (digest, size):
            raise OSError(f"Incomplete artifact upload: {relative}")
        return Artifact(relative, digest, size, name, rows)

    def put_json(self, relative: str, value) -> Artifact:
        with tempfile.TemporaryDirectory(prefix="lerobot-control-") as directory:
            source = Path(directory) / "record.json"
            source.write_bytes(canonical_json(value))
            return self.put_file(relative, source)

    def checksum(self, relative: str) -> tuple[str, int]:
        with self.open(relative) as stream:
            return _stream_checksum(stream)

    def download(self, artifact: Artifact, destination: Path) -> Path:
        """Copy and hash in one pass; never expose a partial or corrupt destination."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".processing-download-", dir=destination.parent) as directory:
            temporary = Path(directory) / "artifact"
            with self.open(artifact.path) as source, temporary.open("wb") as output:
                if _stream_checksum(source, output) != (artifact.sha256, artifact.size):
                    raise ValueError("Artifact download checksum mismatch")
            temporary.replace(destination)
        return destination

    def verify(self, artifact: Artifact) -> bool:
        try:
            return self.checksum(artifact.path) == (artifact.sha256, artifact.size)
        except (FileNotFoundError, OSError):
            return False

    def list(self, pattern: str) -> list[str]:
        prefix = self.root.rstrip("/") + "/"
        return sorted(path.removeprefix(prefix) for path in self.fs.glob(self.path(pattern)))


def file_checksum(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        return _stream_checksum(stream)


def _stream_checksum(stream: BinaryIO, output: BinaryIO | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while block := stream.read(1024 * 1024):
        digest.update(block)
        size += len(block)
        if output is not None:
            output.write(block)
    return digest.hexdigest(), size
