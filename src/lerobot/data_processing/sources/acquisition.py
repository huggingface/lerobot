# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Pinned acquisition primitives for trusted single-file source recipes."""

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import fsspec
from filelock import FileLock

from ..artifacts import file_checksum


def acquire_file(uri: str, sha256: str, destination: Path) -> Path:
    """Download once, validate content, and atomically publish a local raw input."""
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise ValueError("Acquisition requires the expected SHA256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if destination.exists():
            if file_checksum(destination)[0] != sha256:
                raise ValueError("Existing raw input does not match its pinned checksum")
            return destination
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix="download-")
        os.close(descriptor)
        try:
            with fsspec.open(uri, "rb") as source, open(temporary, "wb") as output:
                shutil.copyfileobj(source, output)
            if file_checksum(Path(temporary))[0] != sha256:
                raise ValueError("Downloaded raw input checksum mismatch")
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return destination


def unpack_tar(archive: Path, destination: Path) -> None:
    """Extract regular files/directories only; never links, devices or traversal."""
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as stream:
        for member in stream:
            target = destination / member.name
            if not target.resolve().is_relative_to(destination.resolve()) or not (
                member.isfile() or member.isdir()
            ):
                raise ValueError("Unsafe archive member")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError("Archive file has no readable content")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
