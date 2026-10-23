# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Recoverable annotation updates for a user-owned local dataset (not a Hub cache)."""

import json
import os
import re
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path

from filelock import FileLock

from .runtime.artifacts import file_checksum


def _sync_directory(path: Path) -> None:
    """Persist POSIX directory entries; Windows has no portable directory fsync."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _sync_directory(directory.parent)


def atomic_json(path: Path, value) -> None:
    _mkdir_durable(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=".journal-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _recover_owned(root: Path) -> None:
    journal_path = root / ".processing_commit.json"
    if not journal_path.exists():
        return
    journal = json.loads(journal_path.read_text())
    for entry in journal["files"]:
        target = root / entry["relative"]
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("Commit journal points outside the dataset")
        desired = (entry["sha256"], entry["size"])
        if target.exists() and file_checksum(target) == desired:
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            _sync_directory(target.parent)
            continue
        current = file_checksum(target) if target.exists() else None
        previous = tuple(entry["previous"]) if entry["previous"] is not None else None
        if current != previous:
            raise RuntimeError(f"Dataset changed during commit: {entry['relative']}")
        prepared = Path(entry["prepared"])
        if file_checksum(prepared) != desired:
            raise ValueError("Prepared commit file is corrupt")
        _mkdir_durable(target.parent)
        descriptor, temporary = tempfile.mkstemp(prefix=".commit-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as destination, prepared.open("rb") as source:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, target)
            _sync_directory(target.parent)
        finally:
            Path(temporary).unlink(missing_ok=True)
    # Keep recovery files until every target is committed. Delete only files
    # created by this journal, including when prepared_root is caller-owned.
    for index, entry in enumerate(journal["files"]):
        prepared = Path(entry["prepared"])
        if prepared.name == f"{index:08d}-{entry['sha256']}":
            prepared.unlink(missing_ok=True)
    for directory in {Path(entry["prepared"]).parent for entry in journal["files"]}:
        if directory.parent.name == "prepared" and re.fullmatch(r"[0-9a-f]{64}", directory.name):
            with suppress(OSError):  # Preserve any caller files; never recursively remove directories.
                directory.rmdir()  # Only the now-empty, explicitly journal-owned plan directory.
    journal_path.unlink()
    _sync_directory(root)


def recover_local_commit(root: Path) -> None:
    """Finish an interrupted commit before fingerprinting or reading new inputs."""
    with FileLock(root / ".processing_commit.lock", timeout=0):
        _recover_owned(root)


def commit_local_files(root: Path, files: dict[str, Path], prepared_root: Path, *, expected=None) -> None:
    """Prepare every changed file first, then roll forward under an exclusive lock.

    Several path replacements are recoverable, not atomically visible to arbitrary
    concurrent readers. Use a separate version directory for concurrent readers.
    """
    root = root.resolve()
    with FileLock(root / ".processing_commit.lock", timeout=0):
        _recover_owned(root)
        for relative, checksum in (expected or {}).items():
            target = root / relative
            if not target.resolve().is_relative_to(root):
                raise ValueError("Unsafe expected input path")
            current = file_checksum(target) if target.exists() else None
            if current != checksum:
                raise RuntimeError(f"Dataset changed after planning: {relative}")
        _mkdir_durable(prepared_root)
        entries = []
        for index, (relative, source) in enumerate(sorted(files.items())):
            target = root / relative
            if not target.resolve().is_relative_to(root) or relative.startswith("."):
                raise ValueError("Unsafe dataset commit target")
            prepared = prepared_root / f"{index:08d}-{file_checksum(source)[0]}"
            if not prepared.exists():
                shutil.copyfile(source, prepared)
            with prepared.open("rb") as stream:
                os.fsync(stream.fileno())
            digest, size = file_checksum(prepared)
            if (digest, size) != file_checksum(source):
                raise ValueError("Prepared commit differs from accepted output")
            entries.append(
                {
                    "relative": relative,
                    "prepared": str(prepared.resolve()),
                    "sha256": digest,
                    "size": size,
                    "previous": file_checksum(target) if target.exists() else None,
                }
            )
        _sync_directory(prepared_root)
        atomic_json(root / ".processing_commit.json", {"files": entries})
        _recover_owned(root)
