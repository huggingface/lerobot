# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Recoverable compatibility commits for a user-owned local dataset (not a Hub cache)."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from filelock import FileLock

from ..artifacts import file_checksum


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".journal-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
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
            continue
        current = file_checksum(target) if target.exists() else None
        previous = tuple(entry["previous"]) if entry["previous"] is not None else None
        if current != previous:
            raise RuntimeError(f"Dataset changed during commit: {entry['relative']}")
        prepared = Path(entry["prepared"])
        if file_checksum(prepared) != desired:
            raise ValueError("Prepared commit file is corrupt")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".commit-", dir=target.parent)
        os.close(descriptor)
        try:
            shutil.copyfile(prepared, temporary)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    journal_path.unlink()


def recover_local_commit(root: Path) -> None:
    """Finish a interrupted commit before fingerprinting or reading new inputs."""
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
        prepared_root.mkdir(parents=True, exist_ok=True)
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
        atomic_json(root / ".processing_commit.json", {"files": entries})
        _recover_owned(root)
