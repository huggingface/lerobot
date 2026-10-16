# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import os

import pytest

from lerobot.annotations import storage
from lerobot.annotations.storage import commit_local_files, recover_local_commit


@pytest.mark.skipif(os.name == "nt", reason="Directory fsync is POSIX-only")
def test_install_fsyncs_file_then_rename_then_parent_before_removing_journal(tmp_path, monkeypatch):
    import stat

    root = tmp_path / "dataset"
    root.mkdir()
    source = tmp_path / "new"
    source.write_bytes(b"durable content")
    events = []
    real_sync, real_replace, real_unlink = os.fsync, os.replace, storage.Path.unlink

    def sync(fd):
        info = os.fstat(fd)
        events.append(("directory" if stat.S_ISDIR(info.st_mode) else "file", info.st_ino))
        real_sync(fd)

    def replace(src, dst):
        if dst == root / ".processing_commit.json":
            assert ("directory", (tmp_path / "prepared").stat().st_ino) in events
            events.append(("journal", root.stat().st_ino))
        if dst == root / "data/part.parquet":
            journal = next(i for i, event in enumerate(events) if event[0] == "journal")
            assert ("directory", root.stat().st_ino) in events[journal + 1 :]
            assert ("file", storage.Path(src).stat().st_ino) in events
            events.append(("install", storage.Path(dst).parent.stat().st_ino))
        real_replace(src, dst)

    def unlink(path, *args, **kwargs):
        if path == root / ".processing_commit.json":
            install = next(i for i, event in enumerate(events) if event[0] == "install")
            assert ("directory", (root / "data").stat().st_ino) in events[install + 1 :]
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(storage.os, "fsync", sync)
    monkeypatch.setattr(storage.os, "replace", replace)
    monkeypatch.setattr(storage.Path, "unlink", unlink)
    commit_local_files(root, {"data/part.parquet": source}, tmp_path / "prepared")
    assert (root / "data/part.parquet").read_bytes() == source.read_bytes()


def test_commit_recovery_and_expected_inputs(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    sources = {}
    for name in ("a", "b"):
        (root / name).write_text("old")
        sources[name] = tmp_path / name
        sources[name].write_text("new")
    real_replace = storage.os.replace

    def interrupt(source, target):
        if target == root / "b":
            raise RuntimeError("interrupted install")
        real_replace(source, target)

    monkeypatch.setattr(storage.os, "replace", interrupt)
    with pytest.raises(RuntimeError, match="interrupted install"):
        commit_local_files(root, sources, tmp_path / "prepared")
    assert (root / "a").read_text() == "new"
    assert (root / "b").read_text() == "old"
    monkeypatch.setattr(storage.os, "replace", real_replace)
    recover_local_commit(root)
    assert (root / "b").read_text() == "new"
    assert not (root / ".processing_commit.json").exists()
    assert not list((tmp_path / "prepared").iterdir())
    assert all(source.read_text() == "new" for source in sources.values())
    with pytest.raises(RuntimeError, match="after planning"):
        commit_local_files(root, sources, tmp_path / "prepared", expected={"a": None})


@pytest.mark.skipif(os.name == "nt", reason="Directory fsync is POSIX-only")
def test_directory_flush_failure_retains_recoverable_journal(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    (root / "data").mkdir(parents=True)
    target = root / "data/part.parquet"
    source = tmp_path / "new"
    source.write_bytes(b"new content")
    real_sync = os.fsync

    def fail_after_install(fd):
        if target.exists() and os.fstat(fd).st_ino == target.parent.stat().st_ino:
            raise OSError("injected directory flush failure")
        real_sync(fd)

    monkeypatch.setattr(storage.os, "fsync", fail_after_install)
    with pytest.raises(OSError, match="directory flush failure"):
        commit_local_files(root, {"data/part.parquet": source}, tmp_path / "prepared")
    assert (root / ".processing_commit.json").exists()
    assert list((tmp_path / "prepared").iterdir())
    monkeypatch.setattr(storage.os, "fsync", real_sync)
    recover_local_commit(root)
    assert target.read_bytes() == source.read_bytes()
    assert not (root / ".processing_commit.json").exists()
    assert not list((tmp_path / "prepared").iterdir())
