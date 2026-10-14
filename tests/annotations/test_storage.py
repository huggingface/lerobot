# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import pytest

from lerobot.annotations import storage
from lerobot.annotations.storage import commit_local_files, recover_local_commit


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
    with pytest.raises(RuntimeError, match="after planning"):
        commit_local_files(root, sources, tmp_path / "prepared", expected={"a": None})
