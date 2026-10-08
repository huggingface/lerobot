# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import pytest

from lerobot.data_processing.configs import StageConfig
from lerobot.data_processing.pipeline import ordered_stages
from lerobot.data_processing.sinks.local_commit import commit_local_files, recover_local_commit


def test_graph_checks_and_order():
    a = StageConfig("a", "unused:Class")
    b = StageConfig("b", "unused:Class", depends_on=("a",))
    assert ordered_stages([b, a]) == [a, b]
    for graph in ([a, a], [b], [StageConfig("a", "unused:Class", depends_on=("a",))]):
        with pytest.raises(ValueError):
            ordered_stages(graph)


def test_commit_recovery_and_expected_inputs(tmp_path, monkeypatch):
    from lerobot.data_processing.sinks import local_commit

    root = tmp_path / "dataset"
    root.mkdir()
    sources = {}
    for name in ("a", "b"):
        (root / name).write_text("old")
        sources[name] = tmp_path / name
        sources[name].write_text("new")
    real_replace = local_commit.os.replace

    def interrupt(source, target):
        if target == root / "b":
            raise RuntimeError("power loss")
        real_replace(source, target)

    monkeypatch.setattr(local_commit.os, "replace", interrupt)
    with pytest.raises(RuntimeError, match="power loss"):
        commit_local_files(root, sources, tmp_path / "prepared")
    assert (root / "a").read_text() == "new"
    assert (root / "b").read_text() == "old"
    monkeypatch.setattr(local_commit.os, "replace", real_replace)
    recover_local_commit(root)
    assert (root / "b").read_text() == "new"
    assert not (root / ".processing_commit.json").exists()
    with pytest.raises(RuntimeError, match="after planning"):
        commit_local_files(root, sources, tmp_path / "prepared", expected={"a": None})
