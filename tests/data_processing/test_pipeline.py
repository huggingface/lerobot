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


def test_fresh_dependency_ids_do_not_depend_on_attempt_paths(tmp_path):
    pytest.importorskip("pyarrow", reason="Work plans require lerobot[dataset]")
    from lerobot.data_processing.configs import RuntimeConfig
    from lerobot.data_processing.pipeline import run_pipeline
    from lerobot.data_processing.types import DatasetRef, InputItem

    class Source:
        dataset_ref = DatasetRef("fixture/source", "a" * 40)

        def discover(self, stage, store, upstream):
            for index in range(6):
                yield InputItem(str(index), {"value": index})

    stages = [
        StageConfig("a", "tests.data_processing._modules:Echo"),
        StageConfig("b", "tests.data_processing._modules:Echo", depends_on=("a",)),
    ]
    identifiers = []
    for workers in (1, 2):
        store, completed = run_pipeline(
            Source(),
            stages,
            RuntimeConfig(
                run_uri=str(tmp_path / f"run-{workers}"),
                workers=workers,
                shard_size=workers,
                batch_size=workers,
            ),
        )
        identifiers.append(
            {
                name: [item.item_id for shard in range(plan.shards) for item in plan.read_shard(store, shard)]
                for name, (plan, _) in completed.items()
            }
        )
    assert identifiers[0]["a"] == identifiers[1]["a"]
    assert identifiers[0]["b"] == identifiers[1]["b"]


def test_unavailable_telemetry_does_not_block_processing(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow", reason="Work plans require lerobot[dataset]")
    from lerobot.data_processing.artifacts import ArtifactStore
    from lerobot.data_processing.configs import RuntimeConfig
    from lerobot.data_processing.pipeline import run_pipeline
    from lerobot.data_processing.types import DatasetRef, InputItem

    class Source:
        dataset_ref = DatasetRef("fixture/source", "a" * 40)

        def discover(self, stage, store, upstream):
            yield InputItem("0", {"value": 0})

    original = ArtifactStore.list

    def listing(self, pattern):
        if pattern.startswith("metrics/"):
            raise OSError("injected telemetry outage")
        return original(self, pattern)

    monkeypatch.setattr(ArtifactStore, "list", listing)
    _, completed = run_pipeline(
        Source(),
        [StageConfig("echo", "tests.data_processing._modules:Echo")],
        RuntimeConfig(run_uri=str(tmp_path / "run")),
    )
    assert completed["echo"][1].completed == 1
