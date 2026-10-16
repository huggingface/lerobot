# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import pytest

from lerobot.annotations.runtime.configs import StageConfig
from lerobot.annotations.runtime.pipeline import ordered_stages


def test_graph_checks_and_order():
    a = StageConfig("a", "unused:Class")
    b = StageConfig("b", "unused:Class", depends_on=("a",))
    assert ordered_stages([b, a]) == [a, b]
    for graph in ([a, a], [b], [StageConfig("a", "unused:Class", depends_on=("a",))]):
        with pytest.raises(ValueError):
            ordered_stages(graph)


def test_fresh_dependency_ids_do_not_depend_on_attempt_paths(tmp_path):
    pytest.importorskip("pyarrow", reason="Work plans require lerobot[dataset]")
    from lerobot.annotations.runtime.configs import RuntimeConfig
    from lerobot.annotations.runtime.pipeline import run_pipeline
    from lerobot.annotations.runtime.types import DatasetRef, InputItem

    class Source:
        dataset_ref = DatasetRef("fixture/source", "a" * 40)

        def discover(self, stage, store, upstream):
            for index in range(6):
                yield InputItem(str(index), {"value": index})

    stages = [
        StageConfig("a", "tests.annotations.runtime._modules:Echo"),
        StageConfig("b", "tests.annotations.runtime._modules:Echo", depends_on=("a",)),
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
    from lerobot.annotations.runtime.artifacts import ArtifactStore
    from lerobot.annotations.runtime.configs import RuntimeConfig
    from lerobot.annotations.runtime.pipeline import run_pipeline
    from lerobot.annotations.runtime.types import DatasetRef, InputItem

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
        [StageConfig("echo", "tests.annotations.runtime._modules:Echo")],
        RuntimeConfig(run_uri=str(tmp_path / "run")),
    )
    assert completed["echo"][1].completed == 1
