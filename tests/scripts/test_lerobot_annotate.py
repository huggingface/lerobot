# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")


def test_plan_does_not_publish(tmp_path, monkeypatch):
    from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig
    from lerobot.annotations.steerable_pipeline.executor import PipelineRunSummary
    from lerobot.annotations.steerable_pipeline.validator import ValidationReport
    from lerobot.scripts import lerobot_annotate

    cfg = AnnotationPipelineConfig(root=tmp_path, push_to_hub=True, new_repo_id="processed/example")
    cfg.runtime.mode = "plan"
    monkeypatch.setattr(
        lerobot_annotate,
        "run_annotation_pipeline",
        lambda *args: PipelineRunSummary([], [], ValidationReport()),
    )
    monkeypatch.setattr(
        lerobot_annotate, "_push_to_hub", lambda *a, **k: pytest.fail("Planning uploaded a dataset")
    )
    lerobot_annotate.annotate.__wrapped__(cfg)


def test_hub_working_copy_preserves_snapshot_and_reuses_revision(tmp_path, monkeypatch):
    from lerobot.annotations.steerable_pipeline.config import AnnotationPipelineConfig
    from lerobot.scripts import lerobot_annotate

    source = tmp_path / ("a" * 40)
    (source / "videos").mkdir(parents=True)
    (source / "meta").mkdir()
    (source / "meta/info.json").write_text("original")
    monkeypatch.setattr(lerobot_annotate, "snapshot_download", lambda **kwargs: str(source))
    cfg = AnnotationPipelineConfig(repo_id="owner/example")
    cfg.runtime.run_uri = str(tmp_path / "runs")
    root = lerobot_annotate._resolve_root(cfg)
    (root / "meta/info.json").write_text("edited")
    assert (source / "meta/info.json").read_text() == "original"
    assert (root / "videos").resolve() == (source / "videos").resolve()
    assert cfg.revision == "a" * 40
    assert lerobot_annotate._resolve_root(cfg) == root


def test_changed_paths_include_ownership_but_not_media(tmp_path):
    from lerobot.scripts.lerobot_annotate import _changed_paths

    data = tmp_path / "data/chunk-000/file-000.parquet"
    owner = tmp_path / "meta/annotations/ownership/chunk-000/file-000.parquet"
    owner.parent.mkdir(parents=True)
    owner.write_bytes(b"owned")
    assert _changed_paths(tmp_path, SimpleNamespace(written_paths=[data])) == [
        data,
        tmp_path / "meta/info.json",
        owner,
    ]
    quality = tmp_path / "meta/annotations/quality/episode-0.parquet"
    windows = tmp_path / "meta/annotations/windows/episode-0.parquet"
    assert _changed_paths(
        tmp_path, SimpleNamespace(written_paths=[data], metadata_paths=[quality, windows])
    ) == [
        data,
        quality,
        windows,
        tmp_path / "meta/info.json",
        owner,
    ]
